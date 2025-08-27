"""
Module to convert stored images and annotations into YOLO-compatible datasets.
Generates YOLO YAML configuration and organizes images/labels with a train/validation split.
This version operates entirely in-memory and streams directly to cloud storage.
"""

import asyncio
import logging
import random
from typing import Dict, Optional, Tuple, List

import yaml

from app.cloud_storage import CloudStorage

logger = logging.getLogger(__name__)


class YOLODatasetConverter:
    """
    Converts a given batch of image data into a YOLO dataset format directly
    in Google Cloud Storage, without using local storage.
    """

    def __init__(self, storage: CloudStorage, train_split: float = 0.7):
        """
        Initializes the converter.

        Args:
            storage: An instance of the CloudStorage client.
            train_split: The fraction of data to use for the training set (e.g., 0.7 for 70%).
        """
        self.storage = storage
        self.train_split = train_split

    async def prepare_dataset(
        self, dataset_name: str, submissions_to_process: List[Dict]
    ) -> Tuple[Optional[str], Dict]:
        """
        Takes a specific batch of submissions and creates a complete YOLO dataset in GCS.

        Args:
            dataset_name: A unique name for the new dataset folder in GCS.
            submissions_to_process: The list of submission data for this specific batch.

        Returns:
            A tuple containing the GCS path to the generated YAML file and a dictionary of stats.
        """
        all_data = submissions_to_process
        if not all_data:
            logger.warning("No images were provided to create a dataset.")
            return None, {}

        # 1. Create a mapping of class names to integer IDs
        class_names_list = sorted(list({bbox["label"] for item in all_data for bbox in item.get("bounding_boxes", [])}))
        name_to_id = {name: i for i, name in enumerate(class_names_list)}

        # 2. Shuffle and split the data into training and validation sets
        random.seed(42) # for reproducible splits
        random.shuffle(all_data)
        train_count = int(len(all_data) * self.train_split)
        logger.info(f"Dataset split: {train_count} training, {len(all_data) - train_count} validation images")

        # 3. Concurrently process each item and upload its image/label pair to GCS
        logger.info(f"Streaming dataset '{dataset_name}' directly to GCS...")
        tasks = [
            self._process_and_upload_item(item, idx, train_count, name_to_id, dataset_name)
            for idx, item in enumerate(all_data)
        ]
        await asyncio.gather(*tasks)

        # 4. Generate and upload the final dataset.yaml configuration file
        gcs_dataset_root = f"gs://{self.storage.gcs_bucket_name}/datasets/{dataset_name}"
        yaml_content = self._generate_yaml_config(gcs_dataset_root, class_names_list)
        gcs_yaml_path = f"datasets/{dataset_name}/{dataset_name}.yaml"
        
        await self.storage.upload_file_from_bytes_async(
            gcs_path=gcs_yaml_path,
            file_bytes=yaml_content.encode("utf-8"),
            content_type="application/x-yaml",
        )
        
        final_gcs_yaml_path = f"{gcs_dataset_root}/{dataset_name}.yaml"
        logger.info(f"Cloud-only dataset created. YAML path for training: {final_gcs_yaml_path}")
        
        stats = {
            "total_images": len(all_data),
            "train_images": train_count,
            "val_images": len(all_data) - train_count,
            "gcs_yaml_path": final_gcs_yaml_path,
        }
        
        return final_gcs_yaml_path, stats

    async def _process_and_upload_item(self, item: Dict, idx: int, train_count: int, name_to_id: Dict, dataset_name: str):
        """Downloads, processes, and uploads a single image and its label file."""
        img_bytes = await self.storage.download_file_async(item["image_path"])
        if not img_bytes:
            logger.warning(f"Skipping submission {item['submission_id']} as image could not be downloaded.")
            return

        subfolder = "train" if idx < train_count else "val"
        submission_id = item["submission_id"]

        # Upload the image file
        gcs_image_path = f"datasets/{dataset_name}/images/{subfolder}/{submission_id}.jpg"
        await self.storage.upload_file_from_bytes_async(
            gcs_path=gcs_image_path, file_bytes=img_bytes, content_type="image/jpeg"
        )

        # Create the YOLO label content
        yolo_lines = []
        for bbox in item.get("bounding_boxes", []):
            class_id = name_to_id.get(bbox["label"])
            if class_id is None: continue # Skip if label is not found

            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # Upload the label file
        label_content = "".join(yolo_lines)
        gcs_label_path = f"datasets/{dataset_name}/labels/{subfolder}/{submission_id}.txt"
        await self.storage.upload_file_from_bytes_async(
            gcs_path=gcs_label_path,
            file_bytes=label_content.encode("utf-8"),
            content_type="text/plain",
        )

    def _generate_yaml_config(self, gcs_dataset_root: str, class_names: List[str]) -> str:
        """Generates the content for the dataset.yaml file."""
        yaml_data = {
            'path': gcs_dataset_root,
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(class_names),
            'names': {i: name for i, name in enumerate(class_names)},
        }
        return yaml.dump(yaml_data, sort_keys=False)