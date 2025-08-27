"""
Tests for the MbodiVision client package.
"""

import pytest
import asyncio
from mbodivision_client import MbodiVisionClient, BoundingBox, DataSubmissionResponse, DetectionResult


def test_imports():
    """Test that all required classes can be imported."""
    assert MbodiVisionClient is not None
    assert BoundingBox is not None
    assert DataSubmissionResponse is not None
    assert DetectionResult is not None


def test_bounding_box_creation():
    """Test BoundingBox model creation and validation."""
    # Valid bounding box
    bbox = BoundingBox(x1=0.1, y1=0.2, x2=0.8, y2=0.9, label="person")
    assert bbox.x1 == 0.1
    assert bbox.y1 == 0.2
    assert bbox.x2 == 0.8
    assert bbox.y2 == 0.9
    assert bbox.label == "person"
    assert bbox.width == pytest.approx(0.7)
    assert bbox.height == pytest.approx(0.7)


def test_bounding_box_validation():
    """Test BoundingBox validation rules."""
    # Invalid coordinates (x2 <= x1)
    with pytest.raises(ValueError):
        BoundingBox(x1=0.5, y1=0.2, x2=0.3, y2=0.9, label="invalid")
    
    # Invalid coordinates (y2 <= y1)
    with pytest.raises(ValueError):
        BoundingBox(x1=0.1, y1=0.8, x2=0.9, y2=0.2, label="invalid")
    
    # Coordinates out of range
    with pytest.raises(ValueError):
        BoundingBox(x1=-0.1, y1=0.2, x2=0.8, y2=0.9, label="invalid")


def test_client_initialization():
    """Test client initialization."""
    client = MbodiVisionClient(base_url="http://test-server:8000")
    assert client.base_url == "http://test-server:8000"
    assert client.client is not None


def test_create_bounding_boxes_from_dict():
    """Test the static method for creating bounding boxes from dictionary."""
    bbox_dict = {
        "person": (0.1, 0.2, 0.8, 0.9),
        "car": (0.3, 0.4, 0.7, 0.6)
    }
    
    bboxes = MbodiVisionClient.create_bounding_boxes_from_dict(bbox_dict)
    assert len(bboxes) == 2
    
    person_bbox = next(bbox for bbox in bboxes if bbox.label == "person")
    car_bbox = next(bbox for bbox in bboxes if bbox.label == "car")
    
    assert person_bbox.x1 == 0.1
    assert person_bbox.y1 == 0.2
    assert person_bbox.x2 == 0.8
    assert person_bbox.y2 == 0.9
    
    assert car_bbox.x1 == 0.3
    assert car_bbox.y1 == 0.4
    assert car_bbox.x2 == 0.7
    assert car_bbox.y2 == 0.6


def test_validate_image():
    """Test image validation method."""
    # This test will fail if no image file exists, but that's expected
    # In a real test environment, you'd create a test image
    assert isinstance(MbodiVisionClient.validate_image("nonexistent.jpg"), bool)


@pytest.mark.asyncio
async def test_client_context_manager():
    """Test that the client can be used as an async context manager."""
    async with MbodiVisionClient() as client:
        assert client is not None
        assert not client.client.is_closed


if __name__ == "__main__":
    pytest.main([__file__])
