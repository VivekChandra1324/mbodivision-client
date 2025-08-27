#!/usr/bin/env python3
"""
Setup script for mbodivision-client package.
"""

from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mbodivision-client",
    version="1.0.0",
    author="MbodiVision",
    author_email="info@example.com",
    description="A Python client for interacting with the MbodiVision computer vision API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mbodivision/mbodivision-client",
    packages=find_packages(include=["client*"]),

    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.1",
            "black>=23.7.0",
            "isort>=5.12.0",
            "mypy>=1.5.1",
            "flake8>=6.0.0",
        ],
    },
    
    project_urls={
        "Repository": "https://github.com/VivekChandra1324/mbodivision-client.git",
    },
)
