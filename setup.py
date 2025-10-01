"""Setup script for CT super-resolution package."""

from setuptools import setup, find_packages

setup(
    name="ct-superres",
    version="0.1.0",
    description="CT Through-Plane Super-Resolution for macOS Apple Silicon",
    author="Kuntal Kokate",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.2.0",
        "numpy>=1.24",
        "scipy>=1.10",
        "nibabel>=5.0",
        "pydicom>=2.3",
        "scikit-image>=0.20",
        "tqdm>=4.65",
        "pyyaml>=6.0",
        "tensorboard>=2.12",
        "matplotlib>=3.7",
        "pandas>=2.0",
        "lpips>=0.1.4",
        "einops>=0.7.0",
    ],
)