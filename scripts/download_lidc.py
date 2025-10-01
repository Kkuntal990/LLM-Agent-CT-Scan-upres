#!/usr/bin/env python3
"""Download LIDC-IDRI dataset from TCIA.

This script downloads a subset of LIDC-IDRI dataset using the TCIA REST API.
For full dataset download, use the NBIA Data Retriever application.

Note: LIDC-IDRI is ~124 GB. This script downloads a configurable subset.
"""

import argparse
import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import shutil


# TCIA API endpoints
TCIA_BASE_URL = "https://services.cancerimagingarchive.net/services/v4/TCIA/query"


def download_file(url: str, output_path: Path, desc: str = "Downloading"):
    """Download file with progress bar.

    Args:
        url: URL to download from
        output_path: Where to save the file
        desc: Description for progress bar
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f, tqdm(
        desc=desc,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def get_patient_list(collection: str = "LIDC-IDRI"):
    """Get list of patient IDs from TCIA.

    Args:
        collection: Collection name (default: LIDC-IDRI)

    Returns:
        List of patient IDs
    """
    url = f"{TCIA_BASE_URL}/getPatient"
    params = {"Collection": collection, "format": "json"}

    print(f"Fetching patient list from TCIA...")
    response = requests.get(url, params=params)
    response.raise_for_status()

    patients = response.json()
    patient_ids = [p["PatientID"] for p in patients]

    print(f"Found {len(patient_ids)} patients in {collection}")
    return patient_ids


def download_patient_images(patient_id: str, output_dir: Path):
    """Download all images for a patient.

    Args:
        patient_id: Patient ID to download
        output_dir: Output directory
    """
    url = f"{TCIA_BASE_URL}/getImage"
    params = {
        "PatientID": patient_id,
        "format": "zip"
    }

    zip_path = output_dir / f"{patient_id}.zip"
    patient_dir = output_dir / patient_id

    try:
        # Download ZIP
        print(f"Downloading {patient_id}...")
        download_file(
            f"{url}?PatientID={patient_id}",
            zip_path,
            desc=f"{patient_id}"
        )

        # Extract
        print(f"Extracting {patient_id}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(patient_dir)

        # Remove ZIP
        zip_path.unlink()

        print(f"✓ Downloaded {patient_id}")
        return True

    except Exception as e:
        print(f"✗ Failed to download {patient_id}: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download LIDC-IDRI dataset from TCIA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download first 10 patients
  python scripts/download_lidc.py --num-patients 10

  # Download specific patients
  python scripts/download_lidc.py --patient-ids LIDC-IDRI-0001 LIDC-IDRI-0002

Note:
  - Full dataset is ~124 GB (1018 patients)
  - Each patient is ~100-200 MB
  - For full download, use NBIA Data Retriever:
    https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images
        """
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='/Users/kuntalkokate/Desktop/LLM Agent - CT scan upres/data/unprocessed',
        help='Output directory for DICOM files'
    )
    parser.add_argument(
        '--num-patients',
        type=int,
        default=10,
        help='Number of patients to download (default: 10)'
    )
    parser.add_argument(
        '--patient-ids',
        nargs='+',
        help='Specific patient IDs to download'
    )
    parser.add_argument(
        '--start-index',
        type=int,
        default=0,
        help='Start index for patient list (default: 0)'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LIDC-IDRI Dataset Downloader")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print()

    # Warning about dataset size
    estimated_size_gb = args.num_patients * 0.15  # ~150 MB per patient
    print(f"⚠️  Estimated download size: ~{estimated_size_gb:.1f} GB")
    print()

    # Get patient list
    try:
        if args.patient_ids:
            patient_ids = args.patient_ids
            print(f"Downloading {len(patient_ids)} specific patients")
        else:
            all_patients = get_patient_list()
            start = args.start_index
            end = start + args.num_patients
            patient_ids = all_patients[start:end]
            print(f"Downloading patients {start+1}-{end} of {len(all_patients)}")
    except Exception as e:
        print(f"✗ Error fetching patient list: {e}")
        print()
        print("Alternative: Use NBIA Data Retriever for reliable downloads:")
        print("  1. Visit: https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images")
        print("  2. Download NBIA Data Retriever for macOS")
        print("  3. Search for 'LIDC-IDRI' collection")
        print("  4. Download to:", output_dir)
        sys.exit(1)

    print("=" * 70)
    print()

    # Download each patient
    successful = 0
    failed = 0

    for i, patient_id in enumerate(patient_ids, 1):
        print(f"\n[{i}/{len(patient_ids)}] Processing {patient_id}")

        if download_patient_images(patient_id, output_dir):
            successful += 1
        else:
            failed += 1

    # Summary
    print()
    print("=" * 70)
    print("Download Summary")
    print("=" * 70)
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    print(f"Total:      {len(patient_ids)}")
    print()
    print(f"Data saved to: {output_dir}")
    print("=" * 70)
    print()

    if successful > 0:
        print("Next steps:")
        print("1. Convert DICOM to NIfTI:")
        print(f"   python scripts/prepare_dataset.py \\")
        print(f"     --dicom-root {output_dir} \\")
        print(f"     --output-dir ./data/processed")
        print()


if __name__ == '__main__':
    main()