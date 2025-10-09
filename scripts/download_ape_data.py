#!/usr/bin/env python3
"""
Download APE-data from Hugging Face

Downloads the t2ance/APE-data dataset from Hugging Face Hub.
Requires authentication (gated dataset).
"""

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, login, whoami
from huggingface_hub.utils import HfHubHTTPError


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download APE-data from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download using cached login
  python download_ape_data.py --output-dir ./data/ape-raw

  # Download with explicit token
  python download_ape_data.py --token hf_xxxxx --output-dir ./data/ape-raw

  # Download only APE subset
  python download_ape_data.py --subset APE

  # Force re-download
  python download_ape_data.py --force
        """
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/ape-raw",
        help="Directory to download dataset (default: ./data/ape-raw)"
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["APE", "non APE", "both"],
        default="both",
        help="Which subset to download (default: both)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (optional, uses cached login if not provided)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use HF cache directory instead of local_dir (faster but uses more disk)"
    )

    return parser.parse_args()


def check_authentication(token=None):
    """Check if user is authenticated with Hugging Face."""
    try:
        if token:
            whoami(token=token)
            print("✓ Authentication successful with provided token")
            return token
        else:
            user_info = whoami()
            print(f"✓ Already logged in as: {user_info['name']}")
            return True
    except Exception as e:
        print("✗ Not authenticated with Hugging Face")
        print("\nTo download this gated dataset, you need to:")
        print("  1. Create a Hugging Face account: https://huggingface.co/join")
        print("  2. Request access to the dataset: https://huggingface.co/datasets/t2ance/APE-data")
        print("  3. Login via CLI: huggingface-cli login")
        print("\nOr provide a token:")
        print("  python download_ape_data.py --token hf_xxxxx")
        return None


def download_dataset(
    repo_id="t2ance/APE-data",
    output_dir="./data/ape-raw",
    subset="both",
    token=None,
    force=False,
    use_cache=False
):
    """
    Download APE-data from Hugging Face.

    Args:
        repo_id: HuggingFace dataset repository ID
        output_dir: Local directory to save dataset
        subset: Which subset to download ("APE", "non APE", or "both")
        token: HuggingFace authentication token
        force: Force re-download even if exists
        use_cache: Use HF cache instead of local_dir

    Returns:
        Path to downloaded dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set up download patterns based on subset
    if subset == "APE":
        allow_patterns = ["APE/*", "README.md", ".gitattributes"]
    elif subset == "non APE":
        allow_patterns = ["non APE/*", "README.md", ".gitattributes"]
    else:  # both
        allow_patterns = None

    print(f"\n{'='*70}")
    print(f"Downloading APE-data from Hugging Face")
    print(f"{'='*70}")
    print(f"Repository: {repo_id}")
    print(f"Subset: {subset}")
    print(f"Output: {output_path.absolute()}")
    print(f"Force download: {force}")
    print(f"{'='*70}\n")

    try:
        if use_cache:
            # Download to HF cache (symlinks)
            print("Downloading to HuggingFace cache...")
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                allow_patterns=allow_patterns,
                force_download=force,
                resume_download=True,
            )
        else:
            # Download directly to local_dir
            print("Downloading to local directory...")
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=output_path,
                token=token,
                allow_patterns=allow_patterns,
                force_download=force,
                resume_download=True,
                local_dir_use_symlinks=False,
            )

        print(f"\n✓ Download complete!")
        print(f"Dataset location: {downloaded_path}")

        # Verify download
        downloaded_path_obj = Path(downloaded_path)
        if subset in ["APE", "both"]:
            ape_dir = downloaded_path_obj / "APE"
            if ape_dir.exists():
                zip_count = len(list(ape_dir.glob("*.zip")))
                print(f"  APE subset: {zip_count} ZIP files")
            else:
                print(f"  ⚠ Warning: APE directory not found")

        if subset in ["non APE", "both"]:
            non_ape_dir = downloaded_path_obj / "non APE"
            if non_ape_dir.exists():
                zip_count = len(list(non_ape_dir.glob("*.zip")))
                print(f"  non APE subset: {zip_count} ZIP files")
            else:
                print(f"  ⚠ Warning: non APE directory not found")

        return downloaded_path

    except HfHubHTTPError as e:
        if "401" in str(e) or "403" in str(e):
            print("\n✗ Access denied!")
            print("\nThis dataset requires access approval:")
            print("  1. Visit: https://huggingface.co/datasets/t2ance/APE-data")
            print("  2. Click 'Request access to this repo'")
            print("  3. Wait for approval from dataset owner")
            print("  4. Login: huggingface-cli login")
            print("  5. Try downloading again")
        else:
            print(f"\n✗ Download failed: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        sys.exit(1)


def main():
    args = parse_args()

    # Check authentication
    auth_result = check_authentication(args.token)
    if auth_result is None:
        sys.exit(1)

    # Use token if provided, otherwise rely on cached login
    token = args.token if args.token else True

    # Download dataset
    downloaded_path = download_dataset(
        output_dir=args.output_dir,
        subset=args.subset,
        token=token,
        force=args.force,
        use_cache=args.use_cache
    )

    print(f"\n{'='*70}")
    print("Next steps:")
    print(f"{'='*70}")
    print("1. Convert to NIfTI:")
    print(f"   python scripts/convert_ape_data.py \\")
    print(f"       --ape-cache-dir {downloaded_path} \\")
    print(f"       --subset APE \\")
    print(f"       --output-dir ./data/ape-nifti")
    print()
    print("2. Train model:")
    print("   ./run_custom_training.sh")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
