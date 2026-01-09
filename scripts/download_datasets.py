#!/usr/bin/env python3
"""
Dataset downloader for multi-camera tracking evaluation.
Downloads and prepares PETS, EPFL, and WILDTRACK datasets.
"""

import os
import sys
import shutil
import tarfile
import zipfile
import argparse
import urllib.request
from pathlib import Path
from tqdm import tqdm

# Dataset URLs and info
DATASETS = {
    'pets2009': {
        'description': 'PETS 2009 Benchmark Data - Multi-camera pedestrian tracking',
        'url': 'http://www.cvg.reading.ac.uk/PETS2009/a_data/dev/PETS2009-S2L1.tar.bz2',
        'size': '~200MB',
        'format': 'tar.bz2'
    },
    'epfl': {
        'description': 'EPFL Multi-Camera Pedestrian Dataset',
        'urls': {
            'terrace': 'https://cvlab.epfl.ch/data/pom/terrace.tar.gz',
            'basketball': 'https://cvlab.epfl.ch/data/pom/basketball.tar.gz',
        },
        'size': '~1GB per sequence',
        'format': 'tar.gz'
    },
    'wildtrack': {
        'description': 'WILDTRACK Seven-Camera HD Dataset',
        'url': 'https://drive.google.com/uc?export=download&id=1b30X9pEZzxPL-R3JTwzz_-68SG-cPMM3',
        'size': '~4GB',
        'format': 'zip',
        'requires_gdrive': True
    },
    'mot17': {
        'description': 'MOT17 Challenge Dataset (single-camera but useful for tracking eval)',
        'url': 'https://motchallenge.net/data/MOT17.zip',
        'size': '~5GB',
        'format': 'zip'
    }
}


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str, desc: str = "Downloading"):
    """Download a file with progress bar."""
    print(f"\n{desc}...")
    print(f"URL: {url}")
    print(f"Output: {output_path}")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    
    print(f"Downloaded: {output_path}")


def extract_archive(archive_path: str, output_dir: str):
    """Extract tar/zip archive."""
    print(f"Extracting {archive_path} to {output_dir}...")
    
    if archive_path.endswith('.tar.gz') or archive_path.endswith('.tar.bz2'):
        mode = 'r:gz' if archive_path.endswith('.gz') else 'r:bz2'
        with tarfile.open(archive_path, mode) as tar:
            tar.extractall(output_dir)
    elif archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    else:
        print(f"Unknown archive format: {archive_path}")
        return
    
    print(f"Extracted to: {output_dir}")


def download_with_gdrive(file_id: str, output_path: str):
    """Download from Google Drive using gdown."""
    try:
        import gdown
        gdown.download(id=file_id, output=output_path, quiet=False)
    except ImportError:
        print("gdown is required for Google Drive downloads.")
        print("Install with: pip install gdown")
        sys.exit(1)


def download_pets(data_dir: Path):
    """Download PETS 2009 dataset."""
    dataset = DATASETS['pets2009']
    output_dir = data_dir / 'pets2009'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    archive_path = output_dir / 'PETS2009-S2L1.tar.bz2'
    
    if not archive_path.exists():
        download_file(dataset['url'], str(archive_path), "Downloading PETS 2009")
    else:
        print(f"Archive already exists: {archive_path}")
    
    # Extract
    extract_archive(str(archive_path), str(output_dir))
    
    print(f"PETS 2009 ready at: {output_dir}")


def download_epfl(data_dir: Path, sequence: str = 'terrace'):
    """Download EPFL dataset."""
    dataset = DATASETS['epfl']
    output_dir = data_dir / 'epfl'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if sequence not in dataset['urls']:
        print(f"Unknown EPFL sequence: {sequence}")
        print(f"Available: {list(dataset['urls'].keys())}")
        return
    
    url = dataset['urls'][sequence]
    archive_path = output_dir / f'{sequence}.tar.gz'
    
    if not archive_path.exists():
        download_file(url, str(archive_path), f"Downloading EPFL {sequence}")
    else:
        print(f"Archive already exists: {archive_path}")
    
    # Extract
    extract_archive(str(archive_path), str(output_dir))
    
    print(f"EPFL {sequence} ready at: {output_dir}")


def download_wildtrack(data_dir: Path):
    """Download WILDTRACK dataset."""
    dataset = DATASETS['wildtrack']
    output_dir = data_dir / 'wildtrack'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    archive_path = output_dir / 'wildtrack.zip'
    
    print("\nWILDTRACK dataset requires manual download or Google Drive access.")
    print("Option 1: Download manually from https://www.epfl.ch/labs/cvlab/data/data-wildtrack/")
    print("Option 2: Use gdown to download from Google Drive")
    
    if dataset.get('requires_gdrive'):
        confirm = input("\nAttempt automatic download with gdown? [y/N]: ")
        if confirm.lower() == 'y':
            # Extract file ID from URL
            file_id = '1b30X9pEZzxPL-R3JTwzz_-68SG-cPMM3'
            download_with_gdrive(file_id, str(archive_path))
            extract_archive(str(archive_path), str(output_dir))
    
    print(f"WILDTRACK directory: {output_dir}")


def download_mot17(data_dir: Path):
    """Download MOT17 dataset."""
    dataset = DATASETS['mot17']
    output_dir = data_dir / 'mot17'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    archive_path = output_dir / 'MOT17.zip'
    
    print("\nMOT17 is a large dataset (~5GB).")
    print("URL:", dataset['url'])
    print("\nYou can download manually or proceed with automatic download.")
    
    confirm = input("Proceed with download? [y/N]: ")
    if confirm.lower() != 'y':
        print("Skipping MOT17 download.")
        return
    
    if not archive_path.exists():
        download_file(dataset['url'], str(archive_path), "Downloading MOT17")
    
    extract_archive(str(archive_path), str(output_dir))
    print(f"MOT17 ready at: {output_dir}")


def list_datasets():
    """List available datasets."""
    print("\nAvailable Datasets:")
    print("=" * 60)
    for name, info in DATASETS.items():
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        print(f"  Size: {info['size']}")
        if info.get('requires_gdrive'):
            print("  Note: Requires gdown for Google Drive download")


def main():
    parser = argparse.ArgumentParser(
        description='Download datasets for multi-camera tracking evaluation'
    )
    parser.add_argument(
        '--dataset', '-d',
        choices=['pets2009', 'epfl', 'wildtrack', 'mot17', 'all'],
        help='Dataset to download'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/datasets',
        help='Directory to store datasets (default: data/datasets)'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available datasets'
    )
    parser.add_argument(
        '--epfl-sequence',
        type=str,
        default='terrace',
        choices=['terrace', 'basketball'],
        help='EPFL sequence to download'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
        return
    
    if not args.dataset:
        parser.print_help()
        print("\n\nUse --list to see available datasets")
        return
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Data directory: {data_dir.absolute()}")
    
    if args.dataset == 'all':
        download_pets(data_dir)
        download_epfl(data_dir, args.epfl_sequence)
        # download_wildtrack(data_dir)  # Requires confirmation
        # download_mot17(data_dir)  # Large, requires confirmation
    elif args.dataset == 'pets2009':
        download_pets(data_dir)
    elif args.dataset == 'epfl':
        download_epfl(data_dir, args.epfl_sequence)
    elif args.dataset == 'wildtrack':
        download_wildtrack(data_dir)
    elif args.dataset == 'mot17':
        download_mot17(data_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()

