#!/usr/bin/env python3

"""Script to extract embeddings from DICOM folders using Pillar0 models.

This script can work in two modes:
1. With a prepared data directory (train.json, manifest.csv, volumes/)
2. With a direct NIfTI file path (automatically creates the structure)

Data directory structure:
    data_dir/
        train.json           # JSON file with sample metadata: {"sample_name": "ACC123", ...}
        manifest.csv         # CSV with columns: sample_name, image_cache_path
        volumes/             # Directory containing processed volumes
            ACC123/
                metadata.json
                volume.mp4 (or .npy, .nii.gz)

See /home/lukas.folle/Code/external/rate-evals/data/rve_example for reference.

Examples:
    # Using a NIfTI file directly (easiest)
    python scripts/extract_from_dcm.py --nifti-file /path/to/scan.nii.gz --anatomy brain

    # Using the example data
    python scripts/extract_from_dcm.py --data-dir data/rve_example --anatomy abdomen

    # Basic usage - extract chest CT embeddings
    python scripts/extract_from_dcm.py --data-dir /data/chest_scans --anatomy chest

    # Manually specify paths (advanced)
    python scripts/extract_from_dcm.py --data-dir /data/scans --anatomy chest \\
        --train-json /data/scans/train.json \\
        --cache-manifest /data/scans/manifest.csv
"""

import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import tyro


@dataclass
class ExtractConfig:
    """Configuration for extracting embeddings from DICOM folders using Pillar0 models."""

    data_dir: Optional[Path] = None
    """Path to the data directory containing train.json, manifest.csv, and volumes/ subdirectory."""

    nifti_file: Optional[Path] = None
    """Path to a NIfTI file (.nii.gz) to process directly. If provided, data_dir is auto-generated."""

    anatomy: Literal["chest", "abdomen", "brain"] = "chest"
    """Anatomy type: chest, abdomen, or brain."""

    sample_name: Optional[str] = None
    """Sample name/accession ID for the NIfTI file (default: derived from filename)."""

    batch_size: int = 4
    """Batch size per GPU."""

    ct_window_type: str = "all"
    """CT window type."""

    split: Literal["train", "valid", "test"] = "train"
    """Dataset split to process."""

    output_dir: Optional[Path] = None
    """Output directory for embeddings (default: cache/pillar0_rve_<anatomy>_ct)."""

    train_json: Optional[Path] = None
    """Path to train.json file (default: data_dir/train.json)."""

    cache_manifest: Optional[Path] = None
    """Path to manifest.csv file (default: data_dir/manifest.csv)."""

    num_gpus: Optional[int] = None
    """Number of GPUs to use (default: all available)."""


def setup_nifti_data_structure(
    nifti_file: Path, anatomy: str, sample_name: Optional[str] = None
) -> Path:
    """Create a temporary data directory structure for a NIfTI file.

    Args:
        nifti_file: Path to the NIfTI file
        anatomy: Anatomy type (brain, chest, abdomen)
        sample_name: Optional sample name (default: filename without extension)

    Returns:
        Path to the created data directory
    """
    import nibabel as nib

    if not nifti_file.exists():
        print(f"Error: NIfTI file does not exist: {nifti_file}", file=sys.stderr)
        sys.exit(1)

    # Derive sample name from filename if not provided
    if sample_name is None:
        sample_name = nifti_file.stem
        if sample_name.endswith(".nii"):
            sample_name = sample_name[:-4]

    # Create temporary directory structure
    temp_dir = Path(tempfile.mkdtemp(prefix=f"rate_extract_{sample_name}_"))
    volumes_dir = temp_dir / "volumes" / sample_name
    volumes_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating temporary data structure at: {temp_dir}")

    # Load NIfTI file and convert to numpy array
    print(f"Loading NIfTI file: {nifti_file}")
    nifti_img = nib.load(str(nifti_file))
    volume_data = nifti_img.get_fdata()

    print(f"NIfTI shape: {volume_data.shape}, dtype: {volume_data.dtype}")

    # Convert to numpy array and save as .npy
    # RVE expects numpy format
    volume_npy = volumes_dir / "volume.npy"
    np.save(volume_npy, volume_data.astype(np.float32))
    print(f"Converted and saved volume to: {volume_npy}")

    # Get spacing information if available
    try:
        spacing = nifti_img.header.get_zooms()
        original_spacing = [float(s) for s in spacing[:3]]
    except Exception:
        original_spacing = [1.0, 1.0, 1.0]

    # Create metadata.json
    metadata = {
        "series_info": {
            "accession": sample_name,
            "series_number": 1,
            "series_uid": f"{sample_name}.nii.gz",
            "series_description": f"{anatomy.capitalize()} CT",
            "modality": "CT",
            "slice_count": int(volume_data.shape[0]),
        },
        "processing_metadata": {
            "modality": "CT",
            "anatomy": anatomy,
            "original_spacing": original_spacing,
            "original_shape": [int(s) for s in volume_data.shape],
            "nifti_source": str(nifti_file),
        },
    }

    metadata_file = volumes_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Created metadata.json at: {metadata_file}")

    # Create train.json
    train_data = {
        "sample_name": sample_name,
        "nii_path": str(nifti_file),
        "report_metadata": f"{anatomy.capitalize()} CT scan",
    }

    train_json = temp_dir / "train.json"
    with open(train_json, "w") as f:
        json.dump(train_data, f)
    print(f"Created train.json at: {train_json}")

    # Create manifest.csv
    manifest_csv = temp_dir / "manifest.csv"
    with open(manifest_csv, "w") as f:
        f.write("sample_name,image_cache_path\n")
        f.write(f"{sample_name},{volumes_dir}\n")
    print(f"Created manifest.csv at: {manifest_csv}")

    print("Data structure setup complete!\n")

    return temp_dir


def main(config: ExtractConfig) -> None:
    """Extract embeddings from DICOM folders using Pillar0 models."""

    # Validate input: either data_dir or nifti_file must be provided
    if config.data_dir is None and config.nifti_file is None:
        print(
            "Error: Either --config.data-dir or --config.nifti-file must be provided",
            file=sys.stderr,
        )
        sys.exit(1)

    if config.data_dir is not None and config.nifti_file is not None:
        print(
            "Error: Cannot specify both --config.data-dir and --config.nifti-file", file=sys.stderr
        )
        sys.exit(1)

    # Track if we created a temporary directory for cleanup
    temp_dir_created = None

    try:
        # If nifti_file is provided, create temporary data structure
        if config.nifti_file is not None:
            nifti_file = config.nifti_file.resolve()
            print(f"NIfTI file mode: {nifti_file}")

            # Map anatomy for consistency
            anatomy = config.anatomy
            if anatomy in ["abdomen", "abd"]:
                anatomy = "abdomen"

            # Create temporary data structure
            data_dir = setup_nifti_data_structure(nifti_file, anatomy, config.sample_name)
            temp_dir_created = data_dir
        else:
            # Validate data directory
            data_dir = config.data_dir
            if not data_dir.exists():
                print(f"Error: Data directory does not exist: {data_dir}", file=sys.stderr)
                sys.exit(1)
            if not data_dir.is_dir():
                print(f"Error: {data_dir} is not a directory", file=sys.stderr)
                sys.exit(1)

            # Get absolute path
            data_dir = data_dir.resolve()

        # Map anatomy to dataset and model repo
        anatomy = config.anatomy
        if anatomy in ["abdomen", "abd"]:
            anatomy = "abdomen"
            dataset = "rve_abd_ct"
            model_repo_id = "YalaLab/Pillar0-AbdomenCT"
        elif anatomy == "chest":
            dataset = "rve_chest_ct"
            model_repo_id = "YalaLab/Pillar0-ChestCT"
        elif anatomy == "brain":
            dataset = "rve_brain_ct"
            model_repo_id = "YalaLab/Pillar0-HeadCT"
        else:
            print(
                f"Error: Invalid anatomy type '{anatomy}'. Must be chest, abdomen, or brain",
                file=sys.stderr,
            )
            sys.exit(1)

        # Set default output directory
        output_dir = config.output_dir
        if output_dir is None:
            output_dir = Path(f"cache/pillar0_{dataset}")
        output_dir = output_dir.resolve()

        # Auto-detect or validate train.json
        train_json = config.train_json
        if train_json is None:
            train_json = data_dir / "train.json"
            if train_json.exists():
                print(f"Found train.json at: {train_json}")
            else:
                print(f"Warning: train.json not found at {train_json}")
                print("You may need to create a train.json file with the following structure:")
                print(
                    '  {"sample_name": "EXAMPLE_ACCESSION", "nii_path": null, "report_metadata": "..."}'
                )
                print()
        else:
            train_json = train_json.resolve()

        # Auto-detect or validate cache_manifest
        cache_manifest = config.cache_manifest
        if cache_manifest is None:
            cache_manifest = data_dir / "manifest.csv"
            if cache_manifest.exists():
                print(f"Found manifest.csv at: {cache_manifest}")
            else:
                print(f"Warning: manifest.csv not found at {cache_manifest}")
                print(
                    "You may need to create a manifest.csv file with columns: sample_name, image_cache_path"
                )
                print()
        else:
            cache_manifest = cache_manifest.resolve()

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build command-line arguments for rate-extract CLI
        cli_args = [
            "--model",
            "pillar0",
            "--dataset",
            dataset,
            "--split",
            config.split,
            "--batch-size",
            str(config.batch_size),
            "--model-repo-id",
            model_repo_id,
            "--ct-window-type",
            config.ct_window_type,
            "--output-dir",
            str(output_dir),
        ]

        # Add optional num-gpus
        if config.num_gpus is not None:
            cli_args.extend(["--num-gpus", str(config.num_gpus)])

        # Add data overrides
        cli_args.extend(
            [
                f"data.train_json={train_json}",
                f"data.cache_manifest={cache_manifest}",
            ]
        )

        # Print configuration
        print("=" * 60)
        print("Pillar0 Embedding Extraction")
        print("=" * 60)
        print(f"Data Directory:  {data_dir}")
        if config.nifti_file:
            print(f"NIfTI File:      {config.nifti_file}")
        print(f"Anatomy Type:    {anatomy}")
        print(f"Dataset:         {dataset}")
        print(f"Model Repo:      {model_repo_id}")
        print(f"Split:           {config.split}")
        print(f"Batch Size:      {config.batch_size}")
        print(f"CT Window Type:  {config.ct_window_type}")
        print(f"Output Dir:      {output_dir}")
        print(f"Train JSON:      {train_json}")
        print(f"Cache Manifest:  {cache_manifest}")
        if config.num_gpus is not None:
            print(f"Num GPUs:        {config.num_gpus}")
        print("=" * 60)
        print()

        # Check if data files exist
        if not train_json.exists():
            print(f"ERROR: train.json not found at: {train_json}", file=sys.stderr)
            print(
                "Please create this file or specify --train-json with a valid path", file=sys.stderr
            )
            sys.exit(1)

        if not cache_manifest.exists():
            print(f"ERROR: manifest.csv not found at: {cache_manifest}", file=sys.stderr)
            print(
                "Please create this file or specify --cache-manifest with a valid path",
                file=sys.stderr,
            )
            sys.exit(1)

        # Run the extraction by directly calling the CLI function
        print("Running extraction...")
        print(f"Arguments: {' '.join(cli_args)}")
        print()

        # Import the extraction CLI
        from rate_eval.cli.extract import main as extract_main

        # Temporarily replace sys.argv with our arguments
        original_argv = sys.argv
        try:
            sys.argv = ["rate-extract"] + cli_args
            extract_main()
        except SystemExit as e:
            # Capture sys.exit() calls from the extraction CLI
            if e.code != 0:
                print(f"\nError: Extraction failed with exit code {e.code}", file=sys.stderr)
                sys.exit(e.code)
        except KeyboardInterrupt:
            print("\nExtraction interrupted by user", file=sys.stderr)
            sys.exit(130)
        finally:
            sys.argv = original_argv

        print()
        print("=" * 60)
        print("Extraction complete!")
        print(f"Embeddings saved to: {output_dir}")
        print("=" * 60)

    finally:
        # Cleanup temporary directory if created
        if temp_dir_created is not None:
            print(f"\nCleaning up temporary directory: {temp_dir_created}")
            shutil.rmtree(temp_dir_created, ignore_errors=True)


if __name__ == "__main__":
    tyro.cli(main)
