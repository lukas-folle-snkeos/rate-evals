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

from functools import lru_cache
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import tyro

import urllib.request

from pydicom import dcmread
from omegaconf import OmegaConf
import concurrent.futures

@dataclass
class DicomWADOURI:
    """Class with the information to create a DICOM via WADO URI."""

    patient_uid: str
    study_uid: str
    series_uid: str
    object_uid: str
    local_proxy: str = "http://localhost:55581/dicomproxy"
    wado_request: str = "wado?requestType=WADO"
    content_type: str = "contentType=application/dicom"

    def create_uri(self) -> str:
        """Create WADO URI.

        Returns
        -------
        str
            DICOM Web URI.
        """
        local_proxy_request = f"{self.local_proxy}/{self.wado_request}"
        uri = ("&").join(
            [
                local_proxy_request,
                # f"patientUID={self.patient_uid}", # some pat ids have spaces, that leads to an error
                f"studyUID={self.study_uid}",
                f"seriesUID={self.series_uid}",
                f"objectUID={self.object_uid}",
                self.content_type,
            ]
        )
        return uri

def get_dcm_from_dicom_web(
    patient_uid: str,
    study_uid: str,
    series_uid: str,
    object_uid: str,
    local_proxy: str,
    wado_request: str,
    content_type: str,
):
    """Get DICOM from DICOM Web.

    Parameters
    ----------
    patient_uid : str
        patient_uid
    study_uid : str
        study_uid
    series_uid : str
        series_uid
    object_uid : str
        object_uid, instance SOP uid
    local_proxy : str
        dicom proxy address
    wado_request : str
        wado request prefix
    content_type : str
        content type of request

    Returns
    -------
    FileDataset
        DICOM
    """
    content_dcm_url = DicomWADOURI(
        patient_uid=patient_uid,
        study_uid=study_uid,
        series_uid=series_uid,
        object_uid=object_uid,
        local_proxy=local_proxy,
        wado_request=wado_request,
        content_type=content_type,
    ).create_uri()
    if content_dcm_url.lower().startswith("http"):
        temp_cont_dcm_name, _ = urllib.request.urlretrieve(content_dcm_url)  # noqa: S310
    else:
        raise ValueError("content_dcm_url must be a http request")
    return temp_cont_dcm_name

IPP_PRIVATE_DCM_TAG = (0x0027, 0x1020)
SIZE_PRIVATE_DCM_TAG = (0x0027, 0x1010)
@lru_cache(maxsize=2)
def get_image_cached(cfg_json: str, pat_list_record_json: str, slice_set_record_json: str):
    """Get image from DICOM Proxy.

    Parameters
    ----------
    cfg : DictConfig
        Evaluation config.
    pat_list_record : DictConfig
        Patient list record.

    Returns
    -------

    """
    cfg = OmegaConf.create(json.loads(cfg_json))
    pat_list_record = OmegaConf.create(json.loads(pat_list_record_json))
    patient_uid = pat_list_record.dcmPatientID
    slice_set = OmegaConf.create(json.loads(slice_set_record_json))
    # slice_set = pat_list_record.SliceSets[0]
    study_uid = slice_set.UID.dcmStudyInstanceUID

    container_dcm = dcmread(get_dcm_from_dicom_web(
        patient_uid=patient_uid,
        study_uid=study_uid,
        series_uid=slice_set.UID.dcmSeriesInstanceUID,
        object_uid=slice_set.UID.dcmSOPInstanceUID,
        local_proxy=cfg.dicom_web_connection.local_proxy,
        wado_request=cfg.dicom_web_connection.wado_request,
        content_type=cfg.dicom_web_connection.content_type,
    ))

    # Extract series_uid once (assumed constant for all slices)
    series_uid = container_dcm[(0x0040, 0xA375)][0][(0x0008, 0x1115)][0].SeriesInstanceUID

    # Helper function to fetch a DICOM image using the web call.
    def fetch_dcm(c_seq):
        return get_dcm_from_dicom_web(
            patient_uid=patient_uid,
            study_uid=study_uid,
            series_uid=series_uid,
            object_uid=c_seq[(0x0008, 0x1199)][0].ReferencedSOPInstanceUID,
            local_proxy=cfg.dicom_web_connection.local_proxy,
            wado_request=cfg.dicom_web_connection.wado_request,
            content_type=cfg.dicom_web_connection.content_type,
        )

    # Use ThreadPoolExecutor to fetch DICOM images concurrently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        dcm_results = list(executor.map(fetch_dcm, container_dcm.ContentSequence))

    # Move dcm results to separate temp folder
    dcm_folder = Path(tempfile.mkdtemp(prefix="dcm_files_"))
    new_dcm_paths = []
    for idx, temp_path in enumerate(dcm_results):
        new_path = dcm_folder / f"slice_{idx:04d}.dcm"
        shutil.move(temp_path, new_path)
        new_dcm_paths.append(new_path)

    dcm_results = new_dcm_paths
    return dcm_folder

def patient_list_to_nifti_files_generator(patientlist_path: Path):
    """Generator that yields NIfTI files for each patient and slice_set.

    Parameters
    ----------
    patientlist_path : Path
        Path to the patient list JSON file.

    Yields
    ------
    tuple[Path, Path, Path, Path | None]
        (nifti_file_path, dcm_folder, nifti_output_folder, decompressed_dcm_folder)
    """
    for patient in json.load(open(patientlist_path))["Patients"]:
        for slice_set in patient["SliceSets"]:
            dcm_folder = get_image_cached(
                json.dumps(
                    {
                        "dicom_web_connection": {
                            "local_proxy": "http://localhost:55581/dicomproxy",
                            "wado_request": "wado?requestType=WADO",
                            "content_type": "contentType=application/dicom",
                        }
                    }
                ),
                json.dumps(patient),
                json.dumps(slice_set)
            )
            print(f"DICOM folder for patient {patient['dcmPatientID']}: {dcm_folder}")

            # If JPEG-compressed DICOMs cause issues, decompress with dcmdjpeg first
            decompressed_folder: Path | None = None
            try:
                import subprocess
                from shutil import which
                if which("dcmdjpeg") is not None:
                    decompressed_folder = Path(tempfile.mkdtemp(prefix="dcm_decompressed_"))
                    for src in dcm_folder.glob("*.dcm"):
                        dst = decompressed_folder / src.name
                        result = subprocess.run(["dcmdjpeg", str(src), str(dst)], capture_output=True, text=True)
                        if result.returncode != 0:
                            print(f"Warning: dcmdjpeg failed for {src.name}: {result.stderr.strip()}")
                            # Fallback: copy original file
                            shutil.copy2(src, dst)
                else:
                    print("dcmdjpeg not found. Install DCMTK to decompress JPEG-compressed DICOMs.")
            except Exception as e:
                print(f"Warning: decompression step encountered an error: {e}")
                decompressed_folder = None

            # Convert DICOM folder to NIfTI using the (possibly) decompressed folder
            nifti_output_folder = Path(tempfile.mkdtemp(prefix="nifti_output_"))
            # Use slice_set hash as prefix for the output files
            hash_prefix = slice_set.get("hash", "nifti")
            input_folder = decompressed_folder if decompressed_folder is not None else dcm_folder
            cmd = [
                "dcm2niix",
                "-z",
                "y",
                "-f",
                f"{hash_prefix}_%p_%s",
                "-o",
                str(nifti_output_folder),
                str(input_folder),
            ]
            subprocess.run(cmd, check=True)

            # Find the generated NIfTI file (can be any series number)
            nifti_files = list(nifti_output_folder.glob("*.nii.gz"))
            if not nifti_files:
                raise FileNotFoundError(f"No .nii.gz files found in {nifti_output_folder}")

            yield nifti_files[0], dcm_folder, nifti_output_folder, decompressed_folder


@dataclass
class ExtractConfig:
    """Configuration for extracting embeddings from DICOM folders using Pillar0 models."""

    data_dir: Optional[Path] = None
    """Path to the data directory containing train.json, manifest.csv, and volumes/ subdirectory."""

    patientlist_path: Optional[Path] = Path("/mnt/x/ML/FOUNDATION/PatientSimilarity/similarityPatientList_flat.json")
    """Path to the patient list JSON file for DICOM Web access."""

    nifti_files: Optional[List[Path]] = field(default_factory=lambda: [Path("custom_path_to_nifti_file.nii.gz")])
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
    
    # Reorder to canonical orientation for consistent axis alignment
    # This converts to RAS+ (Right-Anterior-Superior)
    nifti_img = nib.as_closest_canonical(nifti_img)
    volume_data = nifti_img.get_fdata()
    
    # In RAS+ canonical: axis0=R(right), axis1=A(anterior), axis2=S(superior)
    # User wants: axis0=S(top-bottom), axis1=R(left-right), axis2=A(inf-ant)
    # Transpose from (R,A,S) to (S,R,A)
    volume_data = volume_data.transpose(2, 0, 1)
    np.save(volumes_dir / "volume.npy", volume_data)
    print(f"Saved volume data to: {volumes_dir / 'volume.npy'}")
    print(f"Volume shape (S,R,A): {volume_data.shape}")
    np.save("temp.npy", volume_data)
    # Get spacing information if available
    try:
        spacing = nifti_img.header.get_zooms()
        spacing_list = [float(s) for s in spacing[:3]]
        original_spacing = [spacing_list[i] for i in [2, 0, 1]]
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
    if config.data_dir is None and config.nifti_files is None:
        print(
            "Error: Either --config.data-dir or --config.nifti-file must be provided",
            file=sys.stderr,
        )
        sys.exit(1)

    if config.data_dir is not None and config.nifti_files is not None:
        print(
            "Error: Cannot specify both --config.data-dir and --config.nifti-file", file=sys.stderr
        )
        sys.exit(1)

    # Track if we created a temporary directory for cleanup
    temp_dir_created = None

    if config.patientlist_path is not None:
        nifty_files = patient_list_to_nifti_files_generator(config.patientlist_path)
    elif config.nifti_files is not None:
        nifty_files = [(f, None, None, None) for f in config.nifti_files]
    else:
        raise ValueError("Either patientlist_path or nifti_files must be provided.")
    for nifti_file, dcm_folder, nifti_output_folder, decompressed_dcm_folder in nifty_files:
        run_extraction_for_nifti(
            nifti_file,
            config,
            temp_dir_created,
            dcm_folder,
            nifti_output_folder,
            decompressed_dcm_folder,
        )

def run_extraction_for_nifti(
    nifti_file: Path,
    config: ExtractConfig,
    temp_dir_created: Optional[Path],
    dcm_folder: Optional[Path] = None,
    nifti_output_folder: Optional[Path] = None,
    decompressed_dcm_folder: Optional[Path] = None,
) -> None:
    try:

        nifti_file = nifti_file.resolve()
        print(f"NIfTI file mode: {nifti_file}")

        # Map anatomy for consistency
        anatomy = config.anatomy
        if anatomy in ["abdomen", "abd"]:
            anatomy = "abdomen"

        # Create temporary data structure
        data_dir = setup_nifti_data_structure(nifti_file, anatomy, config.sample_name)
        temp_dir_created = data_dir


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
        if config.nifti_files:
            print(f"NIfTI File:      {config.nifti_files}")
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

        # Cleanup DICOM and NIfTI conversion temporary folders
        if dcm_folder is not None:
            print(f"Cleaning up DICOM folder: {dcm_folder}")
            shutil.rmtree(dcm_folder, ignore_errors=True)

        if decompressed_dcm_folder is not None:
            print(f"Cleaning up decompressed DICOM folder: {decompressed_dcm_folder}")
            shutil.rmtree(decompressed_dcm_folder, ignore_errors=True)

        if nifti_output_folder is not None:
            print(f"Cleaning up NIfTI output folder: {nifti_output_folder}")
            shutil.rmtree(nifti_output_folder, ignore_errors=True)


if __name__ == "__main__":
    tyro.cli(main)
