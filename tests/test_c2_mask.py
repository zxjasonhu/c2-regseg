import nibabel as nib
import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from c2_reg_cls import (
    convert_dicom_to_nifti,
    generate_c2_mask,
    validate_c2_mask_alignment,
    validate_nifti_dicom_alignment,
)
from scripts import run_model_inference, run_pipeline
import c2_reg_cls.c2_mask as c2_mask_module


def _write_scan(path, shape=(20, 22, 24)):
    data = np.zeros(shape, dtype=np.int16)
    affine = np.array(
        [[-0.8, 0, 0, 10], [0, -0.8, 0, 20], [0, 0, 1.5, -30], [0, 0, 0, 1]],
        dtype=float,
    )
    nib.save(nib.Nifti1Image(data, affine), path)
    return affine


def _write_dicom_series(directory, slices=3, rows=4, columns=5):
    directory.mkdir(parents=True)
    series_uid = generate_uid()
    for index in range(slices):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        dataset = FileDataset(
            str(directory / f"slice_{index:03d}.dcm"),
            {},
            file_meta=meta,
            preamble=b"\0" * 128,
        )
        dataset.SeriesInstanceUID = series_uid
        dataset.SOPClassUID = CTImageStorage
        dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        dataset.Modality = "CT"
        dataset.Rows = rows
        dataset.Columns = columns
        dataset.PixelSpacing = [0.8, 0.7]
        dataset.SliceThickness = 1.5
        dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        dataset.ImagePositionPatient = [0, 0, index * 1.5]
        dataset.InstanceNumber = index + 1
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 1
        dataset.PixelData = np.full((rows, columns), index, dtype=np.int16).tobytes()
        dataset.is_little_endian = True
        dataset.is_implicit_VR = False
        dataset.save_as(dataset.filename, write_like_original=False)
    return np.asarray(
        [[-0.7, 0, 0, 0], [0, -0.8, 0, 0], [0, 0, 1.5, 0], [0, 0, 0, 1]],
        dtype=float,
    )


def test_dicom_conversion_preserves_historical_direction(tmp_path):
    dicom_dir = tmp_path / "dicom"
    affine = _write_dicom_series(dicom_dir)
    output_path = tmp_path / "scans" / "case" / "ct.nii.gz"
    calls = []

    def fake_converter(**kwargs):
        calls.append(kwargs)
        data = np.zeros((5, 4, 3), dtype=np.int16)
        nib.save(nib.Nifti1Image(data, affine), kwargs["output_file"])

    converted = convert_dicom_to_nifti(
        dicom_dir, output_path, converter=fake_converter
    )
    report = validate_nifti_dicom_alignment(dicom_dir, converted)

    assert converted == output_path.resolve()
    assert calls[0]["reorient_nifti"] is False
    assert report["nifti_shape_xyz"] == (5, 4, 3)
    assert report["scan_shape_zyx"] == (3, 4, 5)
    assert report["orientation"] == ("L", "P", "S")
    assert report["geometry_check"] == "dicom2nifti_reorient_false_affine"


def test_totalsegmentator_c2_generation_and_strict_alignment(tmp_path):
    scan_path = tmp_path / "ct.nii.gz"
    affine = _write_scan(scan_path)
    output_path = tmp_path / "segmentations" / "vertebrae_C2.nii.gz"
    calls = []

    def fake_segmenter(**kwargs):
        calls.append(kwargs)
        assert kwargs["task_id"] == [292]
        assert kwargs["trainer"] == "nnUNetTrainerNoMirroring"
        assert kwargs["resample"] == 1.5
        assert kwargs["crop"] is None
        mask = np.zeros((20, 22, 24), dtype=np.uint8)
        mask[4:16, 5:18, 6:20] = 1
        nib.save(nib.Nifti1Image(mask, affine), kwargs["file_out"] / output_path.name)

    generated = generate_c2_mask(
        scan_path,
        output_path,
        segmenter=fake_segmenter,
    )
    report = validate_c2_mask_alignment(scan_path, generated)

    assert generated == output_path.resolve()
    assert calls[0]["task_name"] == "total"
    assert calls[0]["roi_subset"] == ["vertebrae_C2"]
    assert calls[0]["device"] == "gpu"
    assert report["geometry_check"] == "nifti_shape_affine_orientation"
    assert report["scan_shape_zyx"] == (24, 22, 20)
    assert report["foreground_voxels"] > 0


def test_paper_mask_generation_rejects_a_different_totalsegmentator(monkeypatch):
    monkeypatch.setattr(c2_mask_module, "version", lambda _name: "2.9.0")
    with pytest.raises(RuntimeError, match=r"requires TotalSegmentator 2\.8\.0"):
        c2_mask_module._require_paper_totalsegmentator()


def test_existing_mask_is_validated_without_loading_totalsegmentator(tmp_path):
    scan_path = tmp_path / "ct.nii.gz"
    affine = _write_scan(scan_path)
    output_path = tmp_path / "vertebrae_C2.nii.gz"
    mask = np.zeros((20, 22, 24), dtype=np.uint8)
    mask[2:18, 2:20, 2:22] = 1
    nib.save(nib.Nifti1Image(mask, affine), output_path)

    def should_not_run(**_kwargs):
        raise AssertionError("Existing aligned mask should be reused")

    assert generate_c2_mask(scan_path, output_path, segmenter=should_not_run) == output_path.resolve()


def test_alignment_rejects_affine_mismatch(tmp_path):
    scan_path = tmp_path / "ct.nii.gz"
    affine = _write_scan(scan_path)
    mask_path = tmp_path / "vertebrae_C2.nii.gz"
    mask = np.ones((20, 22, 24), dtype=np.uint8)
    shifted = affine.copy()
    shifted[0, 3] += 5
    nib.save(nib.Nifti1Image(mask, shifted), mask_path)

    with pytest.raises(ValueError, match="affine"):
        validate_c2_mask_alignment(scan_path, mask_path)


def test_generation_requires_canonical_c2_filename(tmp_path):
    scan_path = tmp_path / "ct.nii.gz"
    _write_scan(scan_path)
    with pytest.raises(ValueError, match="vertebrae_C2.nii.gz"):
        generate_c2_mask(scan_path, tmp_path / "mask.nii.gz", segmenter=lambda **_: None)


def test_totalsegmentator_requires_explicit_nifti_conversion(tmp_path):
    dicom_dir = tmp_path / "dicom"
    _write_dicom_series(dicom_dir)
    with pytest.raises(ValueError, match="source NIfTI"):
        generate_c2_mask(
            dicom_dir,
            tmp_path / "vertebrae_C2.nii.gz",
            segmenter=lambda **_: None,
        )


def test_batch_pipeline_generates_mask_before_registration(tmp_path, monkeypatch):
    scan_path = tmp_path / "scans" / "case" / "ct.nii.gz"
    scan_path.parent.mkdir(parents=True)
    affine = _write_scan(scan_path)
    mask_path = tmp_path / "masks" / "case" / "vertebrae_C2.nii.gz"

    def fake_generate(scan, output, **_kwargs):
        assert scan == scan_path
        output.parent.mkdir(parents=True)
        mask = np.zeros((20, 22, 24), dtype=np.uint8)
        mask[3:17, 4:18, 5:20] = 1
        nib.save(nib.Nifti1Image(mask, affine), output)
        return output

    monkeypatch.setattr(run_pipeline, "generate_c2_mask", fake_generate)
    config = {
        "inputs": {
            "scan_base": str(tmp_path / "scans"),
            "scan_relative_path": "{study_id}/ct.nii.gz",
            "mask_base": str(tmp_path / "masks"),
            "mask_relative_path": "{study_id}/vertebrae_C2.nii.gz",
        },
        "c2_segmentation": {"device": "cpu"},
    }

    generated = run_pipeline._segment(config, ["case"], overwrite=False)
    reused = run_pipeline._segment(config, ["case"], overwrite=False)

    assert generated == {
        "complete": 1,
        "existing_aligned": 0,
        "missing_scan": 0,
        "failed": 0,
    }
    assert reused["existing_aligned"] == 1


def test_batch_cam_aggregation_is_restricted_to_c2(tmp_path):
    scan_path = tmp_path / "scans" / "case" / "ct.nii.gz"
    scan_path.parent.mkdir(parents=True)
    affine = _write_scan(scan_path, shape=(6, 7, 8))

    mask_path = tmp_path / "masks" / "case" / "vertebrae_C2.nii.gz"
    mask_path.parent.mkdir(parents=True)
    mask_xyz = np.zeros((6, 7, 8), dtype=np.uint8)
    mask_xyz[1:5, 2:6, 3:7] = 1
    nib.save(nib.Nifti1Image(mask_xyz, affine), mask_path)

    cam_dir = tmp_path / "cams" / "first_place"
    cam_dir.mkdir(parents=True)
    cam_zyx = np.arange(8 * 7 * 6, dtype=np.float32).reshape(8, 7, 6)
    np.savez_compressed(cam_dir / "case.npz", cam=cam_zyx)

    config = {
        "inputs": {
            "scan_base": str(tmp_path / "scans"),
            "scan_relative_path": "{study_id}/ct.nii.gz",
            "mask_base": str(tmp_path / "masks"),
            "mask_relative_path": "{study_id}/vertebrae_C2.nii.gz",
            "model_heatmap_dirs": [str(cam_dir)],
        },
        "outputs": {
            "output_root": str(tmp_path / "outputs"),
            "aggregated_heatmaps": "aggregated_heatmaps",
        },
    }

    counts = run_pipeline._aggregate(config, ["case"], overwrite=False)
    result = np.load(tmp_path / "outputs" / "aggregated_heatmaps" / "case.npz")["cam"]
    mask_zyx = mask_xyz.transpose(2, 1, 0)[::-1] > 0

    assert counts["complete"] == 1
    assert np.all(result[~mask_zyx] == 0)
    assert result[mask_zyx].max() == pytest.approx(1.0)


def test_batch_pipeline_converts_dicom_before_segmentation(tmp_path, monkeypatch):
    dicom_dir = tmp_path / "dicom" / "case"
    affine = _write_dicom_series(dicom_dir)
    scan_path = tmp_path / "scans" / "case" / "ct.nii.gz"

    def fake_convert(source, output, **_kwargs):
        assert source == dicom_dir
        output.parent.mkdir(parents=True)
        nib.save(nib.Nifti1Image(np.zeros((5, 4, 3), dtype=np.int16), affine), output)
        return output

    monkeypatch.setattr(run_pipeline, "convert_dicom_to_nifti", fake_convert)
    config = {
        "inputs": {
            "dicom_base": str(tmp_path / "dicom"),
            "dicom_relative_path": "{study_id}",
            "scan_base": str(tmp_path / "scans"),
            "scan_relative_path": "{study_id}/ct.nii.gz",
        }
    }
    counts = run_pipeline._convert(config, ["case"], overwrite=False)

    assert counts["complete"] == 1
    validate_nifti_dicom_alignment(dicom_dir, scan_path)


def test_model_inference_prepares_same_nonreoriented_nifti(tmp_path, monkeypatch):
    dicom_dir = tmp_path / "dicom" / "case"
    affine = _write_dicom_series(dicom_dir)
    scan_path = tmp_path / "scans" / "case" / "ct.nii.gz"

    def fake_convert(source, output):
        assert source == dicom_dir
        output.parent.mkdir(parents=True)
        nib.save(nib.Nifti1Image(np.zeros((5, 4, 3), dtype=np.int16), affine), output)
        return output

    monkeypatch.setattr(run_model_inference, "convert_dicom_to_nifti", fake_convert)
    inputs = {
        "dicom_base": str(tmp_path / "dicom"),
        "dicom_relative_path": "{study_id}",
        "scan_base": str(tmp_path / "scans"),
        "scan_relative_path": "{study_id}/ct.nii.gz",
    }

    prepared, status = run_model_inference._ensure_scan_nifti(inputs, "case")

    assert prepared == scan_path
    assert status == "converted"
    validate_nifti_dicom_alignment(dicom_dir, prepared)
