"""I/O utilities for DICOM and NIfTI handling."""

from .dicom_reader import read_dicom_series, get_series_uid
from .nifti_io import read_nifti, write_nifti, update_spacing

__all__ = [
    'read_dicom_series',
    'get_series_uid',
    'read_nifti',
    'write_nifti',
    'update_spacing'
]
