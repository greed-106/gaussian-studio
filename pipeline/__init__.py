"""
Pipeline processing modules for Gaussian Reconstruction Backend.

This package contains the core processing scripts:
- frame_extraction: Video frame extraction with sharpness-based selection
- colmap_sfm: Structure-from-Motion reconstruction using COLMAP
- lichtfeld_recon: Gaussian Splatting reconstruction using LichtFeld Studio
- compress: PLY to SOG compression using splat-transform
"""

from pipeline.frame_extraction import extract_frames
from pipeline.colmap_sfm import run_colmap_sfm
from pipeline.lichtfeld_recon import run_lichtfeld_training
from pipeline.compress import compress_splat

__all__ = [
    "extract_frames",
    "run_colmap_sfm",
    "run_lichtfeld_training",
    "compress_splat",
]
