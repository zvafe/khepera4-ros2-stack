"""Decoupled 3D Gaussian Splatting Reconstruction Pipeline for Khepera IV."""

from .coordinate_lifting import CoordinateLifter
from .gsplat_trainer import GaussianModel, GsplatTrainer
from .metric_depth import MetricDepthEstimator
from .pipeline_coordinator import ReconstructionPipeline
from .ply_exporter import SuperSplatPlyExporter
from .rosbag_processor import RosbagKeyframeExtractor, RosbagReader
from .spatial_decimation import KeyframeData, SpatialDecimator

__all__ = [
    "CoordinateLifter",
    "GaussianModel",
    "GsplatTrainer",
    "MetricDepthEstimator",
    "ReconstructionPipeline",
    "SuperSplatPlyExporter",
    "RosbagKeyframeExtractor",
    "RosbagReader",
    "KeyframeData",
    "SpatialDecimator",
]

__version__ = "0.1.0"
