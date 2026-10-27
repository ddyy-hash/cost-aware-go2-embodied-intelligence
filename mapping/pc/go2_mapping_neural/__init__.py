"""LiDAR-constrained neural representations; never a geometry authority."""
from .dataset import NeuralDataset, NeuralDatasetExporter, NeuralFrame
from .gaussian_submap import GaussianBudget, GaussianSubmap
from .training import NeuralCheckpoint, NeuralTrainingError, NeuralTrainingJob

__all__ = [
    "GaussianBudget",
    "GaussianSubmap",
    "NeuralDataset",
    "NeuralDatasetExporter",
    "NeuralFrame",
    "NeuralCheckpoint",
    "NeuralTrainingError",
    "NeuralTrainingJob",
]
