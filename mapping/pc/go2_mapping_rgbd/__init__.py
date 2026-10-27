"""Verified RGB-D decoding, validation, appearance, and augmentation layers."""

from .decode import (
    CameraInfoRecord,
    ImageRecord,
    RgbdDecodeError,
    TransformRecord,
    decode_camera_info,
    decode_image,
    decode_tf_message,
)
from .depth_validation import (
    AcceptedDepthCorrespondence,
    CameraInternalReport,
    CrossSensorDepthReport,
    DepthTileReport,
    DepthValidator,
)
from .monocular_anchors import AnchorBuildResult, ConsensusAnchor, ConsensusAnchorBuilder
from .monocular_validation import (
    AnchorPartition,
    AnchorSeedIdentity,
    BootstrapStability,
    MetricDepthFit,
    MetricDepthUncertainty,
    bootstrap_stability,
    compute_metric_depth_uncertainty,
    fit_metric_depth,
    stratified_holdout,
)
from .texture import ColorAccumulator, ColorObservation, VertexColor, srgb_to_linear
from .augmentation import (
    AugmentationBuilder,
    AugmentedLayer,
    BoundedHoleComponent,
    bounded_hole_components,
)
from .augmentation_identity import RgbdAugmentedIdentity, rgbd_augmented_identity
from .augmentation_store import PublishedAugmentation, RgbdAugmentationStore
from .artifact_refinement import AppearanceEvidence, ArtifactRgbdRefiner
from .depth_anything import (
    DepthAnything3Provider,
    DepthAnythingError,
    DepthAnythingProvider,
    RelativeDepthFrame,
    TransformersDepthAnythingProvider,
)
from .depth_anything_worker import (
    DepthAnythingWorkItem,
    DepthAnythingWorker,
    DepthAnythingWorkResult,
)
from .depth_anything_runtime import (
    DepthAnythingProcessor,
    create_depth_anything_worker_from_environment,
)

__all__ = [
    "CameraInfoRecord",
    "CameraInternalReport",
    "AcceptedDepthCorrespondence",
    "CrossSensorDepthReport",
    "DepthTileReport",
    "DepthValidator",
    "AnchorBuildResult",
    "ConsensusAnchor",
    "ConsensusAnchorBuilder",
    "AnchorPartition",
    "AnchorSeedIdentity",
    "BootstrapStability",
    "MetricDepthFit",
    "MetricDepthUncertainty",
    "bootstrap_stability",
    "compute_metric_depth_uncertainty",
    "fit_metric_depth",
    "stratified_holdout",
    "ColorAccumulator",
    "ColorObservation",
    "VertexColor",
    "srgb_to_linear",
    "AugmentationBuilder",
    "AugmentedLayer",
    "BoundedHoleComponent",
    "bounded_hole_components",
    "RgbdAugmentedIdentity",
    "rgbd_augmented_identity",
    "PublishedAugmentation",
    "RgbdAugmentationStore",
    "ArtifactRgbdRefiner",
    "AppearanceEvidence",
    "DepthAnythingError",
    "DepthAnything3Provider",
    "DepthAnythingProvider",
    "RelativeDepthFrame",
    "TransformersDepthAnythingProvider",
    "DepthAnythingWorkItem",
    "DepthAnythingWorker",
    "DepthAnythingWorkResult",
    "DepthAnythingProcessor",
    "create_depth_anything_worker_from_environment",
    "ImageRecord",
    "RgbdDecodeError",
    "TransformRecord",
    "decode_camera_info",
    "decode_image",
    "decode_tf_message",
]
