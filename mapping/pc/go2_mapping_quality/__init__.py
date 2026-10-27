"""Evidence-based quality gates for Go2 mapping deliverables."""

# Keep the small reconstruction-configuration modules importable in tooling that
# only validates an already-derived LiDAR surface.  The complete acceptance
# stack imports calibration code and OpenCV, which the read-only viewer does
# not need.  Public quality-gate symbols remain backward compatible via lazy
# resolution.
__all__ = ["MappingQualityGate", "QualityLevel", "assess_quality", "meets_requirement"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .acceptance import MappingQualityGate, QualityLevel, assess_quality, meets_requirement

    exports = {
        "MappingQualityGate": MappingQualityGate,
        "QualityLevel": QualityLevel,
        "assess_quality": assess_quality,
        "meets_requirement": meets_requirement,
    }
    value = exports[name]
    globals()[name] = value
    return value
