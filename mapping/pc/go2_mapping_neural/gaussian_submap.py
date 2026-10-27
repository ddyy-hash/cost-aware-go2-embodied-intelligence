from dataclasses import dataclass


@dataclass(frozen=True)
class GaussianBudget:
    max_vram_bytes: int
    bytes_per_gaussian: int


class GaussianSubmap:
    def __init__(self, budget: GaussianBudget) -> None:
        self.budget = budget
        self.surfaces = {}
        self.gaussian_count = 0

    def add_lidar_surface(self, surface_id: str, point_count: int, residual_m: float) -> None:
        if residual_m < 0 or point_count <= 0:
            raise ValueError("invalid LiDAR surface observation")
        self.surfaces[surface_id] = {"points": point_count, "residual_m": residual_m}

    def add_gaussians(self, count: int, source_surface_id: str) -> None:
        if source_surface_id not in self.surfaces:
            raise KeyError("Gaussian centers require a LiDAR surface")
        proposed = self.gaussian_count + count
        if proposed * self.budget.bytes_per_gaussian > self.budget.max_vram_bytes:
            raise MemoryError("Gaussian budget would exceed the VRAM limit")
        self.gaussian_count = proposed

    def overwrite_lidar_surface(self, surface_id: str) -> None:
        raise PermissionError("neural output cannot overwrite the as-built LiDAR measurement layer")
