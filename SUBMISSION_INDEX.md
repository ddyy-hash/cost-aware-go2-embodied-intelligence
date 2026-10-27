# Submission and Reproducibility Index

This repository contains source code and reproducibility assets for a Go2
indoor mapping, semantic navigation, and safety-gated terrain research system.
Start with the root [README](README.md) for the project summary, setup, and
public-release boundary.

## Package contents

- `mapping/`: PC mapping, dashboard API and frontend, RGB-D/LiDAR replay, Nav2
  mission logic, MobileVLA intent adapter, and safety-gated terrain modules.
- `third_party/`: Unitree SDK/ROS interfaces, vendored MobileVLA source, and
  minimal PGTT Go2 MuJoCo assets. Upstream license files remain authoritative.
- `models/pgtt-go2/`: compact, checksum-pinned demo policies used by offline
  terrain replay tests. They are not hardware locomotion authorization.
- `tools/`: WSL setup, validation, deployment, simulation, and robot
  maintenance scripts.
- `materials/`: local technical report, supplied presentation, and original
  competition demonstration video. Large materials are excluded from Git.

## Reproduction boundary

The reference PC environment is WSL Ubuntu 24.04 with the `go2-mapping-pc`
micromamba environment. The system is offline-first and fail-closed:
MobileVLA emits high-level navigation intents only; it cannot emit joint, gait,
torque, or motor commands. Physical locomotion requires separate operator
authorization, fresh terrain evidence, complete motor-health gates, and
supervised field acceptance.

Current evidence supports LiDAR/IMU observation, offline replay, and simulation
validation. It does not claim a universally accepted physical stair run, a
watertight building model, or calibrated final RGB texture.

## Quick checks

From the repository root in WSL:

```bash
python -m pytest -q \
  mapping/tests/test_offline_mapping_nav_pgtt_e2e.py \
  mapping/tests/test_pgtt_adapter.py \
  mapping/tests/test_navigation_package.py
python -m py_compile \
  mapping/pc/go2_mapping_dashboard/server.py \
  mapping/robot/go2_mapping_locomotion/terrain_router.py
```

For the full dashboard, install the declared Node dependencies in
`mapping/pc/dashboard/` with `npm ci` and run the FastAPI entry point described
by the deployment scripts.

## Simulation entry points

The included MuJoCo path is the reproducible simulation target:

```bash
python -m pip install mujoco imageio
python tools/run_pgtt_mujoco_reference.py \
  --xml third_party/phase_guided_terrain_traversal/go2/xmls/terrain_test_mjx.xml \
  --steps 1000
python tools/run_pgtt_mujoco_acceptance_matrix.py --episodes 1 --steps 300
```

These commands write reports under `output/`, which is excluded from Git.
Reports are simulation evidence only and set `executable=false` and
`transport_authorized=false`. Isaac Sim adapters are optional and require an
independent Isaac Sim installation and Go2 USD asset.

## Excluded from the public repository

Raw bags and point clouds, generated meshes/GLBs, model checkpoints, local
runtime output, secrets, deployment bundles, and large competition media are
not tracked. The local working copy may retain the original PDFs, PPTX, and
video. Publish media separately only when its redistribution rights are clear.
