# Reproducibility Notes

This index records the supported offline checks, simulation entry points, and evidence limits for the public Go2 mapping package. The project overview and setup instructions are in the [README](README.md).

## Reference environment

The PC-side reference environment is WSL Ubuntu 24.04 with the `go2-mapping-pc` micromamba environment. Robot-side packages target the Unitree Go2 deployment image and its supported ROS 2 distribution.

## Offline checks

Run these commands from the repository root:

```bash
python -m pytest -q \
  mapping/tests/test_offline_mapping_nav_pgtt_e2e.py \
  mapping/tests/test_pgtt_adapter.py \
  mapping/tests/test_navigation_package.py
python -m py_compile \
  mapping/pc/go2_mapping_dashboard/server.py \
  mapping/robot/go2_mapping_locomotion/terrain_router.py
```

## MuJoCo replay

```bash
python -m pip install mujoco imageio
python tools/run_pgtt_mujoco_reference.py \
  --xml third_party/phase_guided_terrain_traversal/go2/xmls/terrain_test_mjx.xml \
  --steps 1000
python tools/run_pgtt_mujoco_acceptance_matrix.py --episodes 1 --steps 300
```

Replay reports are written to `output/`. They document simulation behavior and do not authorize hardware motion. Isaac Sim adapters require a separate Isaac Sim installation and Go2 USD asset.

## Evidence limits

The checked-in software supports offline replay, simulation checks, and LiDAR/IMU observation. It does not establish a universally accepted physical stair run, a watertight building model, or final calibrated RGB texture for all environments. Hardware operation requires fresh terrain evidence, runtime health checks, and operator authorization.

## Data excluded from Git

Raw bags, point clouds, generated meshes, runtime output, credentials, local network identities, model checkpoints, and large competition media are not tracked. Publish source media separately only when redistribution rights are clear. Third-party code and assets remain under the licenses in their respective directories.
