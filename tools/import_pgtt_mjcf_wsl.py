#!/usr/bin/env python3
"""Import one PGTT MJCF scene into an Isaac Sim USD asset directory."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "None", "physics_gpu": False})
    try:
        from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig
        from pxr import Sdf, Usd, UsdPhysics

        args.output.mkdir(parents=True, exist_ok=True)
        config = MJCFImporterConfig(
            mjcf_path=str(args.mjcf),
            usd_path=str(args.output),
            import_scene=True,
            fix_base=False,
            merge_mesh=False,
            run_asset_transformer=True,
            run_multi_physics_conversion=True,
            joint_drive_type="force",
            joint_target_type="position",
            override_gain_type="fixed",
            override_bias_type="affine",
            override_gain_prm=[50.0, 0.0, 0.0],
            override_bias_prm=[0.0, -50.0, -0.5],
        )
        result = MJCFImporter(config).import_mjcf()
        # MuJoCo's geom friction is explicit; make the corresponding PhysX
        # material values explicit instead of relying on simulator defaults.
        stage = Usd.Stage.Open(result)
        foot_names = {"FL", "FR", "RL", "RR"}
        robot_base_path = None
        for prim in stage.Traverse():
            if prim.GetName() == "base" and prim.GetParent().GetName() == "Geometry" and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                robot_base_path = str(prim.GetPath()).rstrip("/") + "/"
                break
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath()).rstrip("/").split("/")
            path = str(prim.GetPath()).rstrip("/")
            is_robot_geometry = robot_base_path is not None and path.startswith(robot_base_path)
            is_foot = len(prim_path) >= 2 and prim_path[-1] in foot_names and prim_path[-2].endswith("_calf")
            purpose = prim.GetAttribute("purpose").Get() if prim.HasAttribute("purpose") else None
            if is_robot_geometry and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(is_foot)
            elif not is_robot_geometry and (
                purpose == "guide" or prim.GetTypeName() in {"Cube", "Plane"}
            ):
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(True)
            if prim.GetTypeName() != "Material":
                if prim.GetName() in foot_names and prim.GetTypeName() == "Sphere":
                    api = prim.GetPrimTypeInfo().GetAppliedAPISchemas()
                    if "PhysicsCollisionAPI" not in api:
                        prim.ApplyAPI("PhysicsCollisionAPI")
                continue
            prim.CreateAttribute("physics:staticFriction", Sdf.ValueTypeNames.Float).Set(0.6)
            prim.CreateAttribute("physics:dynamicFriction", Sdf.ValueTypeNames.Float).Set(0.6)
            prim.CreateAttribute("physics:restitution", Sdf.ValueTypeNames.Float).Set(0.0)
        stage.GetRootLayer().Save()
        print("PGTT_MJCF_IMPORT_PASS", result, flush=True)
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
