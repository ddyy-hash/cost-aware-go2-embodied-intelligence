"""Audit static room filtering on one existing measured-surface JSON."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mapping.pc.go2_mapping_fusion.room_surface_filter import filter_static_surface_points

parser = argparse.ArgumentParser()
parser.add_argument("surface_json", type=Path)
args = parser.parse_args()
payload = json.loads(args.surface_json.read_text(encoding="utf-8"))
import numpy as np
positions = np.asarray(payload["positions"], dtype=np.float64).reshape(-1, 3)
_, _, report = filter_static_surface_points(positions)
print(json.dumps({"surface": str(args.surface_json), **report}, sort_keys=True))
