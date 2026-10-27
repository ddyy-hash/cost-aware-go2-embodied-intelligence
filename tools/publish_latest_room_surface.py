from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mapping.pc.go2_mapping_dashboard.derived_surface import publish_session_odom_local_preview

root = Path("output/reconstruction/session-odom-local")
session = root / "segment-20260824T140911Z"
surface = max(session.rglob("measured-surface.json"), key=lambda path: path.stat().st_mtime)
print(publish_session_odom_local_preview(surface, output_root=root, status_path=Path("mapping-data/pc-stack/session-odom-local-current.json")))
