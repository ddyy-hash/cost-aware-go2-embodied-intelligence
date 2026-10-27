"""Content-addressed appearance sidecars for measured mapping geometry."""

from importlib import import_module


# Keep the immutable surface identity cheap to import. Dashboard mesh display
# and artifact validation do not need atlas charting or GLB-export modules.
from .identity import MeasuredSurfaceIdentity, measured_surface_identity

__all__ = [
    "MeasuredSurfaceIdentity",
    "measured_surface_identity",
    "ParsedVertexColor", "VertexColorSidecar", "build_historical_vertex_color_sidecar", "parse_vertex_color_v1",
    "HistoricalAppearanceStore", "PublishedAppearance", "DecodedTileV1", "decode_g2tm_v1",
    "RenderIndicesV1", "RenderUvsV1", "RenderVerticesV1", "encode_render_indices_v1",
    "encode_render_uvs_v1", "encode_render_vertices_v1", "parse_render_indices_v1",
    "parse_render_uvs_v1", "parse_render_vertices_v1", "ArtifactRef", "DeliveryAuthorization",
    "LodRecord", "PrimitiveRecord", "TileAppearanceRevision", "build_lod_record", "tile_appearance_revision",
    "ChartedMeshV1", "chart_mesh_v1", "TexelObservation", "ValidatedTexel", "select_validated_texel",
    "RasterizedAtlas", "rasterize_rgba8_atlas", "ChartPlacement", "DensityDecision", "LodPolicy",
    "PackedAtlases", "build_lod_policies", "pack_charts_v1", "select_adaptive_density",
    "PublishedTileAppearance", "TileAppearanceStore", "TileLodPayload", "ParsedTileVertexColor",
    "PublishedTileVertexColor", "TileVertexColorSidecar", "TileVertexColorStore",
    "build_tile_vertex_color_sidecar", "parse_tile_vertex_color_v1", "StaleTileGenerationError",
    "TileAtlasBakeRequest", "TileAtlasBakeResult", "TileAtlasBaker", "TileAppearanceScope",
    "ValidatedTileAppearanceProcessor", "build_validated_tile_bake_request", "BuildingAppearanceMember",
    "BuildingAppearanceStore", "DistanceLodInterval", "PublishedBuildingAppearance", "FieldAcceptanceResult",
    "canonical_acceptance_payload", "evaluate_textured_field_acceptance",
]


_EXPORT_MODULES = {
    "ParsedVertexColor": ".vertex_color", "VertexColorSidecar": ".vertex_color",
    "build_historical_vertex_color_sidecar": ".vertex_color", "parse_vertex_color_v1": ".vertex_color",
    "HistoricalAppearanceStore": ".historical_store", "PublishedAppearance": ".historical_store",
    "DecodedTileV1": ".tile_codec", "decode_g2tm_v1": ".tile_codec",
    "RenderIndicesV1": ".atlas_codec", "RenderUvsV1": ".atlas_codec", "RenderVerticesV1": ".atlas_codec",
    "encode_render_indices_v1": ".atlas_codec", "encode_render_uvs_v1": ".atlas_codec",
    "encode_render_vertices_v1": ".atlas_codec", "parse_render_indices_v1": ".atlas_codec",
    "parse_render_uvs_v1": ".atlas_codec", "parse_render_vertices_v1": ".atlas_codec",
    "ArtifactRef": ".tile_identity", "DeliveryAuthorization": ".tile_identity", "LodRecord": ".tile_identity",
    "PrimitiveRecord": ".tile_identity", "TileAppearanceRevision": ".tile_identity",
    "build_lod_record": ".tile_identity", "tile_appearance_revision": ".tile_identity",
    "ChartedMeshV1": ".charting", "chart_mesh_v1": ".charting",
    "TexelObservation": ".view_selection", "ValidatedTexel": ".view_selection", "select_validated_texel": ".view_selection",
    "RasterizedAtlas": ".rasterizer", "rasterize_rgba8_atlas": ".rasterizer",
    "ChartPlacement": ".lod", "DensityDecision": ".lod", "LodPolicy": ".lod", "PackedAtlases": ".lod",
    "build_lod_policies": ".lod", "pack_charts_v1": ".lod", "select_adaptive_density": ".lod",
    "PublishedTileAppearance": ".tile_store", "TileAppearanceStore": ".tile_store", "TileLodPayload": ".tile_store",
    "ParsedTileVertexColor": ".tile_vertex_color", "PublishedTileVertexColor": ".tile_vertex_color",
    "TileVertexColorSidecar": ".tile_vertex_color", "TileVertexColorStore": ".tile_vertex_color",
    "build_tile_vertex_color_sidecar": ".tile_vertex_color", "parse_tile_vertex_color_v1": ".tile_vertex_color",
    "StaleTileGenerationError": ".baker", "TileAtlasBakeRequest": ".baker", "TileAtlasBakeResult": ".baker", "TileAtlasBaker": ".baker",
    "TileAppearanceScope": ".tile_evidence", "ValidatedTileAppearanceProcessor": ".tile_evidence",
    "build_validated_tile_bake_request": ".tile_evidence", "BuildingAppearanceMember": ".building_store",
    "BuildingAppearanceStore": ".building_store", "DistanceLodInterval": ".building_store",
    "PublishedBuildingAppearance": ".building_store", "FieldAcceptanceResult": ".field_acceptance",
    "canonical_acceptance_payload": ".field_acceptance", "evaluate_textured_field_acceptance": ".field_acceptance",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
