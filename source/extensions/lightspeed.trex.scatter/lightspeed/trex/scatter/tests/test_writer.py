"""Unit tests for ``ScatterLayerWriter`` authoring (task 5.4).

These example-based tests author placements into a real, in-memory ``Usd.Stage``
and assert the documented behaviour of ``ScatterLayerWriter.author_placements``:

Requirements covered:
    3.5  -- a completed stroke authors one referencing Xform per placement
    5.5  -- placements reference the source asset prim (geometry is not copied)
    10.2 -- all edits land in the dedicated scatter sublayer; the prior edit
            target is restored and the capture/mod layer is never touched

The tests need a working USD runtime. ``pxr`` is available in this environment;
if it were ever missing the whole module is skipped rather than failing.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pxr", reason="ScatterLayerWriter authoring requires USD (pxr)")

from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: E402

from lightspeed.trex.scatter.models import AssetRef, Placement  # noqa: E402
from lightspeed.trex.scatter.writer import (  # noqa: E402
    SCATTER_ROOT_PATH,
    ScatterLayerWriter,
)

# Source prim that every placement references. Defined on the stage so that the
# internal reference (AddReference with an empty assetPath) targets a real prim.
SOURCE_PRIM_PATH = "/World/SourceAsset"


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


@pytest.fixture()
def stage() -> "Usd.Stage":
    """An in-memory stage with a single referenceable source Xform."""
    usd_stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(usd_stage, SOURCE_PRIM_PATH)
    return usd_stage


@pytest.fixture()
def writer(stage) -> ScatterLayerWriter:
    """A writer whose mod layer is the stage's (anonymous) root layer."""
    return ScatterLayerWriter(stage, stage.GetRootLayer())


def _make_placement(prim_path: str = SOURCE_PRIM_PATH, scale: float = 1.0) -> Placement:
    """Build a valid Placement referencing ``prim_path``."""
    return Placement(
        asset=AssetRef(prim_path=prim_path, layer_id="anon"),
        translate=Gf.Vec3d(1.0, 2.0, 3.0),
        orient=Gf.Quatf(1.0, 0.0, 0.0, 0.0),  # identity, unit length
        scale=scale,
    )


def _reference_prim_paths(prim_spec) -> list:
    """Collect the source primPaths from a prim spec's reference list."""
    ref_list = prim_spec.referenceList
    items = (
        list(ref_list.prependedItems)
        + list(ref_list.appendedItems)
        + list(ref_list.explicitItems)
    )
    return [ref.primPath for ref in items]


# --------------------------------------------------------------------------- #
# Authoring N placements (Requirement 3.5)
# --------------------------------------------------------------------------- #


def test_authors_one_prim_per_placement(stage, writer):
    placements = [_make_placement(scale=float(i + 1)) for i in range(5)]

    paths = writer.author_placements(placements)

    # One unique path per placement, all under the scatter root.
    assert len(paths) == len(placements)
    assert len(set(paths)) == len(placements)
    assert all(p.startswith(SCATTER_ROOT_PATH + "/") for p in paths)

    # Every authored prim exists on the composed stage.
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        assert prim.IsValid(), f"expected authored prim at {path}"


def test_each_prim_has_transform_op_stack(stage, writer):
    paths = writer.author_placements([_make_placement(scale=2.5)])

    xformable = UsdGeom.Xformable(stage.GetPrimAtPath(paths[0]))
    op_names = [op.GetOpName() for op in xformable.GetOrderedXformOps()]

    assert "xformOp:translate" in op_names
    assert "xformOp:orient" in op_names
    assert "xformOp:scale" in op_names


def test_each_prim_references_the_source(stage, writer):
    paths = writer.author_placements([_make_placement() for _ in range(3)])

    scatter_layer = writer.ensure_scatter_layer()
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        assert prim.HasAuthoredReferences()

        # The authored reference targets the source prim path exactly.
        spec = scatter_layer.GetPrimAtPath(Sdf.Path(path))
        assert Sdf.Path(SOURCE_PRIM_PATH) in _reference_prim_paths(spec)


# --------------------------------------------------------------------------- #
# Edits land only in the scatter layer (Requirement 10.2)
# --------------------------------------------------------------------------- #


def test_edits_land_in_scatter_layer_not_mod_layer(stage, writer):
    root_layer = stage.GetRootLayer()
    paths = writer.author_placements([_make_placement() for _ in range(4)])

    scatter_layer = writer.ensure_scatter_layer()
    assert scatter_layer is not root_layer

    for path in paths:
        sdf_path = Sdf.Path(path)
        # The placement spec exists in the scatter layer ...
        assert scatter_layer.GetPrimAtPath(sdf_path) is not None
        # ... and not in the mod/root (capture-side) layer.
        assert root_layer.GetPrimAtPath(sdf_path) is None


def test_scatter_layer_is_strongest_sublayer_of_mod_layer(stage, writer):
    scatter_layer = writer.ensure_scatter_layer()
    sublayers = list(stage.GetRootLayer().subLayerPaths)

    assert sublayers, "scatter layer should be attached as a sublayer of the mod layer"
    assert sublayers[0] == scatter_layer.identifier


def test_edit_target_restored_after_authoring(stage, writer):
    target_before = stage.GetEditTarget()
    layer_before = target_before.GetLayer()

    writer.author_placements([_make_placement() for _ in range(2)])

    layer_after = stage.GetEditTarget().GetLayer()
    # The stage is never left targeting scatter.usda.
    assert layer_after == layer_before
    assert layer_after == stage.GetRootLayer()
    assert layer_after != writer.ensure_scatter_layer()


# --------------------------------------------------------------------------- #
# Uniqueness and additive accumulation (Requirement 3.5)
# --------------------------------------------------------------------------- #


def test_shared_asset_path_yields_unique_prim_paths(stage, writer):
    # All placements reference the SAME source prim; paths must still be unique.
    placements = [_make_placement(SOURCE_PRIM_PATH) for _ in range(6)]

    paths = writer.author_placements(placements)

    assert len(set(paths)) == len(placements)


def test_repeated_authoring_accumulates_without_collision(stage, writer):
    first = writer.author_placements([_make_placement() for _ in range(3)])
    second = writer.author_placements([_make_placement() for _ in range(3)])

    # No path from the second stroke collides with the first (purely additive).
    assert set(first).isdisjoint(set(second))

    # Both strokes' prims coexist on the stage.
    for path in first + second:
        assert stage.GetPrimAtPath(path).IsValid()


# --------------------------------------------------------------------------- #
# Zero-placement strokes author nothing (Requirements 3.5, 10.2)
# --------------------------------------------------------------------------- #


def test_empty_placements_returns_empty_and_authors_nothing(stage, writer):
    result = writer.author_placements([])

    assert result == []
    # No scatter root scope and no children were authored.
    scatter_root = stage.GetPrimAtPath(SCATTER_ROOT_PATH)
    assert not scatter_root.IsValid()


def test_empty_placements_after_real_stroke_adds_no_prims(stage, writer):
    writer.author_placements([_make_placement() for _ in range(3)])
    before = {p.GetPath() for p in stage.Traverse()}

    result = writer.author_placements([])

    after = {p.GetPath() for p in stage.Traverse()}
    assert result == []
    assert after == before
