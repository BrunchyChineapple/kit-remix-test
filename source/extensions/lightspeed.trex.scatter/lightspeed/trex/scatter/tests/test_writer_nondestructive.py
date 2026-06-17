"""Property-based tests for non-destructive scatter authoring (task 5.2).

These tests validate the design's **Property 5 (Non-destructive authoring)**:

    All authored prims live in ``scatter.usda``; no capture/source layer prim is
    modified by any scatter operation.

The setup mirrors the real composition: a stage whose root layer is the *mod*
layer, plus a distinct *capture* ``Sdf.Layer`` (added to the stage's layer
stack) that holds a captured ``mesh_<HASH>`` prim with authored geometry
attributes. ``ScatterLayerWriter`` is constructed against the mod layer and
authors its dedicated ``scatter.usda`` sublayer.

For random batches of placements we assert that:

    * every authored prim spec lives in the scatter sublayer,
    * none of the authored prim specs appear in the capture layer, and
    * the capture layer is byte-for-byte identical before and after authoring
      (verified by serializing the layer to a string and comparing).

These tests need a working USD runtime. ``pxr`` is available in this
environment; if it were ever missing the whole module is skipped.

Validates: Requirements 10.1, 14.1, 14.3
"""

from __future__ import annotations

from typing import List, Tuple

import pytest

pytest.importorskip("pxr", reason="ScatterLayerWriter authoring requires USD (pxr)")

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: E402

from lightspeed.trex.scatter.models import AssetRef, Placement  # noqa: E402
from lightspeed.trex.scatter.writer import ScatterLayerWriter  # noqa: E402

# A captured ``mesh_<HASH>`` prim (16 hex chars) that the placements reference.
# It lives only in the capture layer and must never be touched by authoring.
CAPTURED_MESH_PATH = "/RootNode/meshes/mesh_0123456789ABCDEF"


# --------------------------------------------------------------------------- #
# Hypothesis strategies
# --------------------------------------------------------------------------- #

# Finite, bounded translate components -- the exact values are irrelevant to the
# property; we only need a varied transform space.
_translate_component = st.floats(
    min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False
)

# Strictly-positive uniform scale (Placement requires scale > 0).
_scale = st.floats(
    min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False
)

# Raw quaternion components; normalised in-test so ``orient`` is unit length.
_quat_component = st.floats(
    min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# One placement's parameters as a flat tuple (tx, ty, tz, qw, qx, qy, qz, scale).
_PlacementParams = Tuple[float, float, float, float, float, float, float, float]

_placement_params = st.tuples(
    _translate_component,
    _translate_component,
    _translate_component,
    _quat_component,
    _quat_component,
    _quat_component,
    _quat_component,
    _scale,
)

# A batch of placements. ``min_size=0`` keeps zero-placement strokes in the
# input space; ``max_size`` is kept modest so each example stays cheap.
_batch = st.lists(_placement_params, min_size=0, max_size=12)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _build_capture_stage() -> Tuple["Usd.Stage", "Sdf.Layer"]:
    """Return a fresh stage (mod root layer + a distinct capture sublayer).

    The capture layer holds a captured ``mesh_<HASH>`` prim with authored
    geometry attributes so there is real data to detect mutation of.
    """
    stage = Usd.Stage.CreateInMemory()

    capture_layer = Sdf.Layer.CreateAnonymous("capture.usda")
    # Add the capture layer to the stage's layer stack (weaker than whatever the
    # writer will insert) so it composes into the stage like a real capture.
    stage.GetRootLayer().subLayerPaths.append(capture_layer.identifier)

    # Author the captured mesh + attributes directly into the capture layer.
    with Usd.EditContext(stage, Usd.EditTarget(capture_layer)):
        mesh = UsdGeom.Mesh.Define(stage, CAPTURED_MESH_PATH)
        mesh.CreatePointsAttr(
            [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)]
        )
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        mesh.GetPrim().SetCustomDataByKey("captured", True)

    return stage, capture_layer


def _make_placements(params: List[_PlacementParams]) -> List[Placement]:
    """Build valid ``Placement`` objects referencing the captured mesh prim."""
    asset = AssetRef(prim_path=CAPTURED_MESH_PATH, layer_id="capture")
    placements: List[Placement] = []
    for tx, ty, tz, qw, qx, qy, qz, scale in params:
        quat = Gf.Quatf(qw, qx, qy, qz)
        # Normalise so the quaternion is unit length (Placement requires it).
        # A (near-)zero quaternion has no valid orientation -> fall back to identity.
        if quat.GetLength() < 1e-6:
            quat = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
        else:
            quat = quat.GetNormalized()
        placements.append(
            Placement(
                asset=asset,
                translate=Gf.Vec3d(tx, ty, tz),
                orient=quat,
                scale=float(scale),
            )
        )
    return placements


# --------------------------------------------------------------------------- #
# Property 5: Non-destructive authoring
# --------------------------------------------------------------------------- #


@given(params=_batch)
@settings(
    max_examples=50,
    deadline=None,  # USD authoring wall-clock varies; avoid flaky deadlines
    suppress_health_check=[HealthCheck.too_slow],
)
def test_authoring_a_batch_never_touches_capture_layer(
    params: List[_PlacementParams],
) -> None:
    """A single authored batch lands only in scatter.usda; capture is untouched.

    Validates: Requirements 10.1, 14.1, 14.3
    """
    stage, capture_layer = _build_capture_stage()
    writer = ScatterLayerWriter(stage, stage.GetRootLayer())
    placements = _make_placements(params)

    # Snapshot the capture layer immediately before authoring (byte-for-byte).
    capture_before = capture_layer.ExportToString()

    authored = writer.author_placements(placements)
    scatter_layer = writer.ensure_scatter_layer()

    # The scatter layer is a real, distinct layer (not the mod/capture layer).
    assert scatter_layer is not stage.GetRootLayer()
    assert scatter_layer is not capture_layer

    # One authored path per placement (Property 5 operates on what was authored).
    assert len(authored) == len(placements)

    for path in authored:
        sdf_path = Sdf.Path(path)
        # Every authored prim spec lives in the scatter layer ...
        assert scatter_layer.GetPrimAtPath(sdf_path) is not None, (
            f"authored prim {path} should exist in the scatter layer"
        )
        # ... and NONE appear in the capture layer.
        assert capture_layer.GetPrimAtPath(sdf_path) is None, (
            f"authored prim {path} must not appear in the capture layer"
        )

    # The captured mesh prim spec still exists and is unchanged: the whole
    # capture layer is byte-for-byte identical before and after authoring.
    assert capture_layer.GetPrimAtPath(Sdf.Path(CAPTURED_MESH_PATH)) is not None
    assert capture_layer.ExportToString() == capture_before, (
        "the capture layer was modified by authoring (Property 5 violated)"
    )


@given(batches=st.lists(_batch, min_size=1, max_size=4))
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_repeated_strokes_never_touch_capture_layer(
    batches: List[List[_PlacementParams]],
) -> None:
    """A sequence of strokes accumulates in scatter.usda; capture stays intact.

    Validates: Requirements 10.1, 14.1, 14.3
    """
    stage, capture_layer = _build_capture_stage()
    writer = ScatterLayerWriter(stage, stage.GetRootLayer())

    capture_before = capture_layer.ExportToString()

    all_authored: List[str] = []
    for params in batches:
        all_authored.extend(writer.author_placements(_make_placements(params)))

    scatter_layer = writer.ensure_scatter_layer()

    for path in all_authored:
        sdf_path = Sdf.Path(path)
        assert scatter_layer.GetPrimAtPath(sdf_path) is not None
        assert capture_layer.GetPrimAtPath(sdf_path) is None

    # After every stroke, the capture layer is still byte-for-byte identical.
    assert capture_layer.ExportToString() == capture_before, (
        "the capture layer was modified across repeated strokes (Property 5 violated)"
    )
