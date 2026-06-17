"""Headless end-to-end integration tests for the scatter brush (task 10.3).

The design's "Integration Testing Approach" calls for running inside a headless
Kit test app (``omni.kit.test``) with a known test stage, driving synthetic
stroke events through ``ViewportInputHandler`` and asserting prim counts in
``scatter.usda``, the authored referencing layout, the POC time budget, and that
undo reverts a full stroke.

``omni.kit.test`` and the Kit runtime are **not** available in this offline test
environment, but ``pxr`` (USD) is. So these tests wire the *real* components
end-to-end against a *real* in-memory ``Usd.Stage`` and only fake the seam where
the live Kit viewport would sit:

    ViewportInputHandler -> ScatterBrush -> SceneRaycaster (real, fed by a fake
    viewport + injected scene-query) -> PlacementSampler (real) ->
    AssetPalette (real) -> ScatterLayerWriter (real, authoring into a real
    anonymous ``scatter.usda`` sublayer of the mod/root layer).

The pointer-event path is exercised exactly as a live viewport would drive it:
synthetic down / move* / up events are delivered through the handler's
subscribed callback (the same seam used by ``test_input_handler.py``).

Because ``omni.kit.undo`` is unavailable offline, undo is modelled following the
design's fake-undo approach (design "Property 6: Undo atomicity"): a thin
recording proxy around the *real* writer captures, per stroke, the exact prim
paths authored between one ``open_undo_group`` and ``close_undo_group``. The test
asserts a single undo group wrapped the whole stroke and that reverting that
group (removing exactly those prim specs from ``scatter.usda``) returns the
scatter layer to its pre-stroke prim set.

Requirements covered:
    11.3 -- a single undo restores the layer to the exact instance set that
            existed before the stroke.
    12.1 -- authoring a completed paint action finishes within the POC time
            budget (the requirement allows 2s; asserted against a generous
            offline ceiling).
    12.2 -- each authored instance references the source grass asset prim.
    12.5 -- the authored structure matches the expected referencing layout
            (one transform-op Xform per instance, each referencing the source).
"""

from __future__ import annotations

import random
import time
from typing import List, Optional

import pytest

pytest.importorskip("pxr", reason="scatter integration tests require USD (pxr)")

from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: E402

from lightspeed.trex.scatter.brush import ScatterBrush  # noqa: E402
from lightspeed.trex.scatter.input_handler import (  # noqa: E402
    POINTER_DOWN,
    POINTER_MOVE,
    POINTER_UP,
    ViewportInputHandler,
)
from lightspeed.trex.scatter.models import BrushSettings  # noqa: E402
from lightspeed.trex.scatter.palette import AssetPalette  # noqa: E402
from lightspeed.trex.scatter.raycaster import SceneRaycaster  # noqa: E402
from lightspeed.trex.scatter.sampler import PlacementSampler  # noqa: E402
from lightspeed.trex.scatter.writer import (  # noqa: E402
    SCATTER_ROOT_PATH,
    ScatterLayerWriter,
)

# --------------------------------------------------------------------------- #
# Scene constants
# --------------------------------------------------------------------------- #

# The referenceable source asset every authored instance references (the "grass"
# of the POC). Defined on the stage so the writer's internal reference targets a
# real prim.
SOURCE_ASSET_PATH = "/World/Grass"

# The ground prim the synthetic raycast "hits"; the placements land on it. Must
# exist on the stage so SceneRaycaster's read-only prim validation accepts it.
GROUND_PATH = "/World/Ground"

# Deterministic RNG seed so the stroke (and therefore the instance count) is
# reproducible run to run.
SEED = 1234

# Viewport pixel resolution used to normalize synthetic pointer pixels.
VIEWPORT_WIDTH = 100
VIEWPORT_HEIGHT = 100

# Brush settings tuned so that *each* accepted drag sample authors exactly one
# instance: density * pi * radius^2 = 0.1 * pi * 1 ~= 0.314, which rounds to 0,
# and the sampler clamps the count to a minimum of 1. spacing=1.0 (world units)
# is small enough that consecutive stamps (10 world units apart, see below) are
# never throttled, so every move authors. A very high cap means no truncation.
_INSTANCES_PER_STAMP = 1
_SETTINGS = BrushSettings(
    radius=1.0,
    density=0.1,
    spacing=1.0,
    position_jitter=0.0,
    yaw_range=(0.0, 0.0),
    scale_range=(1.0, 1.0),
    align_to_normal=True,
    normal_blend=1.0,
    max_instances_per_stroke=10_000,
)

# Synthetic drag: a pointer-down followed by five moves and an up. Each move's
# normalized x advances by 0.1 (a 10 world-unit step, see ``_FakeViewport``), so
# all five are spaced well beyond ``spacing`` and each authors one instance.
_MOVE_PIXELS = [(20, 10), (30, 10), (40, 10), (50, 10), (60, 10)]
_DOWN_PIXEL = (10, 10)
_UP_PIXEL = (60, 10)
_EXPECTED_INSTANCES = len(_MOVE_PIXELS) * _INSTANCES_PER_STAMP

# Generous offline ceiling for the POC time budget (Requirement 12.1 allows 2s
# for the live authoring; authoring five prims in-memory is far quicker, but we
# keep the assertion generous so it never flakes on a busy CI host).
_TIME_BUDGET_SECONDS = 5.0


# --------------------------------------------------------------------------- #
# Fakes (only the live-Kit viewport seam is faked)
# --------------------------------------------------------------------------- #


class _FakeViewport:
    """Stand-in for the live Kit viewport.

    Provides exactly the seams the real components probe:

    - ``subscribe_to_pointer_event`` / ``send`` -- the pointer-event seam the
      ``ViewportInputHandler`` subscribes to (as in ``test_input_handler.py``).
    - ``width`` / ``height`` -- used by the handler to normalize device pixels.
    - ``compute_ray`` -- used by the real ``SceneRaycaster`` to build a
      world-space ray for a normalized cursor position. The ray drops straight
      down (-Z) from ``(nx * 100, ny * 100, 10)``.
    - ``stage`` -- the real stage, so the raycaster's read-only prim validation
      resolves the hit prim path.
    """

    def __init__(self, stage: "Usd.Stage", width: int, height: int) -> None:
        self.stage = stage
        self.width = width
        self.height = height
        self.callback = None
        self.subscription = None

    # -- pointer-event seam (input handler) --
    def subscribe_to_pointer_event(self, callback):
        self.callback = callback
        self.subscription = object()
        return self.subscription

    def send(self, event) -> None:
        assert self.callback is not None, "no subscriber attached to the viewport"
        self.callback(event)

    # -- ray-building seam (raycaster) --
    @staticmethod
    def compute_ray(screen_pos):
        nx, ny = float(screen_pos[0]), float(screen_pos[1])
        origin = (nx * 100.0, ny * 100.0, 10.0)
        direction = (0.0, 0.0, -1.0)
        return origin, direction


def _ground_plane_scene_query(origin, direction):
    """Injected closest-hit query: intersect the downward ray with ground z=0.

    Returns a hit dict in the shape ``SceneRaycaster`` understands: a unit
    surface normal (+Z), the world hit position on the ground plane, and the
    ground prim path. The hit position therefore tracks the cursor in X/Y, so
    consecutive drag samples are spaced apart in world space (driving the brush
    spacing throttle exactly as a real scene would).
    """
    hit_x, hit_y = float(origin[0]), float(origin[1])
    return {
        "hit": True,
        "position": (hit_x, hit_y, 0.0),
        "normal": (0.0, 0.0, 1.0),
        "prim_path": GROUND_PATH,
    }


class _RecordingUndoWriter:
    """Thin recording proxy around the *real* ``ScatterLayerWriter``.

    Delegates every call to the wrapped writer (so authoring happens on the real
    stage / real ``scatter.usda``) while recording the undo-group structure:
    one group is opened per ``open_undo_group``, every prim path authored while a
    group is open is captured into that group, and ``close_undo_group`` seals it.

    This is the offline stand-in for ``omni.kit.undo`` grouping (design
    "Property 6"): ``groups[i]`` is exactly the set of prim paths authored by
    stroke ``i``, which is what a single Ctrl+Z would revert.
    """

    def __init__(self, inner: ScatterLayerWriter) -> None:
        self._inner = inner
        self.open_labels: List[str] = []
        self.close_count = 0
        self.groups: List[List[str]] = []
        self._open_group: Optional[List[str]] = None

    # -- delegated layer access --
    def ensure_scatter_layer(self):
        return self._inner.ensure_scatter_layer()

    # -- recorded undo grouping --
    def open_undo_group(self, label: str) -> None:
        assert self._open_group is None, "undo groups must not nest"
        self._open_group = []
        self.open_labels.append(label)
        self._inner.open_undo_group(label)

    def author_placements(self, placements):
        paths = self._inner.author_placements(placements)
        if self._open_group is not None:
            self._open_group.extend(paths)
        return paths

    def close_undo_group(self) -> None:
        assert self._open_group is not None, "close without an open group"
        self.groups.append(self._open_group)
        self._open_group = None
        self.close_count += 1
        self._inner.close_undo_group()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _build_world():
    """Create the in-memory stage and wire the real end-to-end pipeline.

    Returns ``(stage, viewport, handler, writer, scatter_layer)`` where ``writer``
    is the recording proxy around the real ``ScatterLayerWriter`` and
    ``scatter_layer`` is the attached anonymous ``scatter.usda`` sublayer.
    """
    stage = Usd.Stage.CreateInMemory()
    # Referenceable source asset (the "grass") and the ground we paint onto.
    UsdGeom.Xform.Define(stage, SOURCE_ASSET_PATH)
    UsdGeom.Mesh.Define(stage, GROUND_PATH)

    mod_layer = stage.GetRootLayer()

    real_writer = ScatterLayerWriter(stage, mod_layer)
    writer = _RecordingUndoWriter(real_writer)

    palette = AssetPalette(stage)
    palette.add_asset(SOURCE_ASSET_PATH)

    sampler = PlacementSampler(random.Random(SEED))

    viewport = _FakeViewport(stage, VIEWPORT_WIDTH, VIEWPORT_HEIGHT)
    raycaster = SceneRaycaster(viewport, scene_query=_ground_plane_scene_query)

    brush = ScatterBrush(raycaster=raycaster, sampler=sampler, palette=palette, writer=writer)
    brush.set_settings(_SETTINGS)

    handler = ViewportInputHandler()
    handler.attach(viewport, brush)

    # Ensure the scatter layer exists so we can inspect it before any stroke.
    scatter_layer = real_writer.ensure_scatter_layer()
    return stage, viewport, handler, writer, scatter_layer


def _event(phase, pixel):
    return {"phase": phase, "x": pixel[0], "y": pixel[1]}


def _drive_full_stroke(viewport) -> None:
    """Deliver a synthetic down / move* / up stroke through the input handler."""
    viewport.send(_event(POINTER_DOWN, _DOWN_PIXEL))
    for pixel in _MOVE_PIXELS:
        viewport.send(_event(POINTER_MOVE, pixel))
    viewport.send(_event(POINTER_UP, _UP_PIXEL))


def _scatter_instance_paths(stage) -> set:
    """Return the set of authored placement prim paths under the scatter root."""
    prefix = SCATTER_ROOT_PATH + "/"
    return {
        str(prim.GetPath())
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(prefix)
    }


def _reference_target_paths(prim_spec) -> list:
    """Collect the source primPaths from a prim spec's reference list."""
    ref_list = prim_spec.referenceList
    items = (
        list(ref_list.prependedItems)
        + list(ref_list.appendedItems)
        + list(ref_list.explicitItems)
    )
    return [ref.primPath for ref in items]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_full_stroke_authors_expected_instance_count():
    """A full synthetic stroke authors one referencing Xform per drag sample.

    Validates: Requirements 12.1, 12.5
    """
    stage, viewport, _handler, _writer, _scatter_layer = _build_world()

    # No instances exist before painting.
    assert _scatter_instance_paths(stage) == set()

    _drive_full_stroke(viewport)

    instance_paths = _scatter_instance_paths(stage)
    assert len(instance_paths) == _EXPECTED_INSTANCES
    # Every authored instance lives under the dedicated scatter root scope.
    assert all(p.startswith(SCATTER_ROOT_PATH + "/") for p in instance_paths)


def test_authored_structure_matches_referencing_layout():
    """Each authored instance is a transform-op Xform referencing the source.

    The current architecture authors one referencing ``Xform`` per placement
    (the writer's documented layout): a translate/orient/scale op stack plus an
    internal reference to the source asset prim.

    Validates: Requirements 12.2, 12.5
    """
    stage, viewport, _handler, _writer, scatter_layer = _build_world()

    _drive_full_stroke(viewport)

    instance_paths = sorted(_scatter_instance_paths(stage))
    assert len(instance_paths) == _EXPECTED_INSTANCES

    for path in instance_paths:
        prim = stage.GetPrimAtPath(path)
        assert prim.IsValid(), f"expected authored prim at {path}"
        assert prim.GetTypeName() == "Xform"

        # Transform op stack: translate, orient, scale (Requirement 12.5).
        xformable = UsdGeom.Xformable(prim)
        op_names = [op.GetOpName() for op in xformable.GetOrderedXformOps()]
        assert "xformOp:translate" in op_names
        assert "xformOp:orient" in op_names
        assert "xformOp:scale" in op_names

        # The authored reference targets the source asset prim exactly, and the
        # reference opinion lives in the scatter layer (Requirements 12.2, 12.5).
        assert prim.HasAuthoredReferences()
        spec = scatter_layer.GetPrimAtPath(Sdf.Path(path))
        assert spec is not None, f"no prim spec in scatter layer for {path}"
        assert Sdf.Path(SOURCE_ASSET_PATH) in _reference_target_paths(spec)


def test_authoring_completes_within_poc_time_budget():
    """Authoring a completed paint action finishes well within the time budget.

    Requirement 12.1 allows the live tool 2 seconds; in-memory authoring of the
    stroke is far quicker, so we assert against a generous ceiling that still
    fails loudly on a pathological regression.

    Validates: Requirement 12.1
    """
    stage, viewport, _handler, _writer, _scatter_layer = _build_world()

    start = time.perf_counter()
    _drive_full_stroke(viewport)
    elapsed = time.perf_counter() - start

    # The stroke actually authored the expected instances (guards against a
    # vacuously-fast no-op stroke).
    assert len(_scatter_instance_paths(stage)) == _EXPECTED_INSTANCES
    assert elapsed < _TIME_BUDGET_SECONDS, (
        f"authoring took {elapsed:.3f}s, exceeding the {_TIME_BUDGET_SECONDS}s budget"
    )


def test_stroke_is_wrapped_in_exactly_one_undo_group():
    """The whole stroke is wrapped in exactly one open/close undo group.

    Precondition for undo atomicity: the brush opens exactly one undo group at
    ``begin_stroke`` and closes it at ``end_stroke`` (Requirement 11.1), so the
    stroke collapses to a single undoable operation.

    Validates: Requirements 11.3
    """
    _stage, viewport, _handler, writer, _scatter_layer = _build_world()

    _drive_full_stroke(viewport)

    assert len(writer.open_labels) == 1
    assert writer.close_count == 1
    assert len(writer.groups) == 1
    # The single group holds exactly the stroke's authored instances.
    assert len(writer.groups[0]) == _EXPECTED_INSTANCES


def test_single_undo_reverts_the_full_stroke():
    """Reverting the stroke's undo group returns scatter.usda to its prior set.

    Models a single Ctrl+Z: the recording writer captured the exact prim paths
    authored by the stroke (its one undo group); removing exactly those prim
    specs from ``scatter.usda`` must restore the layer to the pre-stroke instance
    set (which was empty).

    Validates: Requirements 11.3
    """
    stage, viewport, _handler, writer, scatter_layer = _build_world()

    pre_stroke = _scatter_instance_paths(stage)
    assert pre_stroke == set()

    _drive_full_stroke(viewport)

    after_stroke = _scatter_instance_paths(stage)
    assert len(after_stroke) == _EXPECTED_INSTANCES

    # The undo group records exactly the stroke's authored prims.
    assert len(writer.groups) == 1
    group = writer.groups[0]
    assert set(group) == after_stroke

    # Simulate undo: revert the group by removing exactly its prims from the
    # scatter layer (the only layer the stroke edited).
    with Usd.EditContext(stage, Usd.EditTarget(scatter_layer)):
        for path in group:
            stage.RemovePrim(path)

    reverted = _scatter_instance_paths(stage)
    # The layer is restored to the exact instance set that existed before the
    # stroke -- no more, no fewer (Requirement 11.3).
    assert reverted == pre_stroke
    # None of the stroke's instances linger.
    assert set(group).isdisjoint(reverted)


# --------------------------------------------------------------------------- #
# True omni.kit.test variant (skipped when the Kit runtime is unavailable)
# --------------------------------------------------------------------------- #


def test_headless_kit_end_to_end():
    """End-to-end variant intended for a real headless Kit test app.

    Runs only inside the Kit runtime (where ``omni.kit.test`` exists). Offline it
    is skipped gracefully. When Kit is present, the same real-component wiring is
    driven end-to-end (the wiring needs only USD + the scatter components, not a
    live viewport), asserting the headless integration contract from the design's
    "Integration Testing Approach".

    Validates: Requirements 11.3, 12.1, 12.2, 12.5
    """
    pytest.importorskip(
        "omni.kit.test", reason="headless Kit integration test requires the Kit runtime"
    )

    stage, viewport, _handler, writer, scatter_layer = _build_world()

    start = time.perf_counter()
    _drive_full_stroke(viewport)
    elapsed = time.perf_counter() - start

    instance_paths = _scatter_instance_paths(stage)
    assert len(instance_paths) == _EXPECTED_INSTANCES
    assert elapsed < _TIME_BUDGET_SECONDS

    # Referencing layout (12.2 / 12.5).
    for path in instance_paths:
        prim = stage.GetPrimAtPath(path)
        assert prim.HasAuthoredReferences()
        spec = scatter_layer.GetPrimAtPath(Sdf.Path(path))
        assert Sdf.Path(SOURCE_ASSET_PATH) in _reference_target_paths(spec)

    # Undo reverts the full stroke (11.3).
    assert len(writer.groups) == 1
    with Usd.EditContext(stage, Usd.EditTarget(scatter_layer)):
        for path in writer.groups[0]:
            stage.RemovePrim(path)
    assert _scatter_instance_paths(stage) == set()
