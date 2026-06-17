"""Unit tests for the scene raycaster (task 3.2).

These example-based tests drive ``SceneRaycaster`` through its dependency-
injection seams -- a fake ``viewport_api`` (exposing a ``compute_ray`` hook and a
read-only fake ``stage``) and an injected fake ``scene_query`` callable -- so the
ray/hit plumbing is exercised without a live Kit / PhysX runtime.

Requirements covered:
    3.1 -- on hit the service returns a ``SurfaceHit`` with a unit-length normal
           and a valid ``prim_path``; on miss it returns ``None``; the stage is
           never mutated; out-of-range screen positions are rejected.
"""

from __future__ import annotations

import math

import pytest

from lightspeed.trex.scatter.models import NORMAL_TOLERANCE, SurfaceHit
from lightspeed.trex.scatter.raycaster import SceneRaycaster

# A valid prim path that the fake stage will resolve as a real imageable prim.
_VALID_PRIM_PATH = "/World/Ground"
# A fixed world-space ray: origin above the origin looking straight down -Z.
_RAY_ORIGIN = (0.0, 0.0, 10.0)
_RAY_DIRECTION = (0.0, 0.0, -1.0)


# --------------------------------------------------------------------------- #
# Fakes (injection seams)
# --------------------------------------------------------------------------- #


class FakePrim:
    """A stand-in USD prim whose validity is controlled by the test."""

    def __init__(self, valid: bool) -> None:
        self._valid = valid

    def IsValid(self) -> bool:  # noqa: N802 - mirrors pxr.Usd.Prim API
        return self._valid


class RecordingStage:
    """Read-only fake stage that records any mutating call.

    ``GetPrimAtPath`` (a read) is allowed and resolves the configured set of
    valid paths. Every *other* attribute access is captured as a potential
    mutation hook (``DefinePrim``, ``RemovePrim``, ``SetEditTarget``, ...) and
    appended to ``mutations`` so tests can assert the raycaster never wrote to
    the stage.
    """

    def __init__(self, valid_paths) -> None:
        self._valid_paths = set(valid_paths)
        self.mutations = []
        self.read_paths = []

    def GetPrimAtPath(self, path):  # noqa: N802 - mirrors pxr.Usd.Stage API
        self.read_paths.append(str(path))
        return FakePrim(str(path) in self._valid_paths)

    def __getattr__(self, name):
        # Only reached for attributes not defined above; treat as a mutation.
        def _record(*args, **kwargs):
            self.mutations.append((name, args, kwargs))
            return None

        return _record


class FakeViewport:
    """Fake viewport exposing the ``compute_ray`` seam and a fake stage."""

    def __init__(self, ray, stage) -> None:
        self._ray = ray
        self.stage = stage
        self.compute_ray_calls = []

    def compute_ray(self, screen_pos):
        self.compute_ray_calls.append(tuple(screen_pos))
        return self._ray


def _make_scene_query(result):
    """Build an injectable scene_query returning ``result`` and recording calls."""
    calls = []

    def _query(origin, direction):
        calls.append((tuple(origin), tuple(direction)))
        return result

    _query.calls = calls
    return _query


def _vector_length(vec) -> float:
    return math.sqrt(sum(float(c) * float(c) for c in vec))


# --------------------------------------------------------------------------- #
# Hit case: unit normal + valid prim path (Requirement 3.1)
# --------------------------------------------------------------------------- #


def test_hit_returns_unit_normal_and_valid_prim_path():
    stage = RecordingStage(valid_paths=[_VALID_PRIM_PATH])
    viewport = FakeViewport(ray=(_RAY_ORIGIN, _RAY_DIRECTION), stage=stage)
    # Backend reports a hit with a deliberately *non-unit* normal (length 3).
    scene_query = _make_scene_query(
        {
            "hit": True,
            "position": (0.0, 0.0, 0.0),
            "normal": (0.0, 0.0, 3.0),
            "prim_path": _VALID_PRIM_PATH,
        }
    )

    raycaster = SceneRaycaster(viewport, scene_query=scene_query)
    hit = raycaster.raycast((0.5, 0.5))

    assert isinstance(hit, SurfaceHit)
    assert hit.prim_path == _VALID_PRIM_PATH
    # The raycaster must normalize the backend normal to unit length.
    assert abs(_vector_length(hit.normal) - 1.0) <= NORMAL_TOLERANCE
    # Direction preserved (only the magnitude changed): +Z.
    assert pytest.approx(hit.normal[2], abs=1e-6) == 1.0
    # The injected ray was actually used to query the scene.
    assert scene_query.calls == [(_RAY_ORIGIN, _RAY_DIRECTION)]
    assert viewport.compute_ray_calls == [(0.5, 0.5)]


# --------------------------------------------------------------------------- #
# Miss case: falsy / no-hit result -> None (Requirement 3.1)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "miss_result",
    [
        None,
        {},
        {"hit": False, "position": (0.0, 0.0, 0.0), "normal": (0.0, 0.0, 1.0)},
    ],
    ids=["none", "empty", "hit-flag-false"],
)
def test_miss_returns_none(miss_result):
    stage = RecordingStage(valid_paths=[_VALID_PRIM_PATH])
    viewport = FakeViewport(ray=(_RAY_ORIGIN, _RAY_DIRECTION), stage=stage)
    scene_query = _make_scene_query(miss_result)

    raycaster = SceneRaycaster(viewport, scene_query=scene_query)

    assert raycaster.raycast((0.5, 0.5)) is None


def test_unresolvable_prim_is_treated_as_miss():
    # A backend hit on a prim the stage cannot resolve must read as a miss so
    # the "prim_path is valid on hit" postcondition always holds.
    stage = RecordingStage(valid_paths=[_VALID_PRIM_PATH])
    viewport = FakeViewport(ray=(_RAY_ORIGIN, _RAY_DIRECTION), stage=stage)
    scene_query = _make_scene_query(
        {
            "hit": True,
            "position": (0.0, 0.0, 0.0),
            "normal": (0.0, 0.0, 1.0),
            "prim_path": "/World/DoesNotExist",
        }
    )

    raycaster = SceneRaycaster(viewport, scene_query=scene_query)

    assert raycaster.raycast((0.5, 0.5)) is None


# --------------------------------------------------------------------------- #
# No side effects: the stage is never mutated (Requirement 3.1)
# --------------------------------------------------------------------------- #


def test_raycast_does_not_mutate_stage():
    stage = RecordingStage(valid_paths=[_VALID_PRIM_PATH])
    viewport = FakeViewport(ray=(_RAY_ORIGIN, _RAY_DIRECTION), stage=stage)
    scene_query = _make_scene_query(
        {
            "hit": True,
            "position": (1.0, 2.0, 3.0),
            "normal": (0.0, 0.0, 2.0),
            "prim_path": _VALID_PRIM_PATH,
        }
    )

    raycaster = SceneRaycaster(viewport, scene_query=scene_query)
    hit = raycaster.raycast((0.25, 0.75))

    assert isinstance(hit, SurfaceHit)
    # Reading the prim is allowed; no Define/SetEditTarget/etc. may have run.
    assert stage.mutations == []
    assert stage.read_paths == [_VALID_PRIM_PATH]


# --------------------------------------------------------------------------- #
# Precondition: screen_pos must be in normalized [0, 1] (Requirement 3.1)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "screen_pos",
    [(-0.01, 0.5), (1.01, 0.5), (0.5, -0.01), (0.5, 1.01), (2.0, 2.0)],
)
def test_out_of_range_screen_pos_raises_value_error(screen_pos):
    stage = RecordingStage(valid_paths=[_VALID_PRIM_PATH])
    viewport = FakeViewport(ray=(_RAY_ORIGIN, _RAY_DIRECTION), stage=stage)
    scene_query = _make_scene_query({"hit": True})

    raycaster = SceneRaycaster(viewport, scene_query=scene_query)

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        raycaster.raycast(screen_pos)

    # Precondition is checked before any ray build / scene query / stage access.
    assert scene_query.calls == []
    assert viewport.compute_ray_calls == []
    assert stage.mutations == []
