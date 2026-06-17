"""Property-based test for the stroke density cap (task 6.2).

This module validates the design's **Property 1: density cap**:

    The total number of instances authored over a whole stroke never exceeds
    ``settings.max_instances_per_stroke``.
    ``sum(len(author_placements call)) <= max_instances_per_stroke``

``ScatterBrush`` enforces this in ``continue_stroke``: it stops sampling once
``authored_this_stroke`` reaches the cap and otherwise truncates the sampler's
output to the remaining budget before authoring. The test drives a full stroke
(``begin_stroke`` -> many ``continue_stroke`` -> ``end_stroke``) with a sampler
that deliberately returns *more* placements per call than the cap, so the
truncation path is exercised, and asserts the invariant holds against both the
recorded writer calls and the brush's own ``authored_this_stroke`` counter.

The brush imports no ``pxr``/``omni`` modules and only reads ``hit.point`` plus
treats sampler output as an opaque list (length + slice), so this test uses
pure-Python fakes -- no USD runtime required:

* ``_FakeHit``       -- exposes a plain 3-tuple ``point`` (``_world_distance``
                        handles sequences).
* ``_FakeRaycaster`` -- always hits; never misses, so every ``continue_stroke``
                        reaches the cap logic.
* ``_FakeSampler``   -- returns a configurable number of placement stand-ins per
                        call (may exceed the cap, to stress truncation).
* ``_FakePalette``   -- never empty, so ``begin_stroke`` opens the stroke.
* ``_RecordingWriter`` -- records every ``author_placements`` length and counts
                          undo-group open/close calls.

Spacing is set to ``0.0`` so the world-space spacing throttle never skips a
sample; that isolates the density cap as the only thing limiting authoring.

Validates: Requirements 3.3
"""

from __future__ import annotations

from typing import List

from hypothesis import given, settings
from hypothesis import strategies as st

from lightspeed.trex.scatter.brush import ScatterBrush
from lightspeed.trex.scatter.models import BrushSettings


# --------------------------------------------------------------------------- #
# Fakes (no USD / pxr required)
# --------------------------------------------------------------------------- #


class _FakeHit:
    """A minimal surface-hit stand-in exposing only ``point``.

    ``ScatterBrush.continue_stroke`` reads ``hit.point`` and feeds it to
    ``_world_distance``, which accepts plain sequences -- so a 3-tuple is enough.
    """

    def __init__(self, point):
        self.point = point


class _FakeRaycaster:
    """A raycaster that always hits at a fixed point (never misses)."""

    def __init__(self, point=(0.0, 0.0, 0.0)):
        self._hit = _FakeHit(tuple(float(c) for c in point))

    def raycast(self, screen_pos):  # noqa: D401 - simple stub
        return self._hit


class _FakeSampler:
    """A sampler returning a scripted number of placement stand-ins per call.

    ``sample_sizes`` is consumed one entry per ``continue_stroke``; each entry is
    the number of opaque placement stand-ins to return. The brush never inspects
    a placement's contents here (it only takes ``len`` and slices), so plain
    integers serve as stand-ins.
    """

    def __init__(self, sample_sizes: List[int]):
        self._sizes = list(sample_sizes)
        self._index = 0

    def sample(self, hit, settings_, palette) -> List[int]:
        if self._index < len(self._sizes):
            n = self._sizes[self._index]
        else:
            n = 0
        self._index += 1
        return list(range(n))


class _FakePalette:
    """A palette that is never empty so a stroke is allowed to begin."""

    def is_empty(self) -> bool:
        return False


class _RecordingWriter:
    """Captures author_placements lengths and counts undo-group open/close."""

    def __init__(self) -> None:
        self.authored_lengths: List[int] = []
        self.open_groups = 0
        self.close_groups = 0

    def open_undo_group(self, label=None) -> None:
        self.open_groups += 1

    def close_undo_group(self) -> None:
        self.close_groups += 1

    def author_placements(self, placements) -> None:
        self.authored_lengths.append(len(placements))

    @property
    def total_authored(self) -> int:
        return sum(self.authored_lengths)


def _make_settings(cap: int) -> BrushSettings:
    """Valid ``BrushSettings`` with ``spacing=0`` and the given density cap.

    ``spacing=0`` disables the world-space throttle (``moved < 0`` is never
    true), so the density cap is the only constraint on authoring.
    """
    return BrushSettings(
        radius=1.0,
        density=1.0,
        spacing=0.0,
        position_jitter=1.0,
        yaw_range=(0.0, 360.0),
        scale_range=(1.0, 1.0),
        max_instances_per_stroke=cap,
    )


# --------------------------------------------------------------------------- #
# Property 1: density cap
# --------------------------------------------------------------------------- #


@given(
    cap=st.integers(min_value=1, max_value=200),
    sample_sizes=st.lists(
        st.integers(min_value=0, max_value=250),
        min_size=0,
        max_size=40,
    ),
)
@settings(max_examples=150, deadline=None)
def test_stroke_never_exceeds_density_cap(cap: int, sample_sizes: List[int]) -> None:
    """Total authored instances in a stroke never exceed the cap.

    Drives a full stroke with a sampler whose per-call output (often larger than
    the cap) stresses truncation, then asserts both the writer-recorded total and
    the brush's ``authored_this_stroke`` stay within ``max_instances_per_stroke``.

    Validates: Requirements 3.3
    """
    writer = _RecordingWriter()
    brush = ScatterBrush(
        raycaster=_FakeRaycaster(),
        sampler=_FakeSampler(sample_sizes),
        palette=_FakePalette(),
        writer=writer,
    )
    brush.set_settings(_make_settings(cap))

    brush.begin_stroke((0.0, 0.0))
    for _ in sample_sizes:
        brush.continue_stroke((0.0, 0.0))
    brush.end_stroke()

    # The cap is an upper bound on everything authored in the stroke.
    assert writer.total_authored <= cap, (
        f"writer authored {writer.total_authored} placements, exceeding cap {cap}"
    )
    assert brush.authored_this_stroke <= cap, (
        f"authored_this_stroke={brush.authored_this_stroke} exceeds cap {cap}"
    )
    # The brush's counter and the writer's recorded total agree exactly.
    assert brush.authored_this_stroke == writer.total_authored


@given(
    cap=st.integers(min_value=1, max_value=50),
    n_calls=st.integers(min_value=1, max_value=30),
)
@settings(max_examples=100, deadline=None)
def test_cap_is_saturated_when_supply_is_unlimited(cap: int, n_calls: int) -> None:
    """When every call offers more than the cap, the stroke authors exactly the cap.

    This tightens the upper-bound property: with abundant supply the brush must
    author up to -- and never past -- the cap, confirming truncation stops at the
    boundary rather than under- or over-shooting.

    Validates: Requirements 3.3
    """
    # Each continue_stroke offers (cap + 5) placements: always more than remaining.
    sample_sizes = [cap + 5] * n_calls
    writer = _RecordingWriter()
    brush = ScatterBrush(
        raycaster=_FakeRaycaster(),
        sampler=_FakeSampler(sample_sizes),
        palette=_FakePalette(),
        writer=writer,
    )
    brush.set_settings(_make_settings(cap))

    brush.begin_stroke((0.0, 0.0))
    for _ in sample_sizes:
        brush.continue_stroke((0.0, 0.0))
    brush.end_stroke()

    assert writer.total_authored == cap
    assert brush.authored_this_stroke == cap
