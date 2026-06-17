"""Property-based test for "miss is a no-op" (task 6.3).

This module validates the design's **Property 7: a ``continue_stroke`` whose
raycast misses authors zero prims and does not advance ``_last_hit``**:

    For any active stroke, a drag sample whose raycast returns ``None`` (the
    pointer is over empty space) authors nothing and leaves the spacing anchor
    ``_last_hit`` exactly where it was. Painting over empty space is a no-op,
    not an error, and crucially must not reset or advance the anchor used by the
    world-space spacing throttle.

The test exercises ``ScatterBrush.continue_stroke`` with FAKES so no USD is
required:

- ``_FakeRaycaster`` maps each drag sample to a hit or a miss. The drag
  "screen position" we feed in *is* a small script tuple -- ``("hit", point)``
  yields a ``_FakeHit`` carrying a plain-tuple ``point`` (``brush._world_distance``
  handles plain sequences), and ``("miss",)`` yields ``None``.
- ``_FakeSampler`` returns a fixed-size placement list for any hit, so the only
  variable across the stroke is the hit/miss pattern.
- ``_FakePalette`` is never empty, so ``begin_stroke`` opens the stroke.
- ``_RecordingWriter`` records every ``author_placements`` call (and undo-group
  open/close counts) so we can assert a miss authors nothing.

Two properties are checked:

1. With ``spacing == 0`` (which disables the distance throttle so every hit is
   authored), a randomly generated sequence of hit/miss samples authors exactly
   on the hits: each miss adds zero ``author_placements`` calls, leaves
   ``authored_this_stroke`` unchanged, and leaves ``_last_hit`` identical
   (white-box identity check).
2. With ``spacing > 0``, a run of misses inserted between two hits does not
   advance or reset the anchor: a follow-up hit closer than ``spacing`` to the
   pre-miss anchor is still skipped, proving the misses left ``_last_hit``
   untouched.

Validates: Requirements 3.4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from hypothesis import given, settings
from hypothesis import strategies as st

from lightspeed.trex.scatter.brush import ScatterBrush
from lightspeed.trex.scatter.models import BrushSettings

# Fixed number of placements the fake sampler returns per hit. Kept small so the
# density cap is never reached within a generated stroke (so the only reason a
# hit would author nothing is the miss/spacing logic under test).
_PLACEMENTS_PER_HIT = 4


@dataclass
class _FakeHit:
    """Minimal stand-in for ``SurfaceHit`` exposing only ``.point``.

    ``ScatterBrush`` only reads ``hit.point`` (for the spacing distance) and
    otherwise passes the hit straight to the sampler, so a plain object with a
    tuple ``point`` is enough -- and avoids ``SurfaceHit``'s unit-normal
    validation, which is irrelevant to this property.
    """

    point: Tuple[float, float, float]


class _FakeRaycaster:
    """Maps a drag-sample script tuple to a hit or a miss.

    The ``screen_pos`` passed to ``continue_stroke`` is itself the script:
    ``("hit", point)`` -> a ``_FakeHit`` at ``point``; ``("miss",)`` -> ``None``.
    Stateless, so one instance serves a whole stroke.
    """

    def raycast(self, screen_pos: Tuple) -> Optional[_FakeHit]:
        if screen_pos[0] == "miss":
            return None
        return _FakeHit(point=screen_pos[1])


class _FakeSampler:
    """Returns a fixed-size placement list for any hit (USD-free)."""

    def __init__(self, n: int = _PLACEMENTS_PER_HIT) -> None:
        self._n = n
        self.calls = 0

    def sample(self, hit: object, settings_: BrushSettings, palette: object) -> List[object]:
        self.calls += 1
        # The writer is a fake, so the placement contents are irrelevant; only
        # the count matters to the brush (truncation + authored count).
        return [object() for _ in range(self._n)]


class _FakePalette:
    """A never-empty palette so ``begin_stroke`` opens the stroke."""

    def is_empty(self) -> bool:
        return False


class _RecordingWriter:
    """Captures ``author_placements`` calls and undo-group open/close counts."""

    def __init__(self) -> None:
        self.author_calls: List[int] = []  # one entry (placement count) per call
        self.open_count = 0
        self.close_count = 0

    def open_undo_group(self, label: str) -> None:
        self.open_count += 1

    def close_undo_group(self) -> None:
        self.close_count += 1

    def author_placements(self, placements: List[object]) -> None:
        self.author_calls.append(len(placements))


def _make_brush(spacing: float) -> Tuple[ScatterBrush, _RecordingWriter, _FakeSampler]:
    """Build a brush wired to fakes, with the given spacing applied."""
    sampler = _FakeSampler()
    writer = _RecordingWriter()
    brush = ScatterBrush(_FakeRaycaster(), sampler, _FakePalette(), writer)
    brush.set_settings(
        BrushSettings(
            radius=1.0,
            density=1.0,
            spacing=spacing,
            position_jitter=0.5,
            yaw_range=(0.0, 0.0),
            scale_range=(1.0, 1.0),
        )
    )
    return brush, writer, sampler


_coord = st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)
_point = st.tuples(_coord, _coord, _coord)

# A single drag sample: a miss or a hit at some world point.
_step = st.one_of(
    st.just(("miss",)),
    _point.map(lambda p: ("hit", p)),
)


@given(steps=st.lists(_step, min_size=1, max_size=30))
@settings(max_examples=100, deadline=None)
def test_miss_is_noop_with_zero_spacing(steps: List[Tuple]) -> None:
    """Each miss authors nothing and never advances ``_last_hit``.

    ``spacing == 0`` disables the distance throttle (``moved < 0`` is never
    true), so every *hit* is authored; this isolates the miss behaviour: misses
    are the only samples that may author nothing, and they must also leave the
    spacing anchor untouched.

    Validates: Requirements 3.4
    """
    brush, writer, _ = _make_brush(spacing=0.0)
    # The screen_pos passed to begin_stroke is not raycast (no authoring on
    # begin), so the first real hit defines the initial anchor.
    brush.begin_stroke(("hit", (0.0, 0.0, 0.0)))

    for step in steps:
        before_calls = len(writer.author_calls)
        before_authored = brush.authored_this_stroke
        before_anchor = brush._last_hit  # white-box: capture anchor identity

        brush.continue_stroke(step)

        if step[0] == "miss":
            # Property 7: a miss authors zero prims for this sample ...
            assert len(writer.author_calls) == before_calls
            assert brush.authored_this_stroke == before_authored
            # ... and does not advance/reset the spacing anchor.
            assert brush._last_hit is before_anchor
        else:
            # With spacing 0 every hit is accepted and authored exactly once.
            assert len(writer.author_calls) == before_calls + 1
            assert brush.authored_this_stroke == before_authored + _PLACEMENTS_PER_HIT
            # The anchor advances to the hit just authored (only on hits).
            assert brush._last_hit is not None
            assert brush._last_hit.point == step[1]

    brush.end_stroke()
    # Sanity: total authored matches the number of hit samples.
    hit_count = sum(1 for s in steps if s[0] == "hit")
    assert writer.author_calls == [_PLACEMENTS_PER_HIT] * hit_count


@given(
    near_distance=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    n_misses=st.integers(min_value=1, max_value=6),
)
@settings(max_examples=60, deadline=None)
def test_misses_between_hits_do_not_advance_anchor(near_distance: float, n_misses: int) -> None:
    """A run of misses leaves the spacing anchor exactly where the last hit set it.

    With ``spacing = 10`` and a follow-up hit only ``near_distance <= 5`` away
    from the original anchor, the follow-up hit must be throttled (skipped). It
    can only be skipped if ``_last_hit`` still points at the original anchor --
    i.e. the intervening misses neither advanced nor reset it.

    Validates: Requirements 3.4
    """
    spacing = 10.0
    brush, writer, _ = _make_brush(spacing=spacing)
    brush.begin_stroke(("hit", (0.0, 0.0, 0.0)))

    # First real hit at the origin -> authored, anchor = origin.
    brush.continue_stroke(("hit", (0.0, 0.0, 0.0)))
    assert writer.author_calls == [_PLACEMENTS_PER_HIT]
    anchor = brush._last_hit
    assert anchor is not None

    # A run of misses must not touch the anchor (identity preserved each time).
    for _ in range(n_misses):
        before_anchor = brush._last_hit
        before_calls = len(writer.author_calls)
        brush.continue_stroke(("miss",))
        assert brush._last_hit is before_anchor  # not advanced/reset by miss
        assert len(writer.author_calls) == before_calls  # nothing authored

    # A follow-up hit closer than spacing to the (unchanged) anchor is skipped.
    brush.continue_stroke(("hit", (near_distance, 0.0, 0.0)))
    assert writer.author_calls == [_PLACEMENTS_PER_HIT]  # no new authoring
    assert brush._last_hit is anchor  # anchor still the original hit

    brush.end_stroke()
