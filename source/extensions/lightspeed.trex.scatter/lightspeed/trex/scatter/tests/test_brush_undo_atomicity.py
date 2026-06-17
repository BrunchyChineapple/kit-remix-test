"""Property-based test for undo atomicity (task 6.5).

This module validates the design's **Property 6: Undo atomicity**:

    A single undo reverts *exactly* the placements of the most recent completed
    stroke -- no more, no fewer.

``ScatterBrush`` itself owns no USD and no undo stack; it coordinates the writer,
which wraps each stroke in exactly one ``omni.kit.undo`` group
(``open_undo_group`` at ``begin_stroke`` -> one or more ``author_placements``
calls -> ``close_undo_group`` at ``end_stroke``). Because ``omni.kit.undo`` is
not importable outside the Kit runtime, this test follows the design's
fake-undo-model approach: a :class:`RecordingUndoWriter` records, per stroke, the
exact set of placements authored between ``open_undo_group`` and
``close_undo_group`` (i.e. one "undo group" per stroke) and implements a tiny
``undo()`` that pops the most recent *closed* group. We then assert that a single
``undo()`` reverts exactly the placements of the most recent completed stroke and
leaves every earlier stroke intact.

The brush drives this through its real lifecycle: for a random sequence of
strokes (each ``begin_stroke`` / several ``continue_stroke`` / ``end_stroke``),
each ``continue_stroke`` authors a random number of placements via a fake
sampler + raycaster (the raycast always hits; ``spacing == 0`` so no sample is
throttled; the density cap is set high so no placement is truncated). This keeps
the partition between strokes exactly equal to what the sampler produced, so the
test can compare the writer's per-stroke groups against an independent
expectation.

Validates: Requirements 11.1, 11.3
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Set

from hypothesis import given, settings
from hypothesis import strategies as st

from lightspeed.trex.scatter.brush import ScatterBrush
from lightspeed.trex.scatter.models import AssetRef, BrushSettings, Placement, SurfaceHit

# A surface normal of unit length and a normalized identity quaternion satisfy
# the SurfaceHit / Placement validation; the geometry is otherwise irrelevant to
# undo grouping.
_UNIT_NORMAL = (0.0, 0.0, 1.0)
_IDENTITY_ORIENT = (1.0, 0.0, 0.0, 0.0)
_HIT_POINT = (0.0, 0.0, 0.0)

# spacing == 0 disables the world-space throttle (so every continue_stroke
# authors), and a very high cap ensures the sampler's placements are never
# truncated -- both keep each stroke's authored set exactly equal to what the
# sampler returned, making the per-stroke partition unambiguous.
_SETTINGS = BrushSettings(
    radius=1.0,
    density=1.0,
    spacing=0.0,
    position_jitter=0.0,
    yaw_range=(0.0, 0.0),
    scale_range=(1.0, 1.0),
    max_instances_per_stroke=10_000,
)

_SCREEN_POS = (0.5, 0.5)


# --------------------------------------------------------------------------- #
# Fakes / injection seams
# --------------------------------------------------------------------------- #


class _UniquePlacementFactory:
    """Mints globally-unique, value-distinct :class:`Placement` objects.

    A monotonically increasing counter is encoded into each placement's
    ``translate`` so that no two placements are ever equal. That makes
    ``Placement`` value-equality double as identity, so the test can use sets to
    reason about which exact placements a stroke authored and which an ``undo()``
    reverted.
    """

    def __init__(self) -> None:
        self._counter = 0

    def make(self, count: int) -> List[Placement]:
        placements: List[Placement] = []
        for _ in range(count):
            i = self._counter
            self._counter += 1
            placements.append(
                Placement(
                    asset=AssetRef(prim_path=f"/World/Source_{i}", layer_id="anon"),
                    translate=(float(i), 0.0, 0.0),  # unique => identity
                    orient=_IDENTITY_ORIENT,
                    scale=1.0,
                )
            )
        return placements


class _AlwaysHitRaycaster:
    """Raycaster stub whose ray always hits the same valid surface point."""

    def raycast(self, screen_pos: object) -> SurfaceHit:
        return SurfaceHit(point=_HIT_POINT, normal=_UNIT_NORMAL, prim_path="/World/ground")


class _ScheduledSampler:
    """Returns a scheduled number of unique placements per ``sample`` call.

    The per-call counts are popped from ``counts`` in order (one pop per
    ``continue_stroke`` that authors). Every batch of placements returned is
    recorded in :attr:`returned_per_call` so the driver can reconstruct, per
    stroke, the exact placements that stroke authored -- an expectation that is
    independent of the writer under test.
    """

    def __init__(self, factory: _UniquePlacementFactory, counts: Deque[int]) -> None:
        self._factory = factory
        self._counts = counts
        self.returned_per_call: List[List[Placement]] = []

    def sample(self, hit: SurfaceHit, settings: BrushSettings, palette: object) -> List[Placement]:
        count = self._counts.popleft()
        batch = self._factory.make(count)
        self.returned_per_call.append(batch)
        return batch


class _NonEmptyPalette:
    """Palette stub that is never empty, so strokes are never rejected."""

    def is_empty(self) -> bool:
        return False


class _RecordingUndoWriter:
    """Fake undo model: one closed group per stroke, with a tiny ``undo()``.

    ``open_undo_group`` starts a new (initially empty) group; ``author_placements``
    appends to the currently-open group; ``close_undo_group`` seals it and pushes
    it onto the closed-group stack. ``undo()`` pops the most recent *closed*
    group and returns exactly the placements it contained -- modelling a single
    Ctrl+Z reverting exactly the most recent completed stroke.
    """

    def __init__(self) -> None:
        self._open_group: Optional[List[Placement]] = None
        self.open_count = 0
        self.close_count = 0
        self.closed_groups: List[List[Placement]] = []
        self.undone_groups: List[List[Placement]] = []

    def open_undo_group(self, label: str) -> None:
        assert self._open_group is None, "undo groups must not nest (open while open)"
        self._open_group = []
        self.open_count += 1

    def author_placements(self, placements: List[Placement]) -> List[str]:
        assert self._open_group is not None, "author_placements outside an undo group"
        self._open_group.extend(placements)
        return [f"/ScatterBrush/p_{id(p)}" for p in placements]

    def close_undo_group(self) -> None:
        assert self._open_group is not None, "close_undo_group without an open group"
        self.closed_groups.append(list(self._open_group))
        self._open_group = None
        self.close_count += 1

    def undo(self) -> List[Placement]:
        """Pop and return the most recent closed group (the last stroke)."""
        assert self._open_group is None, "cannot undo mid-group"
        if not self.closed_groups:
            return []
        reverted = self.closed_groups.pop()
        self.undone_groups.append(reverted)
        return reverted

    @property
    def has_closed_groups(self) -> bool:
        return bool(self.closed_groups)

    def live_placements(self) -> List[Placement]:
        """Every placement still 'in the scene' (union of all closed groups)."""
        live: List[Placement] = []
        for group in self.closed_groups:
            live.extend(group)
        return live


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #

# A stroke is described by the per-continue placement counts it authors. An empty
# list models a stroke that begins/ends without authoring (a completed but empty
# stroke); a 0 count models a continue that samples nothing.
_stroke = st.lists(st.integers(min_value=0, max_value=5), min_size=0, max_size=5)

# A sequence of strokes painted one after another with the same brush/writer.
_stroke_sequence = st.lists(_stroke, min_size=1, max_size=5)


def _flatten(groups: List[List[Placement]]) -> List[Placement]:
    return [p for group in groups for p in group]


def _drive_strokes(strokes: List[List[int]]):
    """Run ``strokes`` through a real ScatterBrush; return (writer, expected_groups).

    ``expected_groups[i]`` is the exact list of placements the sampler produced
    during stroke ``i`` -- the independent ground truth for what that stroke's
    undo group should contain.
    """
    factory = _UniquePlacementFactory()
    counts: Deque[int] = deque(count for stroke in strokes for count in stroke)
    sampler = _ScheduledSampler(factory, counts)
    writer = _RecordingUndoWriter()
    brush = ScatterBrush(
        raycaster=_AlwaysHitRaycaster(),
        sampler=sampler,
        palette=_NonEmptyPalette(),
        writer=writer,
    )
    brush.set_settings(_SETTINGS)

    expected_groups: List[List[Placement]] = []
    call_cursor = 0
    for stroke_counts in strokes:
        brush.begin_stroke(_SCREEN_POS)
        for _ in stroke_counts:
            brush.continue_stroke(_SCREEN_POS)
        brush.end_stroke()
        # Exactly one sample call per continue (spacing == 0, always-hit, no cap
        # truncation), so this stroke owns sampler calls [call_cursor : +n].
        n_calls = len(stroke_counts)
        group = _flatten(sampler.returned_per_call[call_cursor : call_cursor + n_calls])
        call_cursor += n_calls
        expected_groups.append(group)

    return writer, expected_groups


# --------------------------------------------------------------------------- #
# Property 6
# --------------------------------------------------------------------------- #


@given(strokes=_stroke_sequence)
@settings(max_examples=200, deadline=None)
def test_each_stroke_is_exactly_one_undo_group(strokes: List[List[int]]) -> None:
    """Each completed stroke maps to exactly one undo group with its placements.

    Precondition for atomicity: the brush opens exactly one undo group per stroke
    and closes it (Requirement 11.1), and the placements partition disjointly
    across stroke groups (no placement leaks between strokes).

    Validates: Requirements 11.1, 11.3
    """
    writer, expected_groups = _drive_strokes(strokes)

    # Exactly one open + one close per completed stroke (one undo group/stroke).
    assert writer.open_count == len(strokes)
    assert writer.close_count == len(strokes)
    assert len(writer.closed_groups) == len(strokes)

    # Each undo group holds exactly the placements that stroke authored.
    assert writer.closed_groups == expected_groups

    # The placements partition disjointly across stroke groups: the union has as
    # many distinct placements as the sum of the group sizes (no overlap).
    all_placements = _flatten(writer.closed_groups)
    total = sum(len(group) for group in writer.closed_groups)
    assert len(all_placements) == total
    assert len(set(all_placements)) == total

    # Every pair of stroke groups is disjoint.
    seen: Set[Placement] = set()
    for group in writer.closed_groups:
        group_set = set(group)
        assert seen.isdisjoint(group_set), "a placement appears in more than one stroke group"
        seen |= group_set


@given(strokes=_stroke_sequence)
@settings(max_examples=200, deadline=None)
def test_single_undo_reverts_exactly_the_last_stroke(strokes: List[List[int]]) -> None:
    """A single undo reverts exactly the most recent stroke -- no more, no fewer.

    Undoing repeatedly peels strokes off in reverse order; at each step the
    reverted set equals exactly that stroke's placements (by count and identity)
    and every earlier stroke remains fully intact in the scene.

    Validates: Requirements 11.1, 11.3
    """
    writer, expected_groups = _drive_strokes(strokes)

    remaining = list(expected_groups)  # strokes still "in the scene", oldest first
    while writer.has_closed_groups:
        live_before = set(writer.live_placements())
        expected_last = remaining[-1]

        reverted = writer.undo()

        # Exactly the last completed stroke's placements -- no more, no fewer.
        assert len(reverted) == len(expected_last)
        assert set(reverted) == set(expected_last)

        remaining.pop()
        live_after = set(writer.live_placements())

        # The undo removed exactly the reverted placements and nothing else.
        assert set(reverted) == live_before - live_after
        # The reverted placements are gone; none linger in the scene.
        assert set(reverted).isdisjoint(live_after)
        # Every earlier stroke is left fully intact (union of remaining strokes).
        assert live_after == set(_flatten(remaining))

    # After undoing every stroke the scene is empty.
    assert writer.live_placements() == []
    assert remaining == []


def test_two_stroke_example_single_undo_keeps_first_stroke() -> None:
    """Concrete example: paint two strokes, one undo reverts only the second.

    A focused, example-based companion to the property: the first stroke authors
    2 placements, the second authors 3; a single undo reverts exactly the 3 from
    the second stroke and leaves the first stroke's 2 placements untouched.

    Validates: Requirements 11.1, 11.3
    """
    writer, expected_groups = _drive_strokes([[2], [3]])

    assert [len(g) for g in expected_groups] == [2, 3]
    first_stroke, second_stroke = expected_groups

    reverted = writer.undo()

    assert set(reverted) == set(second_stroke)
    assert len(reverted) == 3
    # The first stroke survives unchanged.
    assert set(writer.live_placements()) == set(first_stroke)
    assert len(writer.live_placements()) == 2
