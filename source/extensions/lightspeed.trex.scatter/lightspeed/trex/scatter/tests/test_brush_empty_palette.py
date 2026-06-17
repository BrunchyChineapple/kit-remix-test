"""Unit test for empty-palette rejection at stroke start (task 6.4).

This module validates the design's **Property 8: ``begin_stroke`` with an empty
palette authors nothing and surfaces a warning**:

    When ``ScatterBrush.begin_stroke`` is called while the palette is empty, the
    stroke is rejected *before* any undo group is opened. The brush authors
    nothing (no ``open_undo_group`` / ``author_placements`` / ``close_undo_group``
    calls), surfaces the user-facing :data:`EMPTY_PALETTE_WARNING` (via the
    optional ``warning_handler`` callback and the testable
    :attr:`ScatterBrush.last_warning` property), and stays inactive.

The test is deliberately USD-free: it drives ``ScatterBrush`` through small
recording fakes (a fake palette, a recording writer, a recording warning
handler, and stub raycaster/sampler) so it exercises the orchestration logic
without a Kit/USD runtime. Because an empty palette is rejected before the
raycaster or sampler are touched, those stubs are intentionally inert and raise
if they are ever called.

Validates: Requirements 2.6, 9.7
"""

from __future__ import annotations

from typing import List

import pytest

from lightspeed.trex.scatter.brush import EMPTY_PALETTE_WARNING, ScatterBrush


# --------------------------------------------------------------------------- #
# Recording fakes (no USD)
# --------------------------------------------------------------------------- #


class FakePalette:
    """A palette whose emptiness is fixed at construction time.

    Only the surface the brush touches is implemented: ``is_empty()``.
    """

    def __init__(self, empty: bool) -> None:
        self._empty = empty

    def is_empty(self) -> bool:
        return self._empty


class RecordingWriter:
    """Records the writer calls the brush makes, in order.

    ``open_undo_group`` / ``close_undo_group`` / ``author_placements`` each
    append to :attr:`calls` so a test can assert exactly which authoring
    operations occurred (and that none did on a rejected stroke).
    """

    def __init__(self) -> None:
        self.calls: List[str] = []
        self.open_count = 0
        self.close_count = 0
        self.author_count = 0
        self.authored_batches: List[object] = []

    def open_undo_group(self, label: str = "") -> None:
        self.open_count += 1
        self.calls.append("open")

    def close_undo_group(self) -> None:
        self.close_count += 1
        self.calls.append("close")

    def author_placements(self, placements: object) -> List[str]:
        self.author_count += 1
        self.authored_batches.append(placements)
        self.calls.append("author")
        return []


class RecordingWarningHandler:
    """Captures every message the brush surfaces via the warning callback."""

    def __init__(self) -> None:
        self.messages: List[str] = []

    def __call__(self, message: str) -> None:
        self.messages.append(message)


class ExplodingRaycaster:
    """A raycaster that must never be called when the palette is empty."""

    def raycast(self, screen_pos: object):  # pragma: no cover - asserts if reached
        raise AssertionError(
            "raycast must not be called: an empty-palette stroke is rejected "
            "before any raycasting"
        )


class ExplodingSampler:
    """A sampler that must never be called when the palette is empty."""

    def sample(self, hit, settings, palette):  # pragma: no cover - asserts if reached
        raise AssertionError(
            "sample must not be called: an empty-palette stroke authors nothing"
        )


# --------------------------------------------------------------------------- #
# Property 8 — empty palette rejection
# --------------------------------------------------------------------------- #


def test_begin_stroke_empty_palette_authors_nothing_and_warns() -> None:
    """An empty-palette ``begin_stroke`` warns and authors nothing.

    Asserts the full Property 8 contract for the rejection path:
        * the writer's ``open_undo_group`` is NEVER called (no empty undo step);
        * no ``author_placements`` call is made;
        * the warning handler is invoked exactly once with
          :data:`EMPTY_PALETTE_WARNING`;
        * :attr:`ScatterBrush.last_warning` equals :data:`EMPTY_PALETTE_WARNING`;
        * the brush remains inactive.

    Validates: Requirements 2.6, 9.7
    """
    writer = RecordingWriter()
    handler = RecordingWarningHandler()
    brush = ScatterBrush(
        raycaster=ExplodingRaycaster(),
        sampler=ExplodingSampler(),
        palette=FakePalette(empty=True),
        writer=writer,
        warning_handler=handler,
    )

    brush.begin_stroke(screen_pos=(0.0, 0.0))

    # No undo group was opened (rejected before opening) and nothing authored.
    assert writer.open_count == 0
    assert writer.close_count == 0
    assert writer.author_count == 0
    assert writer.calls == []

    # The user-facing warning was surfaced through every channel.
    assert handler.messages == [EMPTY_PALETTE_WARNING]
    assert brush.last_warning == EMPTY_PALETTE_WARNING

    # The brush did not enter an active stroke.
    assert brush.is_active is False
    assert brush.authored_this_stroke == 0


def test_continue_after_rejected_begin_is_a_noop() -> None:
    """A ``continue_stroke`` after a rejected begin authors nothing.

    Because the empty-palette ``begin_stroke`` left the brush inactive, the
    following ``continue_stroke`` is a no-op: it must not author and must not
    even raycast (the raycaster/sampler stubs raise if touched).

    Validates: Requirements 2.6, 9.7
    """
    writer = RecordingWriter()
    handler = RecordingWarningHandler()
    brush = ScatterBrush(
        raycaster=ExplodingRaycaster(),
        sampler=ExplodingSampler(),
        palette=FakePalette(empty=True),
        writer=writer,
        warning_handler=handler,
    )

    brush.begin_stroke(screen_pos=(0.0, 0.0))
    # Must not raise (ExplodingRaycaster.raycast is never reached) and must not
    # author anything.
    brush.continue_stroke(screen_pos=(5.0, 5.0))

    assert brush.is_active is False
    assert writer.open_count == 0
    assert writer.author_count == 0
    assert writer.calls == []


def test_begin_stroke_without_handler_still_records_last_warning() -> None:
    """With no ``warning_handler`` wired, the warning is still observable.

    The design exposes the warning through :attr:`last_warning` regardless of
    whether a handler callback is set, so a panel-less brush (or a test) can
    still detect the rejection. Authoring is still suppressed.

    Validates: Requirements 2.6, 9.7
    """
    writer = RecordingWriter()
    brush = ScatterBrush(
        raycaster=ExplodingRaycaster(),
        sampler=ExplodingSampler(),
        palette=FakePalette(empty=True),
        writer=writer,
        warning_handler=None,
    )

    brush.begin_stroke(screen_pos=(0.0, 0.0))

    assert brush.last_warning == EMPTY_PALETTE_WARNING
    assert brush.is_active is False
    assert writer.calls == []


# --------------------------------------------------------------------------- #
# Contrast case — a non-empty palette opens exactly one undo group
# --------------------------------------------------------------------------- #


def test_begin_stroke_nonempty_palette_opens_one_undo_group() -> None:
    """A non-empty palette begins a real stroke: exactly one undo group opens.

    The contrast case confirms the rejection logic is specific to an empty
    palette: with at least one enabled asset, ``begin_stroke`` opens exactly one
    undo group, activates the stroke, and clears any pending warning;
    ``end_stroke`` then closes that one group.

    Validates: Requirements 2.6, 9.7
    """
    writer = RecordingWriter()
    handler = RecordingWarningHandler()
    brush = ScatterBrush(
        raycaster=ExplodingRaycaster(),
        sampler=ExplodingSampler(),
        palette=FakePalette(empty=False),
        writer=writer,
        warning_handler=handler,
    )

    brush.begin_stroke(screen_pos=(0.0, 0.0))

    # Exactly one undo group opened, none closed yet; nothing authored on begin.
    assert writer.open_count == 1
    assert writer.close_count == 0
    assert writer.author_count == 0
    assert writer.calls == ["open"]

    # Active stroke, no warning surfaced or recorded.
    assert brush.is_active is True
    assert brush.last_warning is None
    assert handler.messages == []

    brush.end_stroke()

    # The single per-stroke undo group is now closed.
    assert writer.open_count == 1
    assert writer.close_count == 1
    assert writer.calls == ["open", "close"]
    assert brush.is_active is False


def test_rejected_then_successful_stroke_clears_prior_warning() -> None:
    """A successful stroke after a rejected one clears the stale warning.

    Drives the brush from an empty palette (rejection sets ``last_warning``) to
    a non-empty palette (a real stroke), confirming the second ``begin_stroke``
    clears the prior warning and opens exactly one undo group.

    Validates: Requirements 2.6, 9.7
    """
    writer = RecordingWriter()
    handler = RecordingWarningHandler()
    palette = FakePalette(empty=True)
    brush = ScatterBrush(
        raycaster=ExplodingRaycaster(),
        sampler=ExplodingSampler(),
        palette=palette,
        writer=writer,
        warning_handler=handler,
    )

    # First stroke: rejected, warning recorded, nothing authored.
    brush.begin_stroke(screen_pos=(0.0, 0.0))
    assert brush.last_warning == EMPTY_PALETTE_WARNING
    assert writer.open_count == 0

    # The modder adds an asset; the palette is no longer empty.
    palette._empty = False

    # Second stroke: succeeds, prior warning cleared, one undo group opened.
    brush.begin_stroke(screen_pos=(1.0, 1.0))
    assert brush.last_warning is None
    assert brush.is_active is True
    assert writer.open_count == 1
    assert writer.author_count == 0

    brush.end_stroke()
    assert writer.close_count == 1
    assert brush.is_active is False
