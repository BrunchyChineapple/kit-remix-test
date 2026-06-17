"""Unit tests for the viewport input handler (task 9.2).

These example-based tests drive ``ViewportInputHandler`` through its
dependency-injection seams -- a fake ``viewport_api`` (exposing a single
``subscribe_to_pointer_event`` subscription seam plus a pixel resolution) and a
recording brush -- so the pointer-event -> stroke-lifecycle translation is
exercised without a live Kit runtime.

Requirements covered:
    3.1 -- the handler subscribes to viewport pointer events only while the tool
           is active (``attach`` wires the subscription, ``detach`` unsubscribes
           and delivers no further events), and pointer down/move/up map to the
           ``begin_stroke`` / ``continue_stroke`` / ``end_stroke`` lifecycle with
           coordinates normalized by the viewport resolution.
    3.7 -- exactly one of paint / erase mode is active at any time.
"""

from __future__ import annotations

import pytest

from lightspeed.trex.scatter.input_handler import (
    MODE_ERASE,
    MODE_PAINT,
    POINTER_DOWN,
    POINTER_MOVE,
    POINTER_UP,
    ViewportInputHandler,
)

# A viewport with a deliberately non-square resolution so that x and y are
# normalized by *different* divisors -- a swapped width/height would be caught.
_VIEWPORT_WIDTH = 200
_VIEWPORT_HEIGHT = 100


# --------------------------------------------------------------------------- #
# Fakes (injection seams)
# --------------------------------------------------------------------------- #


class FakeSubscription:
    """A subscription handle that records whether it has been released.

    ``ViewportInputHandler._unsubscribe`` probes for an ``unsubscribe`` method,
    so exposing one lets the test assert that ``detach`` released the handle.
    """

    def __init__(self) -> None:
        self.released = False

    def unsubscribe(self) -> None:
        self.released = True


class FakeViewport:
    """Fake viewport exposing the pointer subscription seam and a resolution.

    ``subscribe_to_pointer_event`` stores the handler's callback (so the test
    can feed it synthetic events) and returns a :class:`FakeSubscription` whose
    release the test can observe.
    """

    def __init__(self, width=_VIEWPORT_WIDTH, height=_VIEWPORT_HEIGHT) -> None:
        self.width = width
        self.height = height
        self.callback = None
        self.subscription = None

    def subscribe_to_pointer_event(self, callback):
        self.callback = callback
        self.subscription = FakeSubscription()
        return self.subscription

    def send(self, event) -> None:
        """Deliver ``event`` to the stored callback (mimics the backend)."""
        assert self.callback is not None, "no subscriber is attached"
        self.callback(event)


class RecordingBrush:
    """Records the stroke lifecycle calls and the screen positions passed in."""

    def __init__(self) -> None:
        self.begin_calls = []
        self.continue_calls = []
        self.end_calls = 0

    def begin_stroke(self, screen_pos) -> None:
        self.begin_calls.append(tuple(screen_pos))

    def continue_stroke(self, screen_pos) -> None:
        self.continue_calls.append(tuple(screen_pos))

    def end_stroke(self) -> None:
        self.end_calls += 1


def _down(x, y):
    return {"phase": POINTER_DOWN, "x": x, "y": y}


def _move(x, y):
    return {"phase": POINTER_MOVE, "x": x, "y": y}


def _up(x, y):
    return {"phase": POINTER_UP, "x": x, "y": y}


# --------------------------------------------------------------------------- #
# Subscription lifecycle (Requirement 3.1)
# --------------------------------------------------------------------------- #


def test_attach_subscribes_to_pointer_events():
    viewport = FakeViewport()
    brush = RecordingBrush()
    handler = ViewportInputHandler()

    assert handler.is_attached is False

    handler.attach(viewport, brush)

    assert handler.is_attached is True
    # The handler wired its callback into the viewport's subscription seam.
    assert viewport.callback is not None
    assert viewport.subscription is not None
    assert viewport.subscription.released is False


def test_detach_unsubscribes_and_stops_delivering_events():
    viewport = FakeViewport()
    brush = RecordingBrush()
    handler = ViewportInputHandler()
    handler.attach(viewport, brush)

    # Keep a reference to the callback so we can prove that, even if a stale
    # backend invokes it after detach, nothing reaches the brush.
    stale_callback = viewport.callback
    subscription = viewport.subscription

    handler.detach()

    assert handler.is_attached is False
    # The subscription handle was released.
    assert subscription.released is True

    # A late event delivered through the (now stale) callback is a no-op: the
    # brush reference was cleared so no stroke call is made.
    stale_callback(_down(100, 50))
    assert brush.begin_calls == []
    assert brush.continue_calls == []
    assert brush.end_calls == 0


# --------------------------------------------------------------------------- #
# Pointer down/move/up -> stroke lifecycle with normalized coords (Req 3.1)
# --------------------------------------------------------------------------- #


def test_pointer_down_begins_stroke_with_normalized_coordinates():
    viewport = FakeViewport()
    brush = RecordingBrush()
    handler = ViewportInputHandler()
    handler.attach(viewport, brush)

    # Pixel (100, 50) on a 200x100 viewport -> normalized (0.5, 0.5).
    viewport.send(_down(100, 50))

    assert len(brush.begin_calls) == 1
    nx, ny = brush.begin_calls[0]
    assert nx == pytest.approx(0.5)
    assert ny == pytest.approx(0.5)
    # Down alone neither continues nor ends a stroke.
    assert brush.continue_calls == []
    assert brush.end_calls == 0


def test_drag_maps_to_begin_continue_end_with_normalized_coordinates():
    viewport = FakeViewport()
    brush = RecordingBrush()
    handler = ViewportInputHandler()
    handler.attach(viewport, brush)

    viewport.send(_down(100, 50))   # -> begin_stroke (0.5, 0.5)
    viewport.send(_move(50, 25))    # -> continue_stroke (0.25, 0.25)
    viewport.send(_up(50, 25))      # -> end_stroke

    assert len(brush.begin_calls) == 1
    assert brush.begin_calls[0] == pytest.approx((0.5, 0.5))

    assert len(brush.continue_calls) == 1
    assert brush.continue_calls[0] == pytest.approx((0.25, 0.25))

    assert brush.end_calls == 1


def test_hover_move_without_down_does_not_continue_stroke():
    viewport = FakeViewport()
    brush = RecordingBrush()
    handler = ViewportInputHandler()
    handler.attach(viewport, brush)

    # A move with no preceding down is a hover -> no stroke calls at all.
    viewport.send(_move(100, 50))

    assert brush.begin_calls == []
    assert brush.continue_calls == []
    assert brush.end_calls == 0


def test_coordinates_are_clamped_to_unit_interval():
    viewport = FakeViewport()
    brush = RecordingBrush()
    handler = ViewportInputHandler()
    handler.attach(viewport, brush)

    # A pointer drifting past the bottom-right corner still normalizes in-range.
    viewport.send(_down(_VIEWPORT_WIDTH + 50, _VIEWPORT_HEIGHT + 25))

    assert len(brush.begin_calls) == 1
    nx, ny = brush.begin_calls[0]
    assert 0.0 <= nx <= 1.0
    assert 0.0 <= ny <= 1.0
    assert (nx, ny) == pytest.approx((1.0, 1.0))


# --------------------------------------------------------------------------- #
# Mode mutual exclusion (Requirement 3.7)
# --------------------------------------------------------------------------- #


def test_default_mode_is_paint_and_exactly_one_mode_active():
    handler = ViewportInputHandler()

    assert handler.mode == MODE_PAINT
    assert handler.is_paint_active is True
    assert handler.is_erase_active is False
    # Exactly one mode is active.
    assert handler.is_paint_active != handler.is_erase_active


def test_activate_erase_then_paint_toggles_exclusively():
    handler = ViewportInputHandler()

    handler.activate_erase()
    assert handler.mode == MODE_ERASE
    assert handler.is_erase_active is True
    assert handler.is_paint_active is False
    assert handler.is_paint_active != handler.is_erase_active

    handler.activate_paint()
    assert handler.mode == MODE_PAINT
    assert handler.is_paint_active is True
    assert handler.is_erase_active is False
    assert handler.is_paint_active != handler.is_erase_active


def test_set_mode_with_invalid_mode_raises_value_error():
    handler = ViewportInputHandler()

    with pytest.raises(ValueError):
        handler.set_mode("scatter")

    # The mode is left unchanged at its previous valid value.
    assert handler.mode == MODE_PAINT


# --------------------------------------------------------------------------- #
# Detach mid-stroke ends the stroke and unsubscribes (Requirement 3.1)
# --------------------------------------------------------------------------- #


def test_detach_during_active_stroke_ends_stroke_and_unsubscribes():
    viewport = FakeViewport()
    brush = RecordingBrush()
    handler = ViewportInputHandler()
    handler.attach(viewport, brush)

    subscription = viewport.subscription

    # Begin a stroke but never lift the pointer.
    viewport.send(_down(100, 50))
    assert len(brush.begin_calls) == 1
    assert brush.end_calls == 0

    handler.detach()

    # The in-progress stroke is closed exactly once and the handle released.
    assert brush.end_calls == 1
    assert subscription.released is True
    assert handler.is_attached is False
