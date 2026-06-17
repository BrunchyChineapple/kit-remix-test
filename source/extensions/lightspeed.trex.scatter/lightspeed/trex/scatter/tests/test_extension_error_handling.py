"""Unit tests for load-time and runtime error handling (task 10.2).

These tests exercise the three error-handling behaviours required by the design
("Error Handling" scenarios) and Requirements 1.4 / 10.5 / 14.4, using small
pure-Python fakes so no Kit/USD runtime is needed:

1. **No active stage or viewport** (design "Scenario: No active stage or
   viewport"): ``on_startup`` with no stage leaves the extension *loaded but
   disabled* -- the panel is greyed out and shows "Open a stage to use the
   scatter brush" -- and a stage-open event re-enables and re-wires the brush
   automatically; a stage-close event disables it again.

2. **Unresolved dependency at load time** (Requirement 1.4): when a declared
   dependency fails to import while wiring, ``on_startup`` aborts the
   extension's own load, records the unresolved dependency in
   :attr:`startup_error`, and does not raise (the host is not crashed).

3. **Missing / read-only mod layer at stroke time** (Requirements 10.5 / 14.4):
   when the writer reports the scatter/mod layer is missing or read-only
   (``ScatterLayerError``), the brush aborts the stroke leaving every layer
   unchanged -- rejecting before opening an undo group when detected at
   ``begin_stroke``, and closing the undo group cleanly (no open group leaks)
   when detected at ``continue_stroke`` -- and surfaces a clear error.

Validates: Requirements 1.4, 10.5, 14.4
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional

import pytest

from lightspeed.trex.scatter import extension as ext_mod
from lightspeed.trex.scatter.extension import NO_STAGE_MESSAGE, ScatterBrushExtension
from lightspeed.trex.scatter.brush import ScatterBrush
from lightspeed.trex.scatter.models import BrushSettings, SurfaceHit
from lightspeed.trex.scatter.writer import ScatterLayerError


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class _FakeStage:
    """Minimal stand-in for a ``Usd.Stage`` (only identity matters here)."""


class _FakeViewport:
    """Minimal stand-in for the active viewport API (only identity matters)."""


class _FakeWriter:
    """A writer that is always writable and records nothing (wiring only)."""

    def ensure_scatter_layer(self) -> object:
        return object()

    def open_undo_group(self, label: str = "") -> None:  # pragma: no cover - unused
        pass

    def close_undo_group(self) -> None:  # pragma: no cover - unused
        pass

    def author_placements(self, placements: object) -> List[str]:  # pragma: no cover
        return []


def _make_extension() -> ScatterBrushExtension:
    return ScatterBrushExtension()


# --------------------------------------------------------------------------- #
# Scenario 1: No active stage or viewport -> loaded but disabled
# --------------------------------------------------------------------------- #


def test_startup_without_stage_loads_but_disables_controls(monkeypatch) -> None:
    """Stage-less startup leaves the tool loaded but disabled with a message.

    Validates: Requirements 1.4 (recovery path) / design "No active stage".
    """
    monkeypatch.setattr(ext_mod, "_get_stage", lambda: None)

    ext = _make_extension()
    ext.on_startup("lightspeed.trex.scatter")

    # The load completed (host not crashed) and reported no error.
    assert ext.started is True
    assert ext.startup_error is None

    # The tool is disabled with no live brush, but the panel still exists.
    assert ext.enabled is False
    assert ext.brush is None
    assert ext.panel is not None

    # The panel is greyed out and shows the "open a stage" guidance.
    assert ext.panel.enabled is False
    assert ext.panel.status_message == NO_STAGE_MESSAGE


def test_stage_open_event_reenables_and_wires_the_brush(monkeypatch) -> None:
    """A stage-open event re-enables the tool and wires the brush.

    Validates: design "No active stage or viewport" (recovery).
    """
    # Start with no stage -> disabled.
    monkeypatch.setattr(ext_mod, "_get_stage", lambda: None)
    ext = _make_extension()
    ext.on_startup("lightspeed.trex.scatter")
    assert ext.enabled is False

    # A stage becomes available; make wiring resolve fakes (no USD needed).
    monkeypatch.setattr(ext_mod, "_get_stage", lambda: _FakeStage())
    monkeypatch.setattr(ext_mod, "_get_active_viewport", lambda: _FakeViewport())
    monkeypatch.setattr(ext_mod, "_resolve_mod_layer", lambda stage: object())
    monkeypatch.setattr(ext_mod, "_make_writer", lambda stage, mod_layer: _FakeWriter())

    # Drive the stage-event handler with an OPENED event (string-typed fake).
    ext._on_stage_event(SimpleNamespace(type="OPENED"))

    assert ext.enabled is True
    assert ext.brush is not None
    assert ext.panel.enabled is True
    assert ext.panel.status_message == ""


def test_stage_close_event_disables_the_brush(monkeypatch) -> None:
    """A stage-close event returns the tool to the disabled state.

    Validates: design "No active stage or viewport".
    """
    monkeypatch.setattr(ext_mod, "_get_stage", lambda: _FakeStage())
    monkeypatch.setattr(ext_mod, "_get_active_viewport", lambda: _FakeViewport())
    monkeypatch.setattr(ext_mod, "_resolve_mod_layer", lambda stage: object())
    monkeypatch.setattr(ext_mod, "_make_writer", lambda stage, mod_layer: _FakeWriter())

    ext = _make_extension()
    ext.on_startup("lightspeed.trex.scatter")
    assert ext.enabled is True
    assert ext.brush is not None

    ext._on_stage_event(SimpleNamespace(type="CLOSED"))

    assert ext.enabled is False
    assert ext.brush is None
    assert ext.panel.enabled is False
    assert ext.panel.status_message == NO_STAGE_MESSAGE


# --------------------------------------------------------------------------- #
# Scenario 2: Unresolved dependency at load time (Requirement 1.4)
# --------------------------------------------------------------------------- #


def test_unresolved_dependency_aborts_load_without_crashing(monkeypatch) -> None:
    """An import failure while wiring is reported and does not crash the host.

    Validates: Requirements 1.4
    """
    monkeypatch.setattr(ext_mod, "_get_stage", lambda: _FakeStage())

    def _boom() -> object:
        raise ImportError("No module named 'omni.kit.viewport.utility'")

    monkeypatch.setattr(ext_mod, "_get_active_viewport", _boom)

    ext = _make_extension()
    # Must not raise: the unresolved dependency is caught and reported.
    ext.on_startup("lightspeed.trex.scatter")

    assert ext.started is False
    assert ext.startup_error is not None
    assert "unresolved dependency" in ext.startup_error
    assert ext.enabled is False


def test_missing_mod_layer_at_load_stays_disabled(monkeypatch) -> None:
    """No writable mod layer at load -> loaded but disabled with the error.

    The writer raising ``ScatterLayerError`` during wiring (e.g. no mod layer)
    is recoverable: the tool stays loaded and disabled, showing the error, and
    re-enables when a writable mod layer becomes available (Requirement 10.5).

    Validates: Requirements 10.5
    """
    monkeypatch.setattr(ext_mod, "_get_stage", lambda: _FakeStage())
    monkeypatch.setattr(ext_mod, "_get_active_viewport", lambda: _FakeViewport())
    monkeypatch.setattr(ext_mod, "_resolve_mod_layer", lambda stage: None)

    def _no_mod_layer(stage, mod_layer):
        raise ScatterLayerError("a mod layer is required to author placements")

    monkeypatch.setattr(ext_mod, "_make_writer", _no_mod_layer)

    ext = _make_extension()
    ext.on_startup("lightspeed.trex.scatter")

    # Loaded (not a hard dependency abort) but disabled with the layer error.
    assert ext.started is True
    assert ext.enabled is False
    assert ext.brush is None
    assert ext.panel.enabled is False
    assert "mod layer is required" in ext.panel.status_message


# --------------------------------------------------------------------------- #
# Scenario 3: Missing / read-only mod layer at stroke time (Req 10.5 / 14.4)
# --------------------------------------------------------------------------- #


_SETTINGS = BrushSettings(
    radius=1.0,
    density=1.0,
    spacing=0.0,
    position_jitter=1.0,
    yaw_range=(0.0, 0.0),
    scale_range=(1.0, 1.0),
)


class _NonEmptyPalette:
    def is_empty(self) -> bool:
        return False


class _AlwaysHitRaycaster:
    def raycast(self, screen_pos: object) -> SurfaceHit:
        return SurfaceHit(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), prim_path="/ground")


class _FixedSampler:
    def sample(self, hit: object, settings: object, palette: object) -> List[object]:
        return [object(), object(), object()]


class _ReadOnlyAtBeginWriter:
    """Writer whose layer is read-only: ``ensure_scatter_layer`` always raises."""

    def __init__(self) -> None:
        self.open_count = 0
        self.close_count = 0
        self.author_count = 0

    def ensure_scatter_layer(self) -> object:
        raise ScatterLayerError("mod layer 'mod.usda' is read-only")

    def open_undo_group(self, label: str = "") -> None:
        self.open_count += 1

    def close_undo_group(self) -> None:
        self.close_count += 1

    def author_placements(self, placements: object) -> List[str]:
        self.author_count += 1
        return []


class _ReadOnlyAtAuthorWriter:
    """Writer that is writable at begin but raises when authoring placements."""

    def __init__(self) -> None:
        self.open_count = 0
        self.close_count = 0
        self.author_count = 0

    def ensure_scatter_layer(self) -> object:
        return object()  # writable at begin time

    def open_undo_group(self, label: str = "") -> None:
        self.open_count += 1

    def close_undo_group(self) -> None:
        self.close_count += 1

    def author_placements(self, placements: object) -> List[str]:
        self.author_count += 1
        raise ScatterLayerError("scatter layer 'scatter.usda' is read-only")


def _make_brush(writer: object):
    errors: List[str] = []
    brush = ScatterBrush(
        raycaster=_AlwaysHitRaycaster(),
        sampler=_FixedSampler(),
        palette=_NonEmptyPalette(),
        writer=writer,
        error_handler=errors.append,
    )
    brush.set_settings(_SETTINGS)
    return brush, errors


def test_read_only_layer_at_begin_aborts_before_opening_undo_group() -> None:
    """A read-only layer detected at ``begin_stroke`` aborts with no undo group.

    The stroke never opens an undo group and never authors -- layers are left
    unchanged -- and the error is surfaced via the error handler and
    :attr:`last_error`.

    Validates: Requirements 10.5, 14.4
    """
    writer = _ReadOnlyAtBeginWriter()
    brush, errors = _make_brush(writer)

    brush.begin_stroke((0.5, 0.5))

    # No undo group opened, nothing authored, stroke not active.
    assert writer.open_count == 0
    assert writer.close_count == 0
    assert writer.author_count == 0
    assert brush.is_active is False

    # The read-only layer error was surfaced.
    assert brush.last_error is not None
    assert "read-only" in brush.last_error
    assert errors == [brush.last_error]

    # A subsequent continue is a no-op (stroke never started).
    brush.continue_stroke((0.6, 0.6))
    assert writer.author_count == 0


def test_read_only_layer_at_author_aborts_stroke_and_closes_group() -> None:
    """A writer failure during authoring aborts the stroke and closes the group.

    ``begin_stroke`` opens exactly one undo group (the layer looked writable);
    the first ``continue_stroke`` then fails to author and the brush aborts:
    it closes the undo group cleanly (no open group leaks), surfaces the error,
    deactivates the stroke, and authors nothing further.

    Validates: Requirements 10.5, 14.4
    """
    writer = _ReadOnlyAtAuthorWriter()
    brush, errors = _make_brush(writer)

    brush.begin_stroke((0.5, 0.5))
    assert writer.open_count == 1
    assert brush.is_active is True

    brush.continue_stroke((0.6, 0.6))

    # The stroke aborted: the undo group was closed (not left open) and the
    # brush is no longer active.
    assert writer.close_count == 1
    assert brush.is_active is False
    assert writer.open_count == writer.close_count  # no dangling open group

    # The error was surfaced through both channels.
    assert brush.last_error is not None
    assert "read-only" in brush.last_error
    assert errors == [brush.last_error]

    # Further drag samples after the abort do nothing.
    before = writer.author_count
    brush.continue_stroke((0.7, 0.7))
    assert writer.author_count == before


def test_aborted_stroke_can_be_followed_by_a_clean_end_stroke() -> None:
    """Calling ``end_stroke`` after an abort is a harmless no-op.

    The abort already closed the single per-stroke undo group, so a trailing
    ``end_stroke`` (e.g. from a pointer-up after the failure) must not close a
    second time.

    Validates: Requirements 10.5
    """
    writer = _ReadOnlyAtAuthorWriter()
    brush, _ = _make_brush(writer)

    brush.begin_stroke((0.5, 0.5))
    brush.continue_stroke((0.6, 0.6))
    assert writer.close_count == 1

    # Trailing pointer-up: no second close.
    brush.end_stroke()
    assert writer.close_count == 1
