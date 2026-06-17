"""Stroke orchestration for the scatter brush.

``ScatterBrush`` owns the paint-stroke lifecycle and is the conductor that turns
viewport input into authored prims. It holds no USD itself: it coordinates the
focused components -- ``SceneRaycaster`` (cursor -> world hit),
``PlacementSampler`` (hit -> placements), ``AssetPalette`` (which asset to
place), and ``ScatterLayerWriter`` (author + undo) -- and enforces the stroke
rules from the design (design "Component 3: ScatterBrush"):

- **One undo group per stroke**: ``begin_stroke`` opens exactly one
  ``omni.kit.undo`` group via the writer and ``end_stroke`` closes it, so a
  whole stroke is a single undo step (Requirement 11.1, design Property 6).
- **Empty palette is rejected up front**: ``begin_stroke`` on an empty palette
  surfaces a user-facing warning and authors nothing -- crucially *without*
  opening an undo group (design Property 8 / "Scenario: Empty palette at stroke
  start").
- **Frame-rate-independent spacing**: ``continue_stroke`` throttles authoring by
  world-space distance, not frame count, so density depends on distance painted
  (design "Stroke continuation with frame-rate-independent spacing",
  Requirements 3.2/3.3).
- **Density cap**: the total instances authored in a stroke never exceeds
  ``settings.max_instances_per_stroke`` (design Property 1, Requirement 3.3).
- **Miss is a no-op**: a ``continue_stroke`` whose raycast misses authors zero
  prims and does not advance ``_last_hit`` (design Property 7, Requirement 3.4).

This module deliberately performs **no direct USD authoring** and imports no
``pxr``/``omni`` modules -- all USD work is delegated to the writer, and the only
geometric operation (distance between two hit points) is done through a small
duck-typed helper that works with both ``pxr.Gf`` vectors and plain sequences.
That keeps the module importable and unit-testable in environments without USD
or Kit (mirroring the ``HAS_PXR`` guarding used across the package).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from .models import BrushSettings, SurfaceHit
from .palette import AssetPalette
from .writer import ScatterLayerError

__all__ = ["ScatterBrush", "DEFAULT_BRUSH_SETTINGS", "EMPTY_PALETTE_WARNING"]

_LOGGER = logging.getLogger(__name__)

# Label applied to the per-stroke undo group (parity with the design interface;
# omni.kit.undo groups are otherwise unnamed).
_UNDO_GROUP_LABEL = "Scatter stroke"

# User-facing message surfaced when a stroke is attempted with no enabled asset
# (design "Scenario: Empty palette at stroke start", Requirements 2.6 / 9.7).
EMPTY_PALETTE_WARNING = "Add at least one asset to the palette before painting."

# Prefix applied to the user-facing message when authoring fails for a reason
# other than a recognised ``ScatterLayerError`` (design "Scenario: Scatter layer
# cannot be created or is read-only").
_AUTHORING_FAILED_PREFIX = "Scatter authoring failed: "

# Sensible default settings so the brush is usable before the panel has pushed a
# snapshot via ``set_settings`` -- the brush is never left without settings
# (defensive: ``continue_stroke`` reads ``spacing`` and the density cap). These
# defaults satisfy ``BrushSettings`` validation.
DEFAULT_BRUSH_SETTINGS = BrushSettings(
    radius=1.0,
    density=1.0,
    spacing=0.0,
    position_jitter=1.0,
    yaw_range=(0.0, 360.0),
    scale_range=(1.0, 1.0),
)


def _world_distance(a: object, b: object) -> float:
    """Return the world-space distance between two hit points ``a`` and ``b``.

    Mirrors the design pseudocode's ``(hit.point - last.point).GetLength()`` but
    is defensive about the point type: it uses ``pxr.Gf`` vector subtraction +
    ``GetLength`` when available, and falls back to a component-wise Euclidean
    distance for plain sequences (so the brush stays testable without USD).
    """
    # Fast path: pxr.Gf vectors support ``-`` and ``GetLength``.
    try:
        diff = a - b  # type: ignore[operator]
        get_length = getattr(diff, "GetLength", None)
        if callable(get_length):
            return float(get_length())
    except TypeError:
        pass
    # Fallback: treat both as 3-component sequences.
    return float(sum((float(ca) - float(cb)) ** 2 for ca, cb in zip(a, b))) ** 0.5


class ScatterBrush:
    """Orchestrate a paint stroke from input through to authored prims.

    Args:
        raycaster: Resolves a cursor position to a world-space ``SurfaceHit``
            (or ``None`` on a miss).
        sampler: Produces placements for a given hit + settings + palette.
        palette: The enabled asset set; an empty palette blocks a stroke.
        writer: Authors placements and owns the per-stroke undo group.
        warning_handler: Optional callback invoked with a user-facing message
            when a stroke is rejected (e.g. empty palette). The panel wires this
            to its notification surface; tests can inject a recorder. Whether or
            not a handler is set, the message is also logged and exposed via
            :attr:`last_warning`.
        error_handler: Optional callback invoked with a user-facing message when
            a stroke is aborted by a writer failure (e.g. a missing or read-only
            mod/scatter layer, design Requirements 10.5 / 14.4). Like the
            warning handler the message is also logged and exposed via
            :attr:`last_error`.
    """

    def __init__(
        self,
        raycaster: object,
        sampler: object,
        palette: AssetPalette,
        writer: object,
        warning_handler: Optional[Callable[[str], None]] = None,
        error_handler: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._raycaster = raycaster
        self._sampler = sampler
        self._palette = palette
        self._writer = writer
        self._warning_handler = warning_handler
        self._error_handler = error_handler

        # Stroke state.
        self._settings: BrushSettings = DEFAULT_BRUSH_SETTINGS
        self._active: bool = False
        self._authored_this_stroke: int = 0
        self._last_hit: Optional[SurfaceHit] = None
        # Most recent user-facing warning (testable flag; ``None`` when clear).
        self._last_warning: Optional[str] = None
        # Most recent user-facing error (testable flag; ``None`` when clear).
        self._last_error: Optional[str] = None

    # -- configuration ---------------------------------------------------

    def set_settings(self, settings: BrushSettings) -> None:
        """Store the active brush settings used for spacing, sampling, and cap.

        Accepts a validated :class:`BrushSettings` snapshot (typically pushed by
        the panel via ``on_settings_changed``). Changing settings mid-stroke is
        allowed; the new values apply to subsequent ``continue_stroke`` calls.
        """
        if not isinstance(settings, BrushSettings):
            raise TypeError(f"settings must be a BrushSettings, got {settings!r}")
        self._settings = settings

    def set_warning_handler(self, handler: Optional[Callable[[str], None]]) -> None:
        """Set (or clear) the callback used to surface user-facing warnings."""
        self._warning_handler = handler

    def set_error_handler(self, handler: Optional[Callable[[str], None]]) -> None:
        """Set (or clear) the callback used to surface user-facing errors.

        Errors are surfaced when a stroke is aborted by a writer failure such as
        a missing or read-only mod/scatter layer (Requirements 10.5 / 14.4).
        """
        self._error_handler = handler

    @property
    def last_warning(self) -> Optional[str]:
        """The most recent user-facing warning, or ``None`` if none is pending."""
        return self._last_warning

    @property
    def last_error(self) -> Optional[str]:
        """The most recent user-facing error, or ``None`` if none is pending."""
        return self._last_error

    @property
    def is_active(self) -> bool:
        """Whether a stroke is currently in progress."""
        return self._active

    @property
    def authored_this_stroke(self) -> int:
        """Count of instances authored so far in the current/last stroke."""
        return self._authored_this_stroke

    # -- stroke lifecycle ------------------------------------------------

    def begin_stroke(self, screen_pos: object) -> None:
        """Begin a paint stroke, opening exactly one undo group.

        If the palette is empty, the stroke is rejected *before* any undo group
        is opened: a warning is surfaced and nothing is authored (design
        Property 8). Otherwise the stroke state is reset and a single undo group
        is opened so the whole stroke collapses to one undo step (Requirement
        11.1).

        Args:
            screen_pos: The pointer position where the stroke starts. Accepted
                for interface parity; the first placement is authored by the
                subsequent ``continue_stroke`` call (which raycasts), matching
                the design pseudocode.
        """
        if self._active:
            # Defensive: a new begin without an end closes the dangling group
            # first so we never nest or leak undo groups.
            self.end_stroke()

        if self._palette.is_empty():
            # Reject before opening an undo group -- nothing is authored and no
            # empty undo step is created.
            self._surface_warning(EMPTY_PALETTE_WARNING)
            return

        # Validate the scatter/mod layer is available and writable BEFORE opening
        # an undo group, so a missing or read-only mod layer aborts the stroke
        # leaving every layer unchanged and creating no empty undo step (design
        # "Scenario: Scatter layer cannot be created or is read-only",
        # Requirements 10.5 / 14.4).
        if not self._ensure_layer_writable():
            return

        self._clear_warning()
        self._last_error = None
        self._active = True
        self._authored_this_stroke = 0
        self._last_hit = None
        # Exactly one undo group per stroke (Requirement 11.1).
        self._writer.open_undo_group(_UNDO_GROUP_LABEL)

    def continue_stroke(self, screen_pos: object) -> None:
        """Process one drag sample of an active stroke.

        Spacing is enforced in world space so the number of placements depends
        on distance painted, not on how many frames the pointer moved (design
        "Stroke continuation with frame-rate-independent spacing").

        No-ops (author nothing, leave ``_last_hit`` unchanged) when: the stroke
        is not active; the raycast misses (Property 7); the hit is closer than
        ``spacing`` to the last accepted hit; or the density cap is reached
        (Property 1). Otherwise samples placements, truncates to the remaining
        cap, authors them, advances the authored count, and advances
        ``_last_hit`` (only on an accepted, authored hit).
        """
        if not self._active:
            return

        hit = self._raycaster.raycast(screen_pos)
        if hit is None:
            return  # painting over empty space is a no-op, not an error (Property 7)

        if self._last_hit is not None:
            moved = _world_distance(hit.point, self._last_hit.point)
            if moved < self._settings.spacing:
                return  # too close to the previous stamp; skip to keep density even

        if self._authored_this_stroke >= self._settings.max_instances_per_stroke:
            return  # safety cap reached; stop adding for this stroke (Property 1)

        placements = self._sampler.sample(hit, self._settings, self._palette)
        remaining = self._settings.max_instances_per_stroke - self._authored_this_stroke
        placements = placements[:remaining]

        try:
            self._writer.author_placements(placements)
        except ScatterLayerError as exc:
            # Missing / read-only mod or scatter layer: abort the stroke leaving
            # every layer unchanged and surface a clear error (Req 10.5 / 14.4).
            self._abort_stroke(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - any writer failure aborts the stroke
            # Any other authoring failure is treated the same way: abort, close
            # the undo group cleanly, and surface the error (design Req 3.6).
            self._abort_stroke(f"{_AUTHORING_FAILED_PREFIX}{exc}")
            return

        self._authored_this_stroke += len(placements)
        # Advance the spacing anchor only on an accepted hit (design
        # Postconditions; keeps Property 7's "miss does not advance" invariant).
        self._last_hit = hit

    def end_stroke(self) -> None:
        """Finish the stroke, closing the undo group opened by ``begin_stroke``.

        Idempotent and safe to call when no stroke is active (no undo group is
        closed in that case), so a stray pointer-up never closes a group the
        brush did not open.
        """
        if not self._active:
            return
        self._active = False
        # Close the single per-stroke undo group (Requirement 11.1).
        self._writer.close_undo_group()

    # -- warnings --------------------------------------------------------

    def _surface_warning(self, message: str) -> None:
        """Record, log, and forward a user-facing warning to the handler."""
        self._last_warning = message
        _LOGGER.warning(message)
        if self._warning_handler is not None:
            self._warning_handler(message)

    def _clear_warning(self) -> None:
        """Clear any pending warning at the start of a successful stroke."""
        self._last_warning = None

    # -- errors ----------------------------------------------------------

    def _ensure_layer_writable(self) -> bool:
        """Return whether the scatter/mod layer is available and writable.

        Probes the writer's ``ensure_scatter_layer`` (when present) so a missing
        or read-only mod/scatter layer is detected *before* an undo group is
        opened. On failure the error is surfaced and ``False`` is returned so
        ``begin_stroke`` aborts without opening a group or authoring anything,
        leaving every layer unchanged (Requirements 10.5 / 14.4).

        Writers that do not expose ``ensure_scatter_layer`` (e.g. recording test
        fakes) are assumed writable; any layer failure then surfaces at author
        time in :meth:`continue_stroke` instead.
        """
        ensure = getattr(self._writer, "ensure_scatter_layer", None)
        if not callable(ensure):
            return True
        try:
            ensure()
            return True
        except ScatterLayerError as exc:
            self._surface_error(str(exc))
            return False
        except Exception as exc:  # noqa: BLE001 - any layer failure blocks the stroke
            self._surface_error(f"Scatter layer unavailable: {exc}")
            return False

    def _abort_stroke(self, message: str) -> None:
        """Abort the active stroke after a writer failure; surface the error.

        Closes the per-stroke undo group cleanly so no open group leaks, marks
        the stroke inactive (so subsequent ``continue_stroke`` calls are
        no-ops), resets the spacing anchor, and surfaces the error. Because the
        writer authors all-or-nothing per call, no partial authoring remains
        when this is reached for a missing/read-only layer.
        """
        was_active = self._active
        self._active = False
        self._last_hit = None
        if was_active:
            try:
                self._writer.close_undo_group()
            except Exception:  # noqa: BLE001 - closing the group must not raise
                _LOGGER.exception(
                    "Scatter brush: error closing undo group while aborting stroke"
                )
        self._surface_error(message)

    def _surface_error(self, message: str) -> None:
        """Record, log, and forward a user-facing error to the handler."""
        self._last_error = message
        _LOGGER.error(message)
        if self._error_handler is not None:
            self._error_handler(message)
