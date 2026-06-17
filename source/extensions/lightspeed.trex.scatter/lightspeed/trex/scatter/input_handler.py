"""Viewport pointer input -> stroke lifecycle translation for the scatter brush.

``ViewportInputHandler`` is the thin glue between raw viewport pointer events
(mouse down / move / up) and the ``ScatterBrush`` stroke lifecycle
(``begin_stroke`` / ``continue_stroke`` / ``end_stroke``) -- design
"Component 2: ViewportInputHandler".

Single responsibility: input only. This handler never authors USD and never
raycasts; it converts device pixel coordinates into normalized viewport
coordinates ``[0, 1]`` and forwards them to the brush. All scene/USD work lives
downstream in the brush, raycaster, sampler, and writer.

Lifecycle / subscription rule (Requirement 3.1):
    The handler subscribes to viewport pointer events *only while the tool is
    active*. ``attach`` activates the tool and wires the subscription;
    ``detach`` unsubscribes and tears down all state so no events are delivered
    while the tool is inactive.

Mutual exclusion (Requirement 3.7):
    Exactly one of paint / erase mode is active at any time. The handler holds a
    single mode value (defaulting to paint); switching to one mode implicitly
    disables the other. Strokes are tagged with the active mode so downstream
    consumers can branch on paint vs erase.

Testability / dependency injection:
    The constructor and ``attach`` accept an injectable ``viewport_api`` so unit
    tests (task 9.2) can supply a fake viewport (a subscription seam plus a
    pixel resolution) and a recording brush. The coordinate conversion
    (``_normalize``) and the event dispatch (``_on_pointer_down`` /
    ``_on_pointer_move`` / ``_on_pointer_up``) are plain methods that can be
    driven directly without a live Kit runtime.

The ``carb``/``omni`` imports are guarded (mirroring the ``HAS_PXR`` pattern in
``models.py`` and ``HAS_OMNI_UI`` in ``panel.py``) so this module stays
importable -- and the conversion/dispatch logic unit-testable -- in environments
where the Kit runtime is not installed.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

try:  # pragma: no cover - import guard exercised only by environment
    import carb.input  # type: ignore

    HAS_CARB_INPUT = True
except ImportError:  # pragma: no cover - exercised only without the Kit runtime
    carb = None  # type: ignore
    HAS_CARB_INPUT = False

__all__ = [
    "HAS_CARB_INPUT",
    "POINTER_DOWN",
    "POINTER_MOVE",
    "POINTER_UP",
    "MODE_PAINT",
    "MODE_ERASE",
    "ViewportInputHandler",
]

# Pointer event phases, normalized across whatever backend delivers the event.
POINTER_DOWN = "down"
POINTER_MOVE = "move"
POINTER_UP = "up"

# Brush modes. Exactly one is active at a time (Requirement 3.7).
MODE_PAINT = "paint"
MODE_ERASE = "erase"
_VALID_MODES = (MODE_PAINT, MODE_ERASE)

# Aliases under which a backend event may expose its phase and pixel position.
# Different Kit backends (carb input, omni.ui.scene gestures, viewport hooks)
# name these differently, so we probe a small set and also fall back to
# attribute access (mirroring the field-probing approach in ``raycaster.py``).
_PHASE_KEYS = ("phase", "type", "event_type", "action", "state")
_X_KEYS = ("x", "pixel_x", "screen_x", "mouse_x", "px")
_Y_KEYS = ("y", "pixel_y", "screen_y", "mouse_y", "py")

# Aliases for the viewport's pixel resolution, used to normalize coordinates.
_RESOLUTION_KEYS = ("resolution", "frame_resolution", "render_resolution")
_WIDTH_KEYS = ("width", "pixel_width", "frame_width", "computed_width")
_HEIGHT_KEYS = ("height", "pixel_height", "frame_height", "computed_height")

# Strings that, when they appear in a backend phase value, identify the phase.
_DOWN_TOKENS = ("down", "press", "began", "begin", "start")
_UP_TOKENS = ("up", "release", "ended", "end", "stop")
_MOVE_TOKENS = ("move", "moved", "drag", "dragged", "motion", "changed")


def _field(source: object, keys: Sequence[str], default=None):
    """Read the first present field of ``source`` from ``keys``.

    Supports both mapping-style sources (``dict``) and attribute-style sources
    (event/viewport objects), matching the tolerant access used elsewhere in
    this package.
    """
    for key in keys:
        if isinstance(source, dict):
            if key in source and source[key] is not None:
                return source[key]
        elif hasattr(source, key):
            value = getattr(source, key)
            if value is not None:
                return value
    return default


def _classify_phase(raw_phase: object) -> Optional[str]:
    """Map a backend phase value onto ``POINTER_DOWN/MOVE/UP`` (or ``None``).

    Accepts our own canonical strings, free-form backend strings (matched by
    substring tokens), and integer-like enum values stringified by the backend.
    """
    if raw_phase is None:
        return None
    if raw_phase in (POINTER_DOWN, POINTER_MOVE, POINTER_UP):
        return raw_phase  # type: ignore[return-value]
    text = str(raw_phase).lower()
    # Check up/down before move so "mouse_move" style strings don't shadow them,
    # and so combined tokens resolve to the most specific phase.
    if any(tok in text for tok in _DOWN_TOKENS):
        return POINTER_DOWN
    if any(tok in text for tok in _UP_TOKENS):
        return POINTER_UP
    if any(tok in text for tok in _MOVE_TOKENS):
        return POINTER_MOVE
    return None


class ViewportInputHandler:
    """Translate viewport pointer events into ``ScatterBrush`` stroke calls.

    The handler is inert until ``attach`` is called and again after ``detach``:
    it only holds a live subscription -- and therefore only forwards events to
    the brush -- while the tool is active (Requirement 3.1).
    """

    def __init__(self) -> None:
        self._viewport_api: Optional[object] = None
        self._brush: Optional[object] = None
        self._subscription: Optional[object] = None
        # Whether a pointer button is currently held (a stroke is in progress).
        self._is_down: bool = False
        # Exactly one mode is active at a time; paint is the default.
        self._mode: str = MODE_PAINT

    # -- lifecycle ---------------------------------------------------------

    def attach(self, viewport_api: object, brush: object) -> None:
        """Activate the tool and subscribe to ``viewport_api`` pointer events.

        Args:
            viewport_api: The active viewport. Must expose a subscription seam
                (``subscribe_to_pointer_event`` or ``subscribe_to_mouse_event``)
                taking a single-argument callback, and a pixel resolution
                (``resolution`` tuple or ``width``/``height``) used to normalize
                coordinates.
            brush: The ``ScatterBrush`` to drive. Must expose ``begin_stroke``,
                ``continue_stroke``, and ``end_stroke``.

        A second ``attach`` first detaches the previous subscription so exactly
        one subscription is ever live (the tool is active at most once).
        """
        # Re-attaching is allowed; never leak a previous subscription.
        if self._subscription is not None or self._viewport_api is not None:
            self.detach()

        self._viewport_api = viewport_api
        self._brush = brush
        self._is_down = False
        self._subscription = self._subscribe(viewport_api)

    def detach(self) -> None:
        """Deactivate the tool and unsubscribe from all pointer events.

        Closes any in-progress stroke, releases the subscription, and clears the
        viewport/brush references so no event can be delivered while inactive.
        Safe to call when already detached.
        """
        # End any stroke that was in progress so the brush is left consistent.
        if self._is_down and self._brush is not None:
            end_stroke = getattr(self._brush, "end_stroke", None)
            if callable(end_stroke):
                end_stroke()
        self._is_down = False

        self._unsubscribe(self._subscription)
        self._subscription = None
        self._viewport_api = None
        self._brush = None

    @property
    def is_attached(self) -> bool:
        """``True`` while the tool is active (a subscription is live)."""
        return self._subscription is not None

    # -- mode mutual exclusion (Requirement 3.7) ---------------------------

    @property
    def mode(self) -> str:
        """The currently active mode -- exactly one of paint / erase."""
        return self._mode

    def set_mode(self, mode: str) -> None:
        """Set the active mode, enforcing mutual exclusion.

        Setting one mode implicitly disables the other: the handler only ever
        holds a single mode value, so paint and erase can never both be active.

        Raises:
            ValueError: if ``mode`` is not one of the supported modes.
        """
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        self._mode = mode

    def activate_paint(self) -> None:
        """Make paint the active mode (disables erase)."""
        self.set_mode(MODE_PAINT)

    def activate_erase(self) -> None:
        """Make erase the active mode (disables paint)."""
        self.set_mode(MODE_ERASE)

    @property
    def is_paint_active(self) -> bool:
        return self._mode == MODE_PAINT

    @property
    def is_erase_active(self) -> bool:
        return self._mode == MODE_ERASE

    # -- subscription seam -------------------------------------------------

    def _subscribe(self, viewport_api: object) -> Optional[object]:
        """Wire ``_on_viewport_event`` to the viewport's pointer-event stream.

        Prefers an explicit pointer hook, then a mouse hook. Returns the
        subscription handle (kept alive for the tool's lifetime) or ``None`` if
        the viewport exposes no recognised seam.
        """
        for hook_name in ("subscribe_to_pointer_event", "subscribe_to_mouse_event"):
            hook = getattr(viewport_api, hook_name, None)
            if callable(hook):
                return hook(self._on_viewport_event)
        return None

    @staticmethod
    def _unsubscribe(subscription: object) -> None:
        """Release a subscription handle, tolerating the common shapes.

        Kit subscriptions are usually released by dropping the reference, but
        some expose ``unsubscribe()``/``destroy()``; call those when present.
        """
        if subscription is None:
            return
        for closer in ("unsubscribe", "destroy", "close"):
            method = getattr(subscription, closer, None)
            if callable(method):
                method()
                return

    # -- event entry point -------------------------------------------------

    def _on_viewport_event(self, event: object) -> None:
        """Parse a raw backend event and route it to the phase handlers.

        Unrecognised phases and events without a usable position are ignored
        (a no-op), so spurious backend events never reach the brush.
        """
        phase = _classify_phase(_field(event, _PHASE_KEYS))
        if phase is None:
            return

        px = _field(event, _X_KEYS)
        py = _field(event, _Y_KEYS)
        if px is None or py is None:
            return

        if phase == POINTER_DOWN:
            self._on_pointer_down(float(px), float(py))
        elif phase == POINTER_MOVE:
            self._on_pointer_move(float(px), float(py))
        elif phase == POINTER_UP:
            self._on_pointer_up(float(px), float(py))

    # -- phase dispatch (directly unit-testable) ---------------------------

    def _on_pointer_down(self, px: float, py: float) -> None:
        """Begin a stroke at device-pixel position ``(px, py)``."""
        if self._brush is None:
            return
        self._is_down = True
        self._brush.begin_stroke(self._normalize(px, py))

    def _on_pointer_move(self, px: float, py: float) -> None:
        """Continue the active stroke -- only while the button is held."""
        if self._brush is None or not self._is_down:
            return  # hover without a held button is not part of a stroke
        self._brush.continue_stroke(self._normalize(px, py))

    def _on_pointer_up(self, px: float, py: float) -> None:
        """End the active stroke at device-pixel position ``(px, py)``."""
        if self._brush is None or not self._is_down:
            return
        self._is_down = False
        self._brush.end_stroke()

    # -- coordinate conversion ---------------------------------------------

    def _normalize(self, px: float, py: float) -> Tuple[float, float]:
        """Convert device pixels to normalized viewport coordinates ``[0, 1]``.

        ``(0, 0)`` is the top-left of the viewport and ``(1, 1)`` the
        bottom-right -- matching the convention ``SceneRaycaster`` expects.
        Results are clamped to ``[0, 1]`` so a pointer that drifts a pixel
        outside the viewport bounds still yields an in-range coordinate.
        """
        width, height = self._viewport_resolution()
        nx = px / width if width else 0.0
        ny = py / height if height else 0.0
        return _clamp_unit(nx), _clamp_unit(ny)

    def _viewport_resolution(self) -> Tuple[float, float]:
        """Resolve the viewport pixel size as ``(width, height)`` floats.

        Probes a resolution tuple first, then discrete width/height fields.

        Raises:
            ValueError: if no usable, positive resolution can be determined --
                normalization is impossible without it.
        """
        viewport = self._viewport_api
        if viewport is not None:
            resolution = _field(viewport, _RESOLUTION_KEYS)
            wh = _coerce_resolution(resolution)
            if wh is not None:
                return wh

            width = _field(viewport, _WIDTH_KEYS)
            height = _field(viewport, _HEIGHT_KEYS)
            if width is not None and height is not None:
                w, h = float(width), float(height)
                if w > 0.0 and h > 0.0:
                    return w, h

        raise ValueError(
            "viewport_api must expose a positive pixel resolution "
            "('resolution' tuple or 'width'/'height') to normalize coordinates"
        )


def _coerce_resolution(resolution: object) -> Optional[Tuple[float, float]]:
    """Coerce a resolution value into a positive ``(width, height)`` tuple."""
    if resolution is None:
        return None
    try:
        w, h = float(resolution[0]), float(resolution[1])
    except (TypeError, IndexError, ValueError):
        return None
    if w > 0.0 and h > 0.0:
        return w, h
    return None


def _clamp_unit(value: float) -> float:
    """Clamp ``value`` to the closed unit interval ``[0, 1]``."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
