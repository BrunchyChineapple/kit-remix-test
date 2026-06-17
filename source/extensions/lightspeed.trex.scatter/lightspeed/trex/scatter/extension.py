"""The ``omni.ext.IExt`` entry point that wires the scatter brush together.

``ScatterBrushExtension`` is the Kit lifecycle object the host instantiates when
the ``lightspeed.trex.scatter`` extension is enabled. ``on_startup`` constructs
and connects every component described in the design ("Example Usage" /
"Components") -- stage, ``AssetPalette``, ``ScatterLayerWriter``,
``PlacementSampler``, ``SceneRaycaster``, ``ScatterBrush``, ``BrushPanel`` and
``ViewportInputHandler`` -- registers the tool entry point, and opens the
``Brush_UI`` (Requirement 1.6). ``on_shutdown`` tears everything back down.

Defensive startup (Requirements 1.3 / 1.4)
    The whole wiring sequence is wrapped in ``try``/``except`` so a missing
    stage/viewport or an unresolved optional dependency aborts *this* extension's
    load gracefully -- it reports the problem and leaves the host StageCraft
    application running rather than crashing it. (Deeper, control-disabling
    error handling and stage-event re-enable belong to task 10.2; this entry
    point only needs to fail soft.)

Tool registration with fallback (Requirement 1.7)
    The extension first tries to register against a host tool/tab registration
    point. If none is available, it falls back to a self-provided ``omni.ui``
    window entry point and reports that the host registration point was
    unavailable -- exposed via :attr:`host_registration_available`.

Import guarding
    ``omni.ext``, ``omni.ui``, ``omni.usd`` and ``omni.kit.viewport.utility`` are
    all imported behind guards (mirroring ``HAS_PXR`` / ``HAS_OMNI_UI`` used
    across the package), so this module stays importable -- and the wiring logic
    inspectable -- without the Kit runtime installed.
"""

from __future__ import annotations

import logging
import random
from typing import Callable, List, Optional

from .brush import ScatterBrush
from .input_handler import ViewportInputHandler
from .palette import AssetPalette
from .panel import BrushPanel
from .raycaster import SceneRaycaster
from .sampler import PlacementSampler
from .writer import ScatterLayerError, ScatterLayerWriter

try:  # pragma: no cover - import guard exercised only by environment
    import omni.ext  # type: ignore

    HAS_OMNI_EXT = True
except ImportError:  # pragma: no cover - exercised only without the Kit runtime
    omni = None  # type: ignore
    HAS_OMNI_EXT = False

try:  # pragma: no cover - import guard exercised only by environment
    import omni.ui as ui  # type: ignore

    HAS_OMNI_UI = True
except ImportError:  # pragma: no cover - exercised only without the Kit runtime
    ui = None  # type: ignore
    HAS_OMNI_UI = False

try:  # pragma: no cover - import guard exercised only by environment
    import omni.usd  # type: ignore

    HAS_OMNI_USD = True
except ImportError:  # pragma: no cover - exercised only without the Kit runtime
    HAS_OMNI_USD = False

__all__ = ["ScatterBrushExtension", "WINDOW_TITLE", "MENU_PATH"]

_LOGGER = logging.getLogger(__name__)

# Title of the self-provided fallback window and the host tool/menu entry.
WINDOW_TITLE = "Scatter Brush"
# Host menu location used when a menu-based registration point is available.
MENU_PATH = "Tools"

# Message shown in the panel while the tool is disabled because no stage (and
# therefore no viewport target) is available (design "Scenario: No active stage
# or viewport"). The tool re-enables automatically when a stage opens.
NO_STAGE_MESSAGE = "Open a stage to use the scatter brush."


class _DependencyError(RuntimeError):
    """An extension dependency could not be resolved at load time.

    Raised when a declared dependency (e.g. the viewport utility extension or
    the ``pxr`` USD libraries) fails to import while wiring the brush. The
    extension catches it in :meth:`ScatterBrushExtension.on_startup`, reports the
    unresolved dependency, and aborts *its own* load without crashing the host
    StageCraft application (Requirement 1.4).
    """


def _make_writer(stage: object, mod_layer: object) -> ScatterLayerWriter:
    """Construct the scatter-layer writer.

    Factored into a module function so the wiring is easy to follow and so tests
    can monkeypatch writer construction with a fake (the real writer needs the
    ``pxr`` USD runtime).
    """
    return ScatterLayerWriter(stage, mod_layer)  # type: ignore[arg-type]


def _get_stage() -> object:
    """Return the active USD stage from the ``omni.usd`` context, or ``None``.

    Isolated as a module function so the wiring is easy to follow and so a test
    or a host integration can monkeypatch stage resolution without a live Kit
    runtime.
    """
    if not HAS_OMNI_USD:
        return None
    context = omni.usd.get_context()  # type: ignore[union-attr]
    if context is None:
        return None
    return context.get_stage()


def _get_active_viewport() -> object:
    """Return the active viewport API via ``omni.kit.viewport.utility``.

    Imported lazily and guarded so the module imports without the viewport
    extension present (the import only needs to succeed inside ``on_startup``).
    """
    from omni.kit.viewport.utility import get_active_viewport  # type: ignore

    return get_active_viewport()


def _resolve_mod_layer(stage: object) -> object:
    """Best-effort resolution of the mod/replacement layer to author into.

    Prefers the stage's current edit-target layer (the layer the host has made
    the active authoring target, which in an open RTX Remix project is the mod
    layer), falling back to the root layer. Returns ``None`` if neither can be
    resolved. Robust mod-layer validation (read-only / missing) is task 10.2;
    here we only need a sensible target so the writer can be constructed.
    """
    if stage is None:
        return None
    get_edit_target = getattr(stage, "GetEditTarget", None)
    if callable(get_edit_target):
        try:
            edit_target = get_edit_target()
            layer = edit_target.GetLayer() if edit_target is not None else None
        except Exception:  # pragma: no cover - defensive against host quirks
            layer = None
        get_session = getattr(stage, "GetSessionLayer", None)
        session = get_session() if callable(get_session) else None
        if layer is not None and layer is not session:
            return layer
    get_root = getattr(stage, "GetRootLayer", None)
    if callable(get_root):
        return get_root()
    return None


# Base class: subclass ``omni.ext.IExt`` when Kit is present so the host drives
# the lifecycle; fall back to ``object`` so the module imports (and the wiring
# can be inspected) without the Kit runtime.
_ExtBase = omni.ext.IExt if HAS_OMNI_EXT else object  # type: ignore[attr-defined]


class ScatterBrushExtension(_ExtBase):  # type: ignore[valid-type, misc]
    """Kit extension entry point that constructs and wires the scatter brush.

    The host calls :meth:`on_startup` when the extension is enabled and
    :meth:`on_shutdown` when it is disabled or the app exits.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        # ``omni.ext.IExt.__init__`` takes no required args; forward defensively
        # so this works both as a real IExt subclass and as a plain object.
        try:
            super().__init__(*args, **kwargs)  # type: ignore[misc]
        except TypeError:  # pragma: no cover - object() takes no args
            super().__init__()

        self._ext_id: Optional[str] = None

        # Wired components (populated by on_startup).
        self._stage: object = None
        self._palette: Optional[AssetPalette] = None
        self._writer: Optional[ScatterLayerWriter] = None
        self._sampler: Optional[PlacementSampler] = None
        self._raycaster: Optional[SceneRaycaster] = None
        self._brush: Optional[ScatterBrush] = None
        self._panel: Optional[BrushPanel] = None
        self._input: Optional[ViewportInputHandler] = None
        self._viewport: object = None

        # Entry-point state.
        self._window: object = None
        self._menu_items: Optional[List[object]] = None
        self._host_registration_available: bool = False
        self._started: bool = False
        self._startup_error: Optional[str] = None

        # Enable/disable + stage-event state (task 10.2). ``_enabled`` tracks
        # whether the brush is wired against a live stage/viewport;
        # ``_stage_event_sub`` holds the omni.usd stage-event subscription that
        # drives automatic re-enable on stage open.
        self._enabled: bool = False
        self._stage_event_sub: object = None

    # ------------------------------------------------------------------
    # Public inspection (useful for task 10.2/10.3 and host integration)
    # ------------------------------------------------------------------
    @property
    def host_registration_available(self) -> bool:
        """Whether a host tool/tab registration point was found (Req 1.7)."""
        return self._host_registration_available

    @property
    def started(self) -> bool:
        """Whether startup wiring completed successfully."""
        return self._started

    @property
    def startup_error(self) -> Optional[str]:
        """The startup failure message, or ``None`` if startup succeeded."""
        return self._startup_error

    @property
    def enabled(self) -> bool:
        """Whether the brush is wired and active against a live stage/viewport.

        ``False`` while the tool is loaded but disabled (no stage/viewport yet);
        flips to ``True`` once a stage opens and the brush is wired (task 10.2).
        """
        return self._enabled

    @property
    def brush(self) -> Optional[ScatterBrush]:
        """The wired ``ScatterBrush`` (``None`` before startup / on failure)."""
        return self._brush

    @property
    def panel(self) -> Optional[BrushPanel]:
        """The wired ``BrushPanel`` (``None`` before startup / on failure)."""
        return self._panel

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def on_startup(self, ext_id: Optional[str] = None) -> None:
        """Load the extension, then wire the brush if a stage is available.

        The load is resilient (Requirements 1.3 / 1.4 and design "Scenario: No
        active stage or viewport"):

        * The panel and entry point are always created so the tool is visible.
        * If no stage/viewport is available yet, the extension stays loaded but
          *disabled* -- the panel shows "Open a stage to use the scatter brush"
          and a stage-event subscription re-enables it automatically when a
          stage opens.
        * If a declared dependency cannot be resolved (an import fails while
          wiring), the load is aborted, the unresolved dependency is reported
          via :attr:`startup_error`, and the host is *not* crashed.
        """
        self._ext_id = ext_id
        self._startup_error = None
        try:
            # Always create the panel (settings sink registered once) so a
            # disabled state can be shown before any stage exists.
            self._panel = BrushPanel()
            self._panel.on_settings_changed(self._forward_settings)

            self._register_entry_point()
            # Open the Brush_UI on activation (Requirement 1.6).
            self._activate()

            # Re-enable automatically when a stage opens (stage-event sub).
            self._subscribe_stage_events()

            # Wire against the current stage/viewport if one exists; otherwise
            # enter the disabled state (recoverable on stage open).
            self._try_enable()
            self._started = True
        except _DependencyError as exc:
            # Unresolved dependency: abort this extension's load, report it, and
            # leave the host running (Requirement 1.4).
            self._startup_error = f"unresolved dependency: {exc}"
            _LOGGER.error(
                "Scatter brush extension %s aborted load: %s", ext_id, self._startup_error
            )
            self._teardown()
        except Exception as exc:  # noqa: BLE001 - fail soft, never crash the host
            self._startup_error = f"{type(exc).__name__}: {exc}"
            _LOGGER.exception(
                "Scatter brush extension %s aborted startup: %s", ext_id, exc
            )
            # Roll back anything partially constructed so a failed load leaves no
            # dangling subscription or window behind.
            self._teardown()

    def on_shutdown(self) -> None:
        """Detach input, destroy the panel/window, and release references."""
        self._teardown()
        self._started = False

    # ------------------------------------------------------------------
    # Enable / disable + stage-event recovery (task 10.2)
    # ------------------------------------------------------------------
    def _try_enable(self) -> bool:
        """Wire the brush against the current stage/viewport if both exist.

        Returns ``True`` when the brush became active. When no stage or viewport
        is available yet, enters the disabled state and returns ``False`` (the
        tool stays loaded and recovers on the next stage-open event). A missing
        or read-only mod layer (``ScatterLayerError``) also leaves the tool
        disabled with a clear message (Requirement 10.5), recovering when a
        writable mod layer is available.

        Raises:
            _DependencyError: if a declared dependency cannot be imported while
                resolving the viewport or wiring components (Requirement 1.4).
        """
        stage = _get_stage()
        if stage is None:
            self._enter_disabled_state(NO_STAGE_MESSAGE)
            return False

        try:
            viewport = _get_active_viewport()
        except ImportError as exc:
            raise _DependencyError(f"viewport extension unavailable: {exc}") from exc

        if viewport is None:
            self._enter_disabled_state(NO_STAGE_MESSAGE)
            return False

        try:
            self._wire_components(stage, viewport)
        except ImportError as exc:
            raise _DependencyError(str(exc)) from exc
        except ScatterLayerError as exc:
            # No / read-only mod layer: stay loaded but disabled with the error,
            # recovering when a writable mod layer becomes available (Req 10.5).
            _LOGGER.warning("Scatter brush: %s", exc)
            self._enter_disabled_state(str(exc))
            return False

        self._enter_enabled_state()
        return True

    def _enter_enabled_state(self) -> None:
        """Mark the tool enabled and clear the panel's disabled message."""
        self._enabled = True
        if self._panel is not None:
            set_enabled = getattr(self._panel, "set_enabled", None)
            if callable(set_enabled):
                set_enabled(True)
            set_status = getattr(self._panel, "set_status_message", None)
            if callable(set_status):
                set_status("")

    def _enter_disabled_state(self, message: str) -> None:
        """Disable brush controls and show ``message`` in the panel.

        Detaches input and releases the wired component graph so no event can
        drive a brush-less tool, then greys out the panel and surfaces the
        message. Safe to call repeatedly and before any wiring (idempotent).
        """
        self._enabled = False

        # Detach input first so no pointer event fires at a half-released graph.
        if self._input is not None:
            try:
                self._input.detach()
            except Exception:  # noqa: BLE001 - disabling must not raise
                _LOGGER.exception("Scatter brush: error detaching input handler")
            self._input = None

        # Release the wired component graph; the panel persists (it shows the
        # disabled state) but its palette is detached.
        self._brush = None
        self._raycaster = None
        self._sampler = None
        self._writer = None
        self._palette = None
        self._viewport = None
        self._stage = None

        if self._panel is not None:
            set_palette = getattr(self._panel, "set_palette", None)
            if callable(set_palette):
                set_palette(None)
            set_enabled = getattr(self._panel, "set_enabled", None)
            if callable(set_enabled):
                set_enabled(False)
            set_status = getattr(self._panel, "set_status_message", None)
            if callable(set_status):
                set_status(message)

    def _subscribe_stage_events(self) -> None:
        """Subscribe to omni.usd stage open/close events (no-op without Kit).

        When a stage opens the tool re-enables and re-wires automatically; when
        a stage closes it returns to the disabled state. Guarded so it is a
        harmless no-op when the Kit runtime / stage-event stream is unavailable
        (the tool then simply does not auto-recover in that environment).
        """
        if not HAS_OMNI_USD:
            return
        try:  # pragma: no cover - requires a live omni.usd context
            context = omni.usd.get_context()  # type: ignore[union-attr]
            event_stream = (
                context.get_stage_event_stream() if context is not None else None
            )
            if event_stream is None:
                return
            self._stage_event_sub = event_stream.create_subscription_to_pop(
                self._on_stage_event, name="lightspeed.trex.scatter stage events"
            )
        except Exception:  # noqa: BLE001 - subscription is best-effort
            _LOGGER.debug(
                "Scatter brush: could not subscribe to stage events", exc_info=True
            )
            self._stage_event_sub = None

    def _on_stage_event(self, event: object) -> None:
        """React to a stage event: re-enable on OPENED, disable on CLOSED.

        The event type is read and matched defensively so the handler works with
        both the real ``omni.usd.StageEventType`` enum and simple test fakes
        that expose a ``type`` string. A dependency error during re-enable is
        caught here (an event callback must never crash the host) and recorded.
        """
        event_type = getattr(event, "type", event)
        if self._stage_event_matches(event_type, "OPENED"):
            try:
                self._try_enable()
            except _DependencyError as exc:
                self._startup_error = f"unresolved dependency: {exc}"
                _LOGGER.error("Scatter brush: %s", self._startup_error)
                self._enter_disabled_state(f"Scatter brush unavailable: {exc}")
        elif self._stage_event_matches(event_type, "CLOSED"):
            self._enter_disabled_state(NO_STAGE_MESSAGE)

    @staticmethod
    def _stage_event_matches(event_type: object, name: str) -> bool:
        """Whether ``event_type`` denotes the stage event called ``name``.

        Compares against ``omni.usd.StageEventType.<name>`` (by enum and int
        value) when Kit is present, and falls back to a case-insensitive string
        match so unit-test fakes can pass ``type="OPENED"`` / ``"CLOSED"``.
        """
        if event_type is None:
            return False
        if HAS_OMNI_USD:
            try:  # pragma: no cover - requires the omni.usd enum
                expected = getattr(omni.usd.StageEventType, name)  # type: ignore[union-attr]
                if event_type == expected or event_type == int(expected):
                    return True
            except Exception:  # noqa: BLE001 - fall through to the string match
                pass
        return str(event_type).upper().endswith(name)

    def _forward_settings(self, settings: object) -> None:
        """Forward a panel settings snapshot to the live brush (if any).

        Registered once on the persistent panel so it survives stage open/close;
        snapshots emitted while the tool is disabled are harmlessly dropped.
        """
        if self._brush is not None:
            self._brush.set_settings(settings)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Wiring (design "Example Usage")
    # ------------------------------------------------------------------
    def _wire_components(self, stage: object, viewport: object) -> None:
        """Build the component graph from the given ``stage`` and ``viewport``.

        Mirrors the design's ``on_startup`` example: stage -> palette/writer,
        viewport -> raycaster, raycaster+sampler+palette+writer -> brush, the
        palette bound to the persistent panel, and the input handler attached to
        the viewport + brush. The brush's warning/error handlers are routed to
        the panel so empty-palette warnings and read-only-layer errors surface
        in the UI (Requirements 10.5 / 14.4).

        Raises:
            ScatterLayerError: if no writable mod layer is available.
            ImportError: if a declared dependency cannot be imported.
        """
        self._stage = stage
        self._viewport = viewport

        mod_layer = _resolve_mod_layer(stage)

        # Palette is validated against the live stage.
        self._palette = AssetPalette(stage)
        # Writer authors into the dedicated scatter sublayer of the mod layer;
        # constructing it validates that a mod layer is available (Req 10.5).
        self._writer = _make_writer(stage, mod_layer)
        # Seeded RNG instance; per-stroke seeding is driven by BrushSettings.
        self._sampler = PlacementSampler(random.Random())
        self._raycaster = SceneRaycaster(viewport)

        self._brush = ScatterBrush(
            raycaster=self._raycaster,
            sampler=self._sampler,
            palette=self._palette,
            writer=self._writer,
        )
        # Route the brush's user-facing warnings/errors to the panel surface.
        self._brush.set_warning_handler(self._on_brush_warning)
        self._brush.set_error_handler(self._on_brush_error)

        # Bind the palette to the persistent panel and push current settings.
        if self._panel is not None:
            set_palette = getattr(self._panel, "set_palette", None)
            if callable(set_palette):
                set_palette(self._palette)
            self._brush.set_settings(self._panel.get_settings())

        # Input handler translates viewport pointer events into stroke calls.
        self._input = ViewportInputHandler()
        self._input.attach(viewport, self._brush)

    def _on_brush_warning(self, message: str) -> None:
        """Forward a brush warning to the panel/log (panel surfaces it in UI)."""
        _LOGGER.warning("Scatter brush: %s", message)
        if self._panel is not None:
            set_status = getattr(self._panel, "set_status_message", None)
            if callable(set_status):
                set_status(message)

    def _on_brush_error(self, message: str) -> None:
        """Forward a brush error (e.g. read-only mod layer) to the panel/log."""
        _LOGGER.error("Scatter brush: %s", message)
        if self._panel is not None:
            set_status = getattr(self._panel, "set_status_message", None)
            if callable(set_status):
                set_status(message)

    # ------------------------------------------------------------------
    # Entry-point registration (Requirement 1.7)
    # ------------------------------------------------------------------
    def _register_entry_point(self) -> None:
        """Register the tool entry point, falling back to a self-provided window.

        Tries a host menu/tool registration point first. When none is available,
        sets up a self-provided ``omni.ui`` window as the entry point and reports
        that the host registration point was unavailable (Requirement 1.7).
        """
        self._host_registration_available = self._try_register_host_entry_point()
        if not self._host_registration_available:
            _LOGGER.info(
                "Scatter brush: host tool registration point unavailable; "
                "falling back to a self-provided window entry point (Requirement 1.7)."
            )

    def _try_register_host_entry_point(self) -> bool:
        """Attempt to register a host menu item that opens the Brush_UI.

        Returns ``True`` if a host registration point accepted the tool entry,
        ``False`` if none is available (or registration failed) so the caller
        can fall back to the self-provided window.
        """
        try:  # pragma: no cover - requires a live Kit menu service
            import omni.kit.menu.utils as menu_utils  # type: ignore
            from omni.kit.menu.utils import MenuItemDescription  # type: ignore
        except ImportError:
            return False

        try:  # pragma: no cover - requires a live Kit menu service
            self._menu_items = [
                MenuItemDescription(
                    name=WINDOW_TITLE,
                    onclick_fn=lambda *_a: self._activate(),
                )
            ]
            menu_utils.add_menu_items(self._menu_items, MENU_PATH)
            return True
        except Exception:  # noqa: BLE001 - any failure -> fall back to a window
            self._menu_items = None
            return False

    # ------------------------------------------------------------------
    # Brush_UI activation (Requirement 1.6)
    # ------------------------------------------------------------------
    def _activate(self) -> None:
        """Open the Brush_UI: create the window (if needed) and show it.

        Building the panel into an ``omni.ui`` window is the concrete "open the
        Brush_UI" action for both the host-registered and the self-provided
        fallback entry points. A no-op (other than recording intent) when
        ``omni.ui`` is unavailable, so the wiring stays exercisable in tests.
        """
        if not HAS_OMNI_UI:
            return
        if self._panel is None:
            return

        if self._window is None:
            self._window = ui.Window(WINDOW_TITLE, width=360, height=560)
            with self._window.frame:
                self._panel.build_ui()

        # Showing the window is what "opens the Brush_UI" on activation.
        self._window.visible = True

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------
    def _teardown(self) -> None:
        """Detach input, remove menu items, destroy the window, drop references."""
        # Release the stage-event subscription first so no event fires mid-teardown.
        if self._stage_event_sub is not None:
            try:
                unsubscribe = getattr(self._stage_event_sub, "unsubscribe", None)
                if callable(unsubscribe):
                    unsubscribe()
            except Exception:  # noqa: BLE001 - teardown must not raise
                _LOGGER.debug(
                    "Scatter brush: error releasing stage-event subscription",
                    exc_info=True,
                )
            self._stage_event_sub = None

        # Detach the input handler first so no event fires mid-teardown.
        if self._input is not None:
            try:
                self._input.detach()
            except Exception:  # noqa: BLE001 - teardown must not raise
                _LOGGER.exception("Scatter brush: error detaching input handler")

        # Remove any host menu items we registered.
        if self._menu_items is not None:
            try:  # pragma: no cover - requires a live Kit menu service
                import omni.kit.menu.utils as menu_utils  # type: ignore

                menu_utils.remove_menu_items(self._menu_items, MENU_PATH)
            except Exception:  # noqa: BLE001 - teardown must not raise
                _LOGGER.debug("Scatter brush: could not remove menu items", exc_info=True)

        # Destroy the self-provided window, if any.
        if self._window is not None:
            destroy = getattr(self._window, "destroy", None)
            try:
                if callable(destroy):
                    destroy()
            except Exception:  # noqa: BLE001 - teardown must not raise
                _LOGGER.exception("Scatter brush: error destroying window")

        # Release references.
        self._input = None
        self._menu_items = None
        self._window = None
        self._panel = None
        self._brush = None
        self._raycaster = None
        self._sampler = None
        self._writer = None
        self._palette = None
        self._viewport = None
        self._stage = None
        self._host_registration_available = False
        self._enabled = False
