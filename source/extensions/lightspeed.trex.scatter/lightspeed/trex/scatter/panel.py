"""The brush control panel (``omni.ui``) and its UI-free settings core.

``BrushPanel`` exposes the scatter brush controls -- radius, density, scale
range, yaw jitter, position jitter, surface-align toggle, normal blend, RNG
seed, and instancing mode -- plus the browsable/evolving asset palette
(discover / add / remove / per-asset weight) and the active brush-asset
selection.

Design reference: "Component 1: BrushPanel". Public interface::

    build_ui() -> None
    get_settings() -> BrushSettings
    on_settings_changed(callback: Callable[[BrushSettings], None]) -> None

The ``omni.ui`` import is guarded (mirroring the ``HAS_PXR`` pattern in
``models.py``) so this module stays importable -- and therefore unit-testable --
without the Kit runtime. All validation and snapshot logic lives in plain
methods that do not touch ``omni.ui``; ``build_ui`` only wires widgets to those
methods. This lets task 8.2 test the snapshot/rejection behaviour with no live
UI.

Validation follows Requirement 4:
    - radius in [0.01, 10000.0]            (4.1, 4.2)
    - density in [0.01, 1000.0]            (4.3)
    - scale min/max each in [0.001, 1000.0], min <= max   (4.4, 4.5)
    - yaw jitter in [0.0, 360.0]           (4.6)
    - instancing mode in {"None", "Point Instancing"}, default
      "Point Instancing"                   (4.11)

The panel holds the *last valid* ``BrushSettings``. When a control change would
produce an out-of-range value, the panel rejects it, retains the last valid
value, records a user-facing error message, and does **not** emit a new
snapshot (Requirements 4.2, 4.5). The active asset selection behaves the same
way: an invalid file is rejected and the previously selected asset is retained
(Requirements 2.2, 2.3).
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

from .models import (
    DEFAULT_INSTANCING_MODE,
    INSTANCING_NONE,
    POINT_INSTANCING,
    BrushSettings,
)

try:  # pragma: no cover - import guard exercised only by environment
    import omni.ui as ui  # type: ignore

    HAS_OMNI_UI = True
except ImportError:  # pragma: no cover - exercised only without the Kit runtime
    ui = None  # type: ignore
    HAS_OMNI_UI = False

# ``AssetPalette`` lives in ``palette.py`` (implemented by task 2.1). Guard the
# import so the panel -- and its settings logic -- stays importable and testable
# even before/without the palette module. The panel only needs the documented
# palette surface (discover_used_assets / add_asset / remove_asset /
# list_assets), so a duck-typed instance is sufficient.
try:  # pragma: no cover - depends on sibling task completion
    from .palette import AssetPalette  # type: ignore

    HAS_PALETTE = True
except ImportError:  # pragma: no cover - palette module not present yet
    AssetPalette = None  # type: ignore
    HAS_PALETTE = False

__all__ = [
    "HAS_OMNI_UI",
    "HAS_PALETTE",
    "VALID_ASSET_EXTENSIONS",
    "RADIUS_MIN",
    "RADIUS_MAX",
    "DENSITY_MIN",
    "DENSITY_MAX",
    "SCALE_MIN",
    "SCALE_MAX",
    "YAW_MIN",
    "YAW_MAX",
    "BrushPanel",
]

# --- Requirement 4 control ranges --------------------------------------------
RADIUS_MIN, RADIUS_MAX = 0.01, 10000.0  # Requirement 4.1 / 4.2
DENSITY_MIN, DENSITY_MAX = 0.01, 1000.0  # Requirement 4.3
SCALE_MIN, SCALE_MAX = 0.001, 1000.0  # Requirement 4.4
YAW_MIN, YAW_MAX = 0.0, 360.0  # Requirement 4.6
JITTER_MIN, JITTER_MAX = 0.0, 1.0  # position_jitter (design)
NORMAL_BLEND_MIN, NORMAL_BLEND_MAX = 0.0, 1.0  # normal_blend (design)

# Accepted brush-asset file extensions (Requirement 2.1).
VALID_ASSET_EXTENSIONS = (".usd", ".usda", ".usdc", ".usdz")

# Default control values. ``yaw_jitter`` maps to ``BrushSettings.yaw_range`` as
# ``(0.0, yaw_jitter)`` so each sample's rotation is drawn uniformly from
# ``[0, jitter]`` (Requirement 4.6).
_DEFAULT_VALUES: Dict[str, object] = {
    "radius": 1.0,
    "density": 1.0,
    "spacing": 0.0,
    "scale_min": 1.0,
    "scale_max": 1.0,
    "yaw_jitter": 360.0,
    "position_jitter": 1.0,
    "align_to_normal": True,
    "normal_blend": 1.0,
    "seed": 0,
    "max_instances_per_stroke": 10000,
    "instancing_mode": DEFAULT_INSTANCING_MODE,
}


class BrushPanel:
    """Brush settings + asset-palette panel.

    The panel keeps a dictionary of raw control values and a cached *last valid*
    :class:`BrushSettings`. Every control mutation goes through :meth:`_apply`,
    which validates the full candidate value set against the Requirement 4
    ranges before committing it. On rejection the prior values and snapshot are
    retained and an error message is recorded.

    Args:
        palette: An ``AssetPalette``-like object (duck-typed to the documented
            ``discover_used_assets`` / ``add_asset`` / ``remove_asset`` /
            ``list_assets`` surface). Optional so the panel can be constructed
            and its logic tested without a live palette.
        asset_validator: Optional callable that decides whether a selected asset
            file can be opened as a valid USD stage (Requirement 2.3). When
            omitted, only the file-extension check is applied.
    """

    def __init__(
        self,
        palette: Optional[object] = None,
        asset_validator: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self._palette = palette
        self._asset_validator = asset_validator

        self._values: Dict[str, object] = dict(_DEFAULT_VALUES)
        # The defaults are valid by construction, so seed the cached snapshot.
        self._last_valid_settings: BrushSettings = self._values_to_settings(self._values)

        self._error: Optional[str] = None
        self._asset_error: Optional[str] = None
        self._selected_asset_path: Optional[str] = None

        # Enable/disable state (design "Scenario: No active stage or viewport").
        # The panel starts enabled; the extension disables it (with a status
        # message) when no stage/viewport is available and re-enables it when a
        # stage opens (Requirement 1.4 recovery path).
        self._enabled: bool = True
        self._status_message: str = ""

        self._callbacks: List[Callable[[BrushSettings], None]] = []

        # ``omni.ui`` widget handles, populated by ``build_ui``.
        self._widgets: Dict[str, object] = {}
        self._error_label: Optional[object] = None
        self._asset_label: Optional[object] = None
        self._status_label: Optional[object] = None
        self._palette_container: Optional[object] = None

    # ------------------------------------------------------------------
    # Public interface (design "Component 1: BrushPanel")
    # ------------------------------------------------------------------
    def get_settings(self) -> BrushSettings:
        """Return the last valid brush settings snapshot."""
        return self._last_valid_settings

    def on_settings_changed(self, callback: Callable[[BrushSettings], None]) -> None:
        """Register a callback invoked with a fresh snapshot on every change."""
        if not callable(callback):
            raise TypeError("callback must be callable")
        self._callbacks.append(callback)

    # ------------------------------------------------------------------
    # Enable / disable + status message (design "No active stage or viewport")
    # ------------------------------------------------------------------
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the brush controls.

        While disabled (no stage/viewport available) the controls are greyed out
        so the user cannot paint into a missing stage; the extension re-enables
        them automatically when a stage opens. UI-free and safe to call without
        the Kit runtime -- the state is tracked and applied to widgets only when
        a UI has been built.
        """
        self._enabled = bool(enabled)
        self._sync_enabled_state()

    @property
    def enabled(self) -> bool:
        """Whether the brush controls are currently enabled."""
        return self._enabled

    def set_status_message(self, message: Optional[str]) -> None:
        """Set the panel's status/notification message (``None`` clears it).

        Used to surface "Open a stage to use the scatter brush" while disabled,
        and to surface brush warnings/errors (e.g. a read-only mod layer).
        """
        self._status_message = message or ""
        self._sync_status_label()

    @property
    def status_message(self) -> str:
        """The current status/notification message (empty string when clear)."""
        return self._status_message

    def set_palette(self, palette: Optional[object]) -> None:
        """Attach (or detach) the asset palette and refresh the palette view.

        Lets the extension keep a single persistent panel across stage
        open/close: the palette is (re)bound when the brush is wired and cleared
        (``None``) when the tool is disabled.
        """
        self._palette = palette
        self._rebuild_palette_view()

    # ------------------------------------------------------------------
    # Settings core (UI-free, unit-testable)
    # ------------------------------------------------------------------
    @property
    def error(self) -> Optional[str]:
        """The most recent control-rejection message, or ``None`` if the last
        change was accepted."""
        return self._error

    @property
    def asset_error(self) -> Optional[str]:
        """The most recent asset-selection rejection message, or ``None``."""
        return self._asset_error

    @property
    def selected_asset_path(self) -> Optional[str]:
        """The currently active brush-asset file path (Requirement 2.2)."""
        return self._selected_asset_path

    def get_value(self, field: str) -> object:
        """Return the current (last valid) value of a control field."""
        return self._values[field]

    # -- individual control setters (called by UI widgets and by tests) --
    def set_radius(self, value: float) -> bool:
        return self._apply(radius=float(value))

    def set_density(self, value: float) -> bool:
        return self._apply(density=float(value))

    def set_scale_min(self, value: float) -> bool:
        return self._apply(scale_min=float(value))

    def set_scale_max(self, value: float) -> bool:
        return self._apply(scale_max=float(value))

    def set_scale_range(self, scale_min: float, scale_max: float) -> bool:
        return self._apply(scale_min=float(scale_min), scale_max=float(scale_max))

    def set_yaw_jitter(self, value: float) -> bool:
        return self._apply(yaw_jitter=float(value))

    def set_position_jitter(self, value: float) -> bool:
        return self._apply(position_jitter=float(value))

    def set_align_to_normal(self, value: bool) -> bool:
        return self._apply(align_to_normal=bool(value))

    def set_normal_blend(self, value: float) -> bool:
        return self._apply(normal_blend=float(value))

    def set_seed(self, value: int) -> bool:
        return self._apply(seed=int(value))

    def set_instancing_mode(self, value: str) -> bool:
        return self._apply(instancing_mode=value)

    def _apply(self, **changes: object) -> bool:
        """Validate a candidate value set, commit + emit on success.

        Returns ``True`` if the change was accepted (snapshot emitted) or
        ``False`` if it was rejected (last valid values retained, error set).
        """
        candidate = dict(self._values)
        candidate.update(changes)

        errors = self._validate_values(candidate)
        if errors:
            # Reject: retain last valid values and snapshot (Req 4.2 / 4.5).
            self._error = "; ".join(errors)
            self._sync_widgets_to_values()
            return False

        try:
            settings = self._values_to_settings(candidate)
        except ValueError as exc:
            # Defensive: BrushSettings caught something the range checks did not.
            self._error = str(exc)
            self._sync_widgets_to_values()
            return False

        self._values = candidate
        self._last_valid_settings = settings
        self._error = None
        self._emit(settings)
        return True

    @staticmethod
    def _validate_values(values: Dict[str, object]) -> List[str]:
        """Return a list of human-readable range errors (empty if all valid)."""
        errors: List[str] = []

        radius = values["radius"]
        if not (RADIUS_MIN <= radius <= RADIUS_MAX):  # type: ignore[operator]
            errors.append(
                f"radius must be in [{RADIUS_MIN}, {RADIUS_MAX}], got {radius}"
            )

        density = values["density"]
        if not (DENSITY_MIN <= density <= DENSITY_MAX):  # type: ignore[operator]
            errors.append(
                f"density must be in [{DENSITY_MIN}, {DENSITY_MAX}], got {density}"
            )

        scale_min = values["scale_min"]
        scale_max = values["scale_max"]
        if not (SCALE_MIN <= scale_min <= SCALE_MAX):  # type: ignore[operator]
            errors.append(
                f"min scale must be in [{SCALE_MIN}, {SCALE_MAX}], got {scale_min}"
            )
        if not (SCALE_MIN <= scale_max <= SCALE_MAX):  # type: ignore[operator]
            errors.append(
                f"max scale must be in [{SCALE_MIN}, {SCALE_MAX}], got {scale_max}"
            )
        if scale_min > scale_max:  # type: ignore[operator]
            errors.append(
                f"min scale ({scale_min}) must not exceed max scale ({scale_max})"
            )

        yaw_jitter = values["yaw_jitter"]
        if not (YAW_MIN <= yaw_jitter <= YAW_MAX):  # type: ignore[operator]
            errors.append(
                f"yaw jitter must be in [{YAW_MIN}, {YAW_MAX}], got {yaw_jitter}"
            )

        position_jitter = values["position_jitter"]
        if not (JITTER_MIN <= position_jitter <= JITTER_MAX):  # type: ignore[operator]
            errors.append(
                f"position jitter must be in [{JITTER_MIN}, {JITTER_MAX}], "
                f"got {position_jitter}"
            )

        normal_blend = values["normal_blend"]
        if not (NORMAL_BLEND_MIN <= normal_blend <= NORMAL_BLEND_MAX):  # type: ignore[operator]
            errors.append(
                f"normal blend must be in [{NORMAL_BLEND_MIN}, {NORMAL_BLEND_MAX}], "
                f"got {normal_blend}"
            )

        instancing_mode = values["instancing_mode"]
        if instancing_mode not in (POINT_INSTANCING, INSTANCING_NONE):
            errors.append(
                f"instancing mode must be one of "
                f"{(INSTANCING_NONE, POINT_INSTANCING)}, got {instancing_mode!r}"
            )

        return errors

    @staticmethod
    def _values_to_settings(values: Dict[str, object]) -> BrushSettings:
        """Map raw control values to an immutable ``BrushSettings`` snapshot."""
        return BrushSettings(
            radius=float(values["radius"]),  # type: ignore[arg-type]
            density=float(values["density"]),  # type: ignore[arg-type]
            spacing=float(values["spacing"]),  # type: ignore[arg-type]
            position_jitter=float(values["position_jitter"]),  # type: ignore[arg-type]
            # Requirement 4.6: rotation drawn uniformly from [0, jitter].
            yaw_range=(0.0, float(values["yaw_jitter"])),  # type: ignore[arg-type]
            scale_range=(
                float(values["scale_min"]),  # type: ignore[arg-type]
                float(values["scale_max"]),  # type: ignore[arg-type]
            ),
            align_to_normal=bool(values["align_to_normal"]),
            normal_blend=float(values["normal_blend"]),  # type: ignore[arg-type]
            seed=int(values["seed"]),  # type: ignore[arg-type]
            max_instances_per_stroke=int(values["max_instances_per_stroke"]),  # type: ignore[arg-type]
            instancing_mode=str(values["instancing_mode"]),
        )

    def _emit(self, settings: BrushSettings) -> None:
        for callback in self._callbacks:
            callback(settings)

    # ------------------------------------------------------------------
    # Active brush-asset selection (Requirements 2.1, 2.2, 2.3)
    # ------------------------------------------------------------------
    def select_asset(self, path: str) -> bool:
        """Set the active brush-asset file, validating the selection.

        Rejects files whose extension is not one of
        :data:`VALID_ASSET_EXTENSIONS` or that the optional ``asset_validator``
        deems unopenable, retaining the previously selected asset and recording
        an error (Requirement 2.3). Returns ``True`` on acceptance.
        """
        if not isinstance(path, str) or not path:
            self._asset_error = "no asset file selected"
            return False

        extension = os.path.splitext(path)[1].lower()
        if extension not in VALID_ASSET_EXTENSIONS:
            self._asset_error = (
                f"'{path}' is not a valid USD asset "
                f"(expected one of {VALID_ASSET_EXTENSIONS})"
            )
            return False

        if self._asset_validator is not None and not self._asset_validator(path):
            self._asset_error = f"'{path}' could not be opened as a valid USD stage"
            return False

        self._selected_asset_path = path
        self._asset_error = None
        self._sync_asset_label()
        return True

    # ------------------------------------------------------------------
    # Asset palette delegation (browsable / evolving library)
    # ------------------------------------------------------------------
    def discover_assets(self) -> List[object]:
        """Discover referenceable source assets already used in the mod."""
        if self._palette is None:
            return []
        return list(self._palette.discover_used_assets())

    def add_palette_asset(self, prim_path: str, weight: float = 1.0) -> None:
        """Register a new asset in the palette and refresh the palette view."""
        if self._palette is None:
            raise RuntimeError("no asset palette is attached to the panel")
        self._palette.add_asset(prim_path, weight)
        self._rebuild_palette_view()

    def remove_palette_asset(self, prim_path: str) -> None:
        """Remove an asset from the palette and refresh the palette view."""
        if self._palette is None:
            raise RuntimeError("no asset palette is attached to the panel")
        self._palette.remove_asset(prim_path)
        self._rebuild_palette_view()

    def set_palette_asset_weight(self, prim_path: str, weight: float) -> None:
        """Update a palette entry's per-asset selection weight."""
        if self._palette is None:
            raise RuntimeError("no asset palette is attached to the panel")
        # The palette validates the path; re-adding updates the stored weight.
        self._palette.add_asset(prim_path, weight)
        self._rebuild_palette_view()

    def list_palette_assets(self) -> List[object]:
        """Return the current palette entries."""
        if self._palette is None:
            return []
        return list(self._palette.list_assets())

    # ------------------------------------------------------------------
    # omni.ui construction (requires the Kit runtime)
    # ------------------------------------------------------------------
    def build_ui(self) -> None:  # pragma: no cover - requires a live Kit runtime
        """Build the ``omni.ui`` controls and wire them to the settings core.

        Raises:
            RuntimeError: if ``omni.ui`` is unavailable (no Kit runtime).
        """
        if not HAS_OMNI_UI:
            raise RuntimeError(
                "omni.ui is not available; build_ui requires the Kit runtime. "
                "The panel's settings logic can still be used without a UI."
            )

        with ui.VStack(spacing=6):
            ui.Label("Scatter Brush", height=0)
            # Status/notification line (e.g. "Open a stage to use the scatter
            # brush" while disabled, or a read-only-layer error).
            self._status_label = ui.Label(self._status_message, height=0)

            self._build_float_row("Radius", "radius", RADIUS_MIN, RADIUS_MAX, self.set_radius)
            self._build_float_row("Density", "density", DENSITY_MIN, DENSITY_MAX, self.set_density)

            with ui.HStack(height=0, spacing=4):
                ui.Label("Scale", width=80)
                scale_min = ui.FloatField(width=70)
                scale_min.model.set_value(self._values["scale_min"])
                scale_min.model.add_value_changed_fn(
                    lambda m: self.set_scale_min(m.get_value_as_float())
                )
                scale_max = ui.FloatField(width=70)
                scale_max.model.set_value(self._values["scale_max"])
                scale_max.model.add_value_changed_fn(
                    lambda m: self.set_scale_max(m.get_value_as_float())
                )
                self._widgets["scale_min"] = scale_min
                self._widgets["scale_max"] = scale_max

            self._build_float_row("Yaw Jitter", "yaw_jitter", YAW_MIN, YAW_MAX, self.set_yaw_jitter)
            self._build_float_row(
                "Position Jitter", "position_jitter", JITTER_MIN, JITTER_MAX, self.set_position_jitter
            )

            with ui.HStack(height=0, spacing=4):
                ui.Label("Align to Surface", width=120)
                align = ui.CheckBox()
                align.model.set_value(bool(self._values["align_to_normal"]))
                align.model.add_value_changed_fn(
                    lambda m: self.set_align_to_normal(m.get_value_as_bool())
                )
                self._widgets["align_to_normal"] = align

            self._build_float_row(
                "Normal Blend", "normal_blend", NORMAL_BLEND_MIN, NORMAL_BLEND_MAX, self.set_normal_blend
            )

            with ui.HStack(height=0, spacing=4):
                ui.Label("Seed", width=80)
                seed = ui.IntField(width=70)
                seed.model.set_value(int(self._values["seed"]))
                seed.model.add_value_changed_fn(
                    lambda m: self.set_seed(m.get_value_as_int())
                )
                self._widgets["seed"] = seed

            with ui.HStack(height=0, spacing=4):
                ui.Label("Instancing", width=80)
                # ComboBox option order matches the index mapping below.
                combo = ui.ComboBox(
                    1 if self._values["instancing_mode"] == POINT_INSTANCING else 0,
                    INSTANCING_NONE,
                    POINT_INSTANCING,
                )
                combo.model.add_item_changed_fn(self._on_instancing_changed)
                self._widgets["instancing_mode"] = combo

            ui.Separator(height=4)
            ui.Label("Brush Asset", height=0)
            self._asset_label = ui.Label(
                self._selected_asset_path or "(none selected)", height=0
            )

            ui.Label("Asset Palette", height=0)
            self._palette_container = ui.VStack(spacing=2)
            self._rebuild_palette_view()

            self._error_label = ui.Label("", height=0)

        # Apply the current enabled/disabled state to the freshly built widgets.
        self._sync_status_label()
        self._sync_enabled_state()

    def _build_float_row(
        self,
        label: str,
        field: str,
        lo: float,
        hi: float,
        setter: Callable[[float], bool],
    ) -> None:  # pragma: no cover - requires a live Kit runtime
        with ui.HStack(height=0, spacing=4):
            ui.Label(label, width=120)
            widget = ui.FloatField(width=70)
            widget.model.set_value(float(self._values[field]))
            widget.model.add_value_changed_fn(
                lambda m: setter(m.get_value_as_float())
            )
            self._widgets[field] = widget

    def _on_instancing_changed(self, model, *_args) -> None:  # pragma: no cover
        index = model.get_item_value_model().get_value_as_int()
        mode = POINT_INSTANCING if index == 1 else INSTANCING_NONE
        self.set_instancing_mode(mode)

    def _rebuild_palette_view(self) -> None:  # pragma: no cover - requires UI
        if not HAS_OMNI_UI or self._palette_container is None:
            return
        self._palette_container.clear()
        with self._palette_container:
            for ref in self.list_palette_assets():
                prim_path = getattr(ref, "prim_path", str(ref))
                weight = getattr(ref, "weight", 1.0)
                with ui.HStack(height=0, spacing=4):
                    ui.Label(str(prim_path))
                    ui.Label(f"w={weight}", width=60)
                    ui.Button(
                        "Remove",
                        width=70,
                        clicked_fn=lambda p=prim_path: self.remove_palette_asset(p),
                    )

    def _sync_asset_label(self) -> None:  # pragma: no cover - requires UI
        if HAS_OMNI_UI and self._asset_label is not None:
            self._asset_label.text = self._selected_asset_path or "(none selected)"

    def _sync_status_label(self) -> None:  # pragma: no cover - requires UI
        """Push the current status message into the status label, if built."""
        if HAS_OMNI_UI and self._status_label is not None:
            self._status_label.text = self._status_message

    def _sync_enabled_state(self) -> None:  # pragma: no cover - requires UI
        """Apply the enabled/disabled state to every built control widget.

        Greys out (``enabled = False``) all input widgets while the tool is
        disabled. No-op without a UI so the state can be toggled in tests.
        """
        if not HAS_OMNI_UI:
            return
        for widget in self._widgets.values():
            if widget is not None and hasattr(widget, "enabled"):
                try:
                    widget.enabled = self._enabled
                except Exception:  # noqa: BLE001 - never fail on a UI quirk
                    pass
        if self._palette_container is not None and hasattr(self._palette_container, "enabled"):
            try:
                self._palette_container.enabled = self._enabled
            except Exception:  # noqa: BLE001
                pass

    def _sync_widgets_to_values(self) -> None:  # pragma: no cover - requires UI
        """Push the last valid values back into widgets after a rejection.

        A rejected control change leaves the widget showing the bad value; this
        restores it to the retained valid value so the UI reflects model state.
        """
        if not HAS_OMNI_UI:
            return
        if self._error_label is not None:
            self._error_label.text = self._error or ""
        for field in ("radius", "density", "scale_min", "scale_max", "yaw_jitter",
                      "position_jitter", "normal_blend"):
            widget = self._widgets.get(field)
            if widget is not None:
                widget.model.set_value(float(self._values[field]))
        seed_widget = self._widgets.get("seed")
        if seed_widget is not None:
            seed_widget.model.set_value(int(self._values["seed"]))
