"""Undoable, non-destructive USD authoring for the scatter brush.

``ScatterLayerWriter`` is the only component that mutates USD. It owns a
dedicated ``scatter.usda`` sublayer of the mod layer and authors one referencing
``Xform`` per :class:`~lightspeed.trex.scatter.models.Placement` into it. All
authoring is:

- **Non-destructive**: edits land only in ``scatter.usda`` (a sublayer of the mod
  layer), never in the capture/source layers (design Property 5).
- **Purely additive**: every call only *defines new* prims at unique paths; prims
  from previous strokes are never modified or removed (design Property 9).
- **Undoable**: ``open_undo_group``/``close_undo_group`` wrap a stroke as a single
  ``omni.kit.undo`` step (design Property 6).

The ``pxr`` and ``omni`` imports are guarded (mirroring ``models.HAS_PXR``) so the
module stays importable in environments without USD or Kit -- but the authoring
logic itself requires a real ``Usd.Stage`` to run.
"""

from __future__ import annotations

import logging
import os
import re
from typing import List, Optional

try:  # pragma: no cover - import guard exercised only by environment
    from pxr import Gf, Sdf, Tf, Usd, UsdGeom  # type: ignore

    HAS_PXR = True
except ImportError:  # pragma: no cover - exercised only without USD installed
    Gf = Sdf = Tf = Usd = UsdGeom = None  # type: ignore
    HAS_PXR = False

try:  # pragma: no cover - import guard exercised only by environment
    import omni.kit.undo as _kit_undo  # type: ignore

    HAS_KIT_UNDO = True
except ImportError:  # pragma: no cover - exercised only without Kit installed
    _kit_undo = None  # type: ignore
    HAS_KIT_UNDO = False

from .models import Placement

__all__ = ["ScatterLayerError", "ScatterLayerWriter", "SCATTER_LAYER_NAME", "SCATTER_ROOT_PATH"]

_LOGGER = logging.getLogger(__name__)

# File name of the dedicated sublayer that holds every scatter placement.
SCATTER_LAYER_NAME = "scatter.usda"

# All authored placements live beneath this scope so they are easy to find,
# select, and (via undo) revert without touching captured geometry.
SCATTER_ROOT_PATH = "/ScatterBrush"

# Characters that are not legal in a USD prim name are collapsed to "_".
_INVALID_NAME_CHARS = re.compile(r"[^A-Za-z0-9_]")


class ScatterLayerError(RuntimeError):
    """Raised when the scatter layer cannot be created or is not writable.

    The brush surfaces this to the user and aborts the stroke, leaving every
    layer unchanged (design "Scenario: Scatter layer cannot be created or is
    read-only").
    """


def _make_valid_prim_name(raw: str) -> str:
    """Turn an arbitrary string into a legal, non-empty USD prim name."""
    if HAS_PXR:
        # Tf.MakeValidIdentifier guarantees a valid, non-empty identifier.
        return Tf.MakeValidIdentifier(raw)
    cleaned = _INVALID_NAME_CHARS.sub("_", raw)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


class ScatterLayerWriter:
    """Authors placements into a dedicated ``scatter.usda`` sublayer.

    Args:
        stage: The open ``Usd.Stage`` to author into.
        mod_layer: The mod/replacement ``Sdf.Layer`` that owns the scatter
            sublayer. Must be in the stage's layer stack and writable.
    """

    def __init__(self, stage: "Usd.Stage", mod_layer: "Sdf.Layer") -> None:
        if not HAS_PXR:  # pragma: no cover - environment without USD
            raise ScatterLayerError(
                "pxr (USD) is not available; ScatterLayerWriter requires a real Usd.Stage"
            )
        if stage is None:
            raise ScatterLayerError("a Usd.Stage is required to author placements")
        if mod_layer is None:
            # Requirement 10.5: a mod layer is required as an edit target.
            raise ScatterLayerError("a mod layer is required to author placements")

        self._stage = stage
        self._mod_layer = mod_layer
        self._scatter_layer: Optional["Sdf.Layer"] = None
        # Monotonic counter that makes generated prim names unique within this
        # writer; uniqueness against the live stage is also verified per prim so
        # placements accumulate additively across strokes (Property 9).
        self._name_counter = 0

    # ------------------------------------------------------------------ layer

    def ensure_scatter_layer(self) -> "Sdf.Layer":
        """Return the scatter sublayer, creating and attaching it on first use.

        Creates ``scatter.usda`` next to the mod layer (or an anonymous layer when
        the mod layer is itself anonymous, e.g. an in-memory test stage) and
        inserts it as the strongest sublayer of the mod layer.

        Raises:
            ScatterLayerError: if the mod layer or the scatter layer is read-only,
                or the layer cannot be created.
        """
        if self._scatter_layer is not None:
            # Guard against the cached layer having become read-only meanwhile.
            if not self._scatter_layer.permissionToEdit:
                raise ScatterLayerError(
                    f"scatter layer {self._scatter_layer.identifier!r} is read-only"
                )
            return self._scatter_layer

        if not self._mod_layer.permissionToEdit:
            # We cannot attach the sublayer without editing the mod layer.
            raise ScatterLayerError(
                f"mod layer {self._mod_layer.identifier!r} is read-only; "
                "cannot attach the scatter sublayer"
            )

        layer = self._create_or_find_layer()
        if layer is None:
            raise ScatterLayerError("failed to create the scatter layer")
        if not layer.permissionToEdit:
            raise ScatterLayerError(
                f"scatter layer {layer.identifier!r} is read-only"
            )

        self._attach_sublayer(layer)
        self._scatter_layer = layer
        return layer

    def _create_or_find_layer(self) -> Optional["Sdf.Layer"]:
        """Create (or reopen) the scatter layer file beside the mod layer."""
        if self._mod_layer.anonymous:
            # In-memory stage (tests / unsaved project): keep the sublayer
            # anonymous so no file is written to disk.
            return Sdf.Layer.CreateAnonymous(SCATTER_LAYER_NAME)

        mod_path = self._mod_layer.realPath or self._mod_layer.identifier
        scatter_path = os.path.join(os.path.dirname(mod_path), SCATTER_LAYER_NAME)
        # Reuse the file if a previous session already created it; otherwise
        # create it fresh. FindOrOpen returns None when the file does not exist.
        return Sdf.Layer.FindOrOpen(scatter_path) or Sdf.Layer.CreateNew(scatter_path)

    def _attach_sublayer(self, layer: "Sdf.Layer") -> None:
        """Insert ``layer`` as the strongest sublayer of the mod layer (idempotent)."""
        # Anonymous layers are referenced by identifier; file layers by a path
        # relative to the mod layer when possible, so the project stays portable.
        if layer.anonymous:
            sublayer_ref = layer.identifier
        else:
            mod_path = self._mod_layer.realPath or self._mod_layer.identifier
            mod_dir = os.path.dirname(mod_path)
            try:
                sublayer_ref = os.path.relpath(layer.realPath, mod_dir).replace(os.sep, "/")
            except ValueError:  # different drives on Windows -> fall back to absolute
                sublayer_ref = layer.identifier

        existing = list(self._mod_layer.subLayerPaths)
        if sublayer_ref in existing or layer.identifier in existing:
            return
        # Strongest position so scatter opinions win over weaker sublayers.
        self._mod_layer.subLayerPaths.insert(0, sublayer_ref)

    # -------------------------------------------------------------- authoring

    def author_placements(self, placements: List["Placement"]) -> List[str]:
        """Define one referencing ``Xform`` per placement in ``scatter.usda``.

        Each authored prim gets a translate/orient/scale op stack and an internal
        reference to the placement's source prim. All edits are made inside an
        ``EditContext`` targeting the scatter layer, which restores the stage's
        prior edit target when the block exits (even on error).

        Args:
            placements: Placements to author. An empty list authors nothing.

        Returns:
            The list of newly created, stage-unique prim paths (one per placement).

        Raises:
            ScatterLayerError: if the scatter layer cannot be created or is
                read-only. The stage is left unchanged in that case.
        """
        if not placements:
            # Zero-placement strokes (miss / cap reached) author nothing.
            return []

        layer = self.ensure_scatter_layer()
        authored_paths: List[str] = []

        # Usd.EditContext directs every edit below to the scatter layer and
        # restores the previously active edit target on exit -- so we never
        # leave the stage targeting scatter.usda, and never touch the capture
        # layer (design Postconditions + Requirement 10.2).
        with Usd.EditContext(self._stage, Usd.EditTarget(layer)):
            for placement in placements:
                path = self._unique_path(placement.asset.prim_path)
                xform = UsdGeom.Xform.Define(self._stage, path)

                # Transform op stack: translate, then orient, then scale.
                xform.AddTranslateOp().Set(placement.translate)
                xform.AddOrientOp().Set(placement.orient)
                scale = float(placement.scale)
                xform.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))

                # Internal reference (empty assetPath) to the source prim so the
                # asset geometry is not copied into scatter.usda.
                xform.GetPrim().GetReferences().AddReference(
                    assetPath="", primPath=Sdf.Path(placement.asset.prim_path)
                )

                authored_paths.append(path)

        return authored_paths

    def _unique_path(self, asset_prim_path: str) -> str:
        """Return a fresh prim path under ``SCATTER_ROOT_PATH`` not used on the stage."""
        base_name = _make_valid_prim_name(asset_prim_path.rstrip("/").rsplit("/", 1)[-1] or "asset")
        while True:
            candidate = f"{SCATTER_ROOT_PATH}/{base_name}_{self._name_counter}"
            self._name_counter += 1
            # Additive guarantee: never reuse a path that already holds a prim
            # (from this or a previous stroke).
            if not self._stage.GetPrimAtPath(candidate).IsValid():
                return candidate

    # ------------------------------------------------------------------ undo

    def open_undo_group(self, label: str) -> None:
        """Begin an ``omni.kit.undo`` group so a stroke is one undo step.

        ``label`` is accepted for parity with the design interface and logging;
        ``omni.kit.undo`` groups are unnamed. When Kit's undo system is
        unavailable (e.g. unit tests without Kit), this is a no-op.
        """
        if HAS_KIT_UNDO:
            _kit_undo.begin_group()
        else:  # pragma: no cover - environment without Kit
            _LOGGER.debug("omni.kit.undo unavailable; skipping undo group %r", label)

    def close_undo_group(self) -> None:
        """End the current ``omni.kit.undo`` group opened by ``open_undo_group``."""
        if HAS_KIT_UNDO:
            _kit_undo.end_group()
        else:  # pragma: no cover - environment without Kit
            _LOGGER.debug("omni.kit.undo unavailable; skipping undo group close")
