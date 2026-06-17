"""The brush asset palette.

``AssetPalette`` holds the set of source asset references the brush can scatter
and selects among them. It is backed by an *evolving* library rather than a
fixed set: a modder can browse assets already referenced in the open stage
(``discover_used_assets``), register brand-new art-team assets as they land
(``add_asset``), drop entries (``remove_asset``), and the brush draws one asset
per placement via weighted-random selection (``pick``).

Validation (design "Component 6: AssetPalette", Requirements 2.1/2.3/9.1-9.3):
every added or discovered prim path must resolve to an existing, *referenceable*
prim on the stage. Unresolved paths are rejected by ``add_asset``; entries that
later become unresolved are silently skipped by ``pick`` so a stale palette
entry can never poison a stroke.

The enabled-asset set is capped at ``MAX_ENABLED_ASSETS`` (64, Requirement 2.4);
the minimum of one enabled asset is a precondition for *painting* and is
enforced by the stroke orchestrator at ``begin_stroke`` time, not here, because
an empty palette is a legitimate intermediate editing state.

``pxr`` is imported behind a guard (mirroring ``models.py``) so this module is
importable -- and unit-testable -- without USD installed. The palette never
requires the real ``pxr`` types directly: it talks to the stage through a small
duck-typed surface (``GetPrimAtPath``, ``Traverse``) so tests can inject a fake
stage, and the extension passes the live ``Usd.Stage``.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from .models import AssetRef

try:  # pragma: no cover - import guard exercised only by environment
    from pxr import Usd  # type: ignore  # noqa: F401

    HAS_PXR = True
except ImportError:  # pragma: no cover - exercised only without USD installed
    Usd = None  # type: ignore
    HAS_PXR = False

__all__ = [
    "HAS_PXR",
    "MIN_ENABLED_ASSETS",
    "MAX_ENABLED_ASSETS",
    "AssetPalette",
]

# Bounds on the enabled-asset set (Requirement 2.4). The maximum is enforced by
# ``add_asset``; the minimum is a stroke precondition enforced upstream.
MIN_ENABLED_ASSETS = 1
MAX_ENABLED_ASSETS = 64


def _prim_path_str(prim: object) -> str:
    """Return a prim's path as a plain string.

    Accepts a ``pxr.Usd.Prim`` (via ``GetPath``) or any object exposing the same
    method; falls back to ``str(prim)`` for minimal fakes.
    """
    get_path = getattr(prim, "GetPath", None)
    if callable(get_path):
        return str(get_path())
    return str(prim)


def _prim_is_referenceable(prim: object) -> bool:
    """Return whether ``prim`` resolves to an existing, referenceable prim.

    A prim is referenceable when it exists on the stage and is valid. ``None``
    (no prim at the path) and invalid prims are rejected. A minimal fake that
    omits ``IsValid`` is treated as valid as long as it is not ``None``.
    """
    if prim is None:
        return False
    is_valid = getattr(prim, "IsValid", None)
    if callable(is_valid):
        return bool(is_valid())
    return True


def _prim_has_references(prim: object) -> bool:
    """Return whether ``prim`` already references/places a source asset.

    Used by discovery to surface "the assets already used in the mod". Prefers
    ``HasAuthoredReferences`` when the prim exposes it; a minimal fake that omits
    it is included as long as it is referenceable.
    """
    has_refs = getattr(prim, "HasAuthoredReferences", None)
    if callable(has_refs):
        return bool(has_refs())
    return True


class AssetPalette:
    """An evolving, weighted set of source asset references for the brush.

    Parameters
    ----------
    stage:
        The open USD stage (``Usd.Stage``) used to validate that asset prim
        paths resolve to existing referenceable prims, and to discover assets
        already used in the mod. May be ``None`` (the extension can set it later
        via :meth:`set_stage`); while no stage is set, path validation is
        skipped because it cannot be performed, and discovery yields nothing.
        Tests inject a duck-typed fake stage exposing ``GetPrimAtPath`` and
        ``Traverse``.
    """

    def __init__(self, stage: object = None) -> None:
        self._stage = stage
        # prim_path -> AssetRef, insertion-ordered (dict preserves order).
        self._assets: Dict[str, AssetRef] = {}

    # -- stage wiring ----------------------------------------------------

    def set_stage(self, stage: object) -> None:
        """Set or replace the stage used for validation and discovery."""
        self._stage = stage

    # -- discovery -------------------------------------------------------

    def discover_used_assets(self) -> List[AssetRef]:
        """Enumerate referenceable source prims already present in the stage.

        Returns one :class:`AssetRef` per prim that exists, is referenceable,
        and already carries an authored reference (i.e. an asset the mod already
        uses). This is a read-only browse helper -- it does not add anything to
        the enabled palette; the caller (panel) decides what to register. When
        no stage is set, returns an empty list.
        """
        if self._stage is None:
            return []
        traverse = getattr(self._stage, "Traverse", None)
        if not callable(traverse):
            return []

        discovered: List[AssetRef] = []
        seen: set = set()
        for prim in traverse():
            if not _prim_is_referenceable(prim):
                continue
            if not _prim_has_references(prim):
                continue
            prim_path = _prim_path_str(prim)
            if prim_path in seen:
                continue
            seen.add(prim_path)
            discovered.append(
                AssetRef(
                    prim_path=prim_path,
                    layer_id=self._resolve_layer_id(prim),
                    weight=1.0,
                )
            )
        return discovered

    # -- mutation --------------------------------------------------------

    def add_asset(self, prim_path: str, weight: float = 1.0) -> None:
        """Register an asset prim in the palette.

        The path must resolve to an existing referenceable prim on the stage;
        unresolved paths are rejected (Requirements 2.3, 9.1). Re-adding an
        existing path updates its selection weight rather than duplicating it.
        Adding a brand-new path is refused once the enabled set already holds
        :data:`MAX_ENABLED_ASSETS` entries (Requirement 2.4).

        Raises
        ------
        ValueError
            If the path does not resolve to a referenceable prim, or adding a
            new entry would exceed the maximum enabled-asset count. ``weight``
            validity (``> 0``) is enforced by :class:`AssetRef`.
        """
        if not isinstance(prim_path, str) or not prim_path:
            raise ValueError(f"prim_path must be a non-empty string, got {prim_path!r}")

        prim = self._get_prim(prim_path)
        if self._stage is not None and not _prim_is_referenceable(prim):
            raise ValueError(
                f"prim_path {prim_path!r} does not resolve to an existing "
                f"referenceable prim on the stage"
            )

        is_new = prim_path not in self._assets
        if is_new and len(self._assets) >= MAX_ENABLED_ASSETS:
            raise ValueError(
                f"cannot enable more than {MAX_ENABLED_ASSETS} assets "
                f"(Requirement 2.4)"
            )

        # AssetRef enforces weight > 0 and path/layer_id validity.
        self._assets[prim_path] = AssetRef(
            prim_path=prim_path,
            layer_id=self._resolve_layer_id(prim),
            weight=weight,
        )

    def remove_asset(self, prim_path: str) -> None:
        """Remove an asset from the palette. No-op if it is not present."""
        self._assets.pop(prim_path, None)

    # -- queries ---------------------------------------------------------

    def list_assets(self) -> List[AssetRef]:
        """Return the enabled assets in insertion order (a fresh list)."""
        return list(self._assets.values())

    def is_empty(self) -> bool:
        """Return ``True`` when no assets are enabled."""
        return not self._assets

    def __len__(self) -> int:
        return len(self._assets)

    # -- selection -------------------------------------------------------

    def pick(self, rng: random.Random) -> Optional[AssetRef]:
        """Select one asset with probability proportional to its weight.

        Entries whose prim path no longer resolves to a referenceable prim are
        skipped (Requirement 2.3), so a stale entry cannot be returned. Returns
        ``None`` when there is no valid asset to pick (empty palette, or every
        entry currently invalid -- Property 8). Selection is read-only: the
        palette contents are never modified. Determinism follows from ``rng``:
        an identically seeded ``random.Random`` reproduces the pick sequence.
        """
        candidates: List[AssetRef] = []
        weights: List[float] = []
        for asset in self._assets.values():
            if self._stage is not None and not _prim_is_referenceable(
                self._get_prim(asset.prim_path)
            ):
                continue
            candidates.append(asset)
            weights.append(asset.weight)

        if not candidates:
            return None

        total = sum(weights)
        threshold = rng.random() * total
        cumulative = 0.0
        for asset, weight in zip(candidates, weights):
            cumulative += weight
            if threshold < cumulative:
                return asset
        # Floating-point guard: return the last candidate if rounding made the
        # threshold land at or beyond the cumulative total.
        return candidates[-1]

    # -- internals -------------------------------------------------------

    def _get_prim(self, prim_path: str) -> object:
        """Return the prim at ``prim_path`` on the stage, or ``None``."""
        if self._stage is None:
            return None
        get_prim = getattr(self._stage, "GetPrimAtPath", None)
        if not callable(get_prim):
            return None
        return get_prim(prim_path)

    def _resolve_layer_id(self, prim: object) -> str:
        """Best-effort identifier of the layer providing ``prim``.

        Prefers the strongest spec's layer from the prim stack; falls back to
        the stage root layer, then to an empty string. ``AssetRef`` accepts any
        string (including empty) for ``layer_id``.
        """
        get_stack = getattr(prim, "GetPrimStack", None)
        if callable(get_stack):
            try:
                stack = get_stack()
            except Exception:  # pragma: no cover - defensive against fakes
                stack = None
            if stack:
                layer = getattr(stack[0], "layer", None)
                identifier = getattr(layer, "identifier", None)
                if identifier:
                    return str(identifier)

        if self._stage is not None:
            get_root = getattr(self._stage, "GetRootLayer", None)
            if callable(get_root):
                try:
                    root = get_root()
                except Exception:  # pragma: no cover - defensive against fakes
                    root = None
                identifier = getattr(root, "identifier", None)
                if identifier:
                    return str(identifier)
        return ""
