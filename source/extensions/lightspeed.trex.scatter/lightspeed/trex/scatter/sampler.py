"""Density-driven placement sampling for the scatter brush.

``PlacementSampler`` turns a single surface hit plus the active brush settings
into a list of concrete :class:`~lightspeed.trex.scatter.models.Placement`
values -- the "where and how do instances land around this hit" service
consumed by ``ScatterBrush`` during a stroke (design "Component 5:
PlacementSampler").

Design contract (design "Placement sampling"):
    Preconditions
        - ``hit.normal`` is unit length; ``palette`` is non-empty.
        - ``BrushSettings`` validation rules hold.
    Postconditions
        - Returns between ``0`` and ``count`` placements (0 only if palette
          picks fail).
        - Every placement's ``translate`` lies within ``radius`` of
          ``hit.point`` in the surface tangent plane.
        - Every placement's ``scale`` is within ``scale_range``; ``orient`` is
          normalized.
    Loop invariants
        - All placements appended so far reference a valid palette asset.
        - The RNG is advanced deterministically, so a given ``seed`` reproduces
          the stroke.

``pxr`` is imported behind a guard (mirroring the ``HAS_PXR`` pattern in
``models.py``) so this module stays importable in environments without USD;
sampling itself uses ``pxr.Gf`` for the position/orientation math at runtime.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

from .models import HAS_PXR, BrushSettings, Placement, SurfaceHit
from .palette import AssetPalette

try:  # pragma: no cover - import guard exercised only by environment
    from pxr import Gf  # type: ignore
except ImportError:  # pragma: no cover - exercised only without USD installed
    Gf = None  # type: ignore

__all__ = [
    "PlacementSampler",
    "world_up",
    "blend_normal",
    "project_to_surface_plane",
    "orientation_from_up_and_yaw",
]

# Stage up axis used as the default orientation reference. The brush disc is
# sampled in the local XY plane (Z up), so the world up axis is +Z; this keeps
# the disc, its projection onto the tangent plane, and the orientation math in
# one consistent frame.
_WORLD_UP: Tuple[float, float, float] = (0.0, 0.0, 1.0)

# Below this length a blended/normal vector is considered degenerate and the
# world-up direction is used instead, so orientation is always well defined.
_DEGENERATE_EPS = 1e-9


def _require_pxr() -> None:
    if not HAS_PXR or Gf is None:
        raise RuntimeError(
            "PlacementSampler requires the USD 'pxr' libraries for placement math"
        )


def world_up() -> "Gf.Vec3d":
    """Return the stage world up axis (+Z) as a ``Gf.Vec3d``."""
    _require_pxr()
    return Gf.Vec3d(*_WORLD_UP)


def blend_normal(up: "Gf.Vec3d", normal: "Gf.Vec3d", blend: float) -> "Gf.Vec3d":
    """Blend ``up`` toward ``normal`` by ``blend`` and return a unit vector.

    ``blend`` runs ``0..1``: ``0`` keeps the world-up direction, ``1`` is fully
    surface-aligned. A degenerate blend (e.g. ``up`` and ``normal`` cancelling)
    falls back to the world-up direction so the result is always unit length.
    """
    _require_pxr()
    blended = up * (1.0 - blend) + normal * blend
    length = blended.GetLength()
    if length <= _DEGENERATE_EPS:
        return Gf.Vec3d(up).GetNormalized()
    return blended.GetNormalized()


def project_to_surface_plane(local: "Gf.Vec3d", normal: "Gf.Vec3d") -> "Gf.Vec3d":
    """Project a local disc offset onto the tangent plane of ``normal``.

    Removes the component of ``local`` along ``normal`` so the offset lies in
    the surface tangent plane. Because the projection is a contraction, the
    projected length never exceeds ``|local|`` -- which keeps placements within
    the brush radius (design Property 2).
    """
    _require_pxr()
    along = local * normal  # Gf.Vec3d.__mul__ with a vector is the dot product
    return local - normal * along


def orientation_from_up_and_yaw(up: "Gf.Vec3d", yaw: float) -> "Gf.Quatf":
    """Build a normalized orientation that aligns +Z to ``up`` then spins ``yaw``.

    ``yaw`` is in degrees and is applied as a rotation about the (already
    aligned) ``up`` axis. The returned ``Gf.Quatf`` is normalized within the
    model layer's tolerance so :class:`Placement` validation passes.
    """
    _require_pxr()
    align = Gf.Rotation(Gf.Vec3d(*_WORLD_UP), up)
    spin = Gf.Rotation(up, float(yaw))
    combined = align * spin
    quat = Gf.Quatf(combined.GetQuat())
    return quat.GetNormalized()


class PlacementSampler:
    """Scatter instances within the brush disc around a surface hit.

    Args:
        rng: A seeded ``random.Random`` instance. All randomness (disc position,
            yaw, uniform scale, and per-placement asset selection) is drawn from
            it, so an identically seeded RNG reproduces a stroke (design
            Property 4 / determinism).
    """

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def sample(
        self,
        hit: SurfaceHit,
        settings: BrushSettings,
        palette: AssetPalette,
    ) -> List[Placement]:
        """Produce placements for one surface hit (design "Placement sampling").

        Args:
            hit: The world-space surface hit to scatter around. ``hit.normal``
                must be unit length.
            settings: The active brush settings (validated ``BrushSettings``).
            palette: The asset palette; one asset is picked per placement.

        Returns:
            Between ``0`` and ``count`` placements, where
            ``count = max(1, round(density * pi * radius**2))``. Placements are
            skipped when the palette has no valid asset to pick.
        """
        _require_pxr()
        s = settings

        area = math.pi * s.radius * s.radius
        count = max(1, round(s.density * area))

        placements: List[Placement] = []
        for _ in range(count):
            # Uniform sample in the brush disc (sqrt for uniform area density).
            theta = self._rng.uniform(0.0, 2.0 * math.pi)
            r = s.radius * math.sqrt(self._rng.random()) * s.position_jitter
            local = Gf.Vec3d(r * math.cos(theta), r * math.sin(theta), 0.0)

            tangent_offset = project_to_surface_plane(local, hit.normal)
            translate = hit.point + tangent_offset

            up = (
                blend_normal(world_up(), hit.normal, s.normal_blend)
                if s.align_to_normal
                else world_up()
            )
            yaw = self._rng.uniform(s.yaw_range[0], s.yaw_range[1])
            orient = orientation_from_up_and_yaw(up, yaw)

            scale = self._rng.uniform(s.scale_range[0], s.scale_range[1])

            asset = palette.pick(self._rng)
            if asset is None:
                continue
            placements.append(
                Placement(asset=asset, translate=translate, orient=orient, scale=scale)
            )

        return placements
