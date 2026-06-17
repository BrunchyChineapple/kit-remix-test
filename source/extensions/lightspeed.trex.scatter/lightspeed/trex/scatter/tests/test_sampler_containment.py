"""Property-based test for in-brush containment (task 4.2).

This module validates the design's **Property 2: Placements stay within the
brush**:

    Every placement produced from a hit lies within ``radius`` of the hit point
    in the surface tangent plane.
    ``∀ p ∈ sample(hit, s). dist_tangent(p.translate, hit.point) <= s.radius``

The test feeds ``PlacementSampler.sample`` random (but valid) ``BrushSettings``
and a random ``SurfaceHit`` (random world point, random unit normal). For every
returned placement it measures the distance from ``hit.point`` to
``placement.translate`` *in the surface tangent plane* -- i.e. it removes the
component along ``hit.normal`` before measuring -- and asserts that distance is
within ``radius`` (plus a tiny float tolerance).

The palette is built with ``stage=None`` and at least one asset so that
``AssetPalette.pick`` always returns a valid asset (path validation is skipped
without a stage), isolating the placement geometry from stage resolution.

Validates: Requirements 3.2, 3.3
"""

from __future__ import annotations

import random

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lightspeed.trex.scatter.models import HAS_PXR, BrushSettings, SurfaceHit
from lightspeed.trex.scatter.palette import AssetPalette
from lightspeed.trex.scatter.sampler import PlacementSampler

try:  # pragma: no cover - import guard exercised only by environment
    from pxr import Gf  # type: ignore
except ImportError:  # pragma: no cover - exercised only without USD installed
    Gf = None  # type: ignore

# The placement math requires USD; skip the whole module when pxr is absent
# rather than reporting spurious failures.
pytestmark = pytest.mark.skipif(
    not HAS_PXR or Gf is None,
    reason="PlacementSampler containment test requires the USD 'pxr' libraries",
)

# Absolute slack on the in-plane distance check. The sampler's disc radius is
# ``radius * sqrt(u) * position_jitter <= radius`` and the tangent-plane
# projection is a contraction, so the true distance never exceeds ``radius``;
# this tolerance only absorbs floating-point accumulation in the vector math.
_DISTANCE_EPS = 1e-6


@st.composite
def _valid_brush_settings(draw: st.DrawFn) -> BrushSettings:
    """Generate a valid ``BrushSettings`` with small expected instance counts.

    ``radius`` and ``density`` are kept modest so that
    ``count = max(1, round(density * pi * radius**2))`` stays small (a few dozen
    at most), keeping each example fast while still exercising the disc math
    across the full radius range.
    """
    radius = draw(st.floats(min_value=0.1, max_value=3.0, allow_nan=False, allow_infinity=False))
    density = draw(st.floats(min_value=1e-3, max_value=0.5, allow_nan=False, allow_infinity=False))
    spacing = draw(st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False))
    position_jitter = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    normal_blend = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    align_to_normal = draw(st.booleans())

    yaw_lo = draw(st.floats(min_value=-360.0, max_value=360.0, allow_nan=False, allow_infinity=False))
    yaw_hi = draw(st.floats(min_value=yaw_lo, max_value=720.0, allow_nan=False, allow_infinity=False))

    scale_lo = draw(st.floats(min_value=1e-3, max_value=5.0, allow_nan=False, allow_infinity=False))
    scale_hi = draw(st.floats(min_value=scale_lo, max_value=10.0, allow_nan=False, allow_infinity=False))

    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))

    return BrushSettings(
        radius=radius,
        density=density,
        spacing=spacing,
        position_jitter=position_jitter,
        yaw_range=(yaw_lo, yaw_hi),
        scale_range=(scale_lo, scale_hi),
        align_to_normal=align_to_normal,
        normal_blend=normal_blend,
        seed=seed,
    )


_finite_coord = st.floats(
    min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False
)


_component = st.floats(
    min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
)


@st.composite
def _surface_hit(draw: st.DrawFn) -> SurfaceHit:
    """Generate a ``SurfaceHit`` at a random point with a random unit normal."""
    point = Gf.Vec3d(draw(_finite_coord), draw(_finite_coord), draw(_finite_coord))

    # Draw a direction and normalise it via Gf so the normal is unit length
    # within the model's tolerance. A (near-)zero draw is replaced by world-up
    # so the result is always a valid unit normal -- no rejection loop needed.
    raw = Gf.Vec3d(draw(_component), draw(_component), draw(_component))
    if raw.GetLength() <= 1e-6:
        raw = Gf.Vec3d(0.0, 0.0, 1.0)
    normal = raw.GetNormalized()

    return SurfaceHit(point=point, normal=normal, prim_path="/World/surface")


def _build_palette() -> AssetPalette:
    """A non-empty, stage-less palette so ``pick`` always returns an asset."""
    palette = AssetPalette(stage=None)
    palette.add_asset("/World/asset_a", weight=1.0)
    palette.add_asset("/World/asset_b", weight=2.0)
    return palette


def _tangent_plane_distance(
    translate: "Gf.Vec3d", point: "Gf.Vec3d", normal: "Gf.Vec3d"
) -> float:
    """Distance from ``point`` to ``translate`` measured in ``normal``'s plane.

    Removes the component of the offset along ``normal`` (its out-of-plane part)
    and returns the length of what remains, i.e. the in-tangent-plane distance.
    """
    offset = translate - point
    along = offset * normal  # Gf.Vec3d.__mul__ with a vector is the dot product
    in_plane = offset - normal * along
    return in_plane.GetLength()


@given(settings_=_valid_brush_settings(), hit=_surface_hit())
@settings(
    max_examples=75,
    deadline=None,  # placement counts vary; wall-clock is not a useful bound
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_placements_stay_within_brush_radius(
    settings_: BrushSettings, hit: SurfaceHit
) -> None:
    """Every placement lies within ``radius`` of the hit in the tangent plane.

    Validates: Requirements 3.2, 3.3
    """
    sampler = PlacementSampler(random.Random(settings_.seed))

    placements = sampler.sample(hit, settings_, _build_palette())

    for placement in placements:
        distance = _tangent_plane_distance(
            placement.translate, hit.point, hit.normal
        )
        assert distance <= settings_.radius + _DISTANCE_EPS, (
            f"placement at tangent-plane distance {distance} exceeds brush "
            f"radius {settings_.radius} (tolerance {_DISTANCE_EPS})"
        )
