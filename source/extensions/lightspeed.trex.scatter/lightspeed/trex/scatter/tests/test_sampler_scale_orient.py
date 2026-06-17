"""Property-based test for well-formed scale and orientation (task 4.3).

Validates the design's **Property 3 (Scale and orientation are well-formed)**:

    Every authored placement has ``scale ∈ [scale_range.min, scale_range.max]``
    and a normalized ``orient``.

The test drives ``PlacementSampler.sample`` with random-but-valid
``BrushSettings`` (a random ``scale_range`` with ``0 < min <= max`` and a random
ordered ``yaw_range``) and a random ``SurfaceHit`` whose normal is unit length,
then asserts that *every* returned placement's uniform ``scale`` lies inside the
configured range (with a tiny float tolerance) and that its ``orient`` quaternion
is normalized (length within ``NORMAL_TOLERANCE`` of 1.0, measured via
``Gf.Quatf.GetLength``).

A non-empty palette is built (``stage=None`` so every entry is pickable) so the
sampler actually produces placements, and the density/radius are constrained so
the per-example instance count stays small and the test runs quickly.

Validates: Requirements 4.4, 4.6
"""

from __future__ import annotations

import math
import random

from hypothesis import given, settings
from hypothesis import strategies as st

from pxr import Gf

from lightspeed.trex.scatter.models import (
    NORMAL_TOLERANCE,
    BrushSettings,
    SurfaceHit,
)
from lightspeed.trex.scatter.palette import AssetPalette
from lightspeed.trex.scatter.sampler import PlacementSampler

# Absolute float tolerance applied to the scale-range bounds. ``random.uniform``
# can return a value a few ULPs outside [a, b] from rounding, so a tiny slack
# keeps the inclusive-range assertion robust without weakening it materially.
_SCALE_EPS = 1e-9


@st.composite
def _scale_ranges(draw) -> tuple:
    """Draw a valid ``scale_range`` with ``0 < min <= max``."""
    lo = draw(st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False))
    hi = draw(st.floats(min_value=lo, max_value=lo + 10.0, allow_nan=False, allow_infinity=False))
    return (lo, hi)


@st.composite
def _yaw_ranges(draw) -> tuple:
    """Draw a valid, ordered ``yaw_range`` in degrees."""
    lo = draw(st.floats(min_value=-360.0, max_value=360.0, allow_nan=False, allow_infinity=False))
    hi = draw(st.floats(min_value=lo, max_value=lo + 360.0, allow_nan=False, allow_infinity=False))
    return (lo, hi)


@st.composite
def _unit_normals(draw) -> "Gf.Vec3d":
    """Draw a unit-length surface normal as a ``Gf.Vec3d``.

    A random direction is normalized; degenerate near-zero vectors are replaced
    with world-up so the result is always unit length (and passes ``SurfaceHit``
    validation).
    """
    comp = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    x = draw(comp)
    y = draw(comp)
    z = draw(comp)
    vec = Gf.Vec3d(x, y, z)
    if vec.GetLength() <= 1e-6:
        vec = Gf.Vec3d(0.0, 0.0, 1.0)
    return vec.GetNormalized()


@given(
    scale_range=_scale_ranges(),
    yaw_range=_yaw_ranges(),
    normal=_unit_normals(),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    align_to_normal=st.booleans(),
    normal_blend=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    # radius/density chosen so density * pi * radius**2 stays small (count <= ~32).
    radius=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
    density=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=75, deadline=None)
def test_scale_and_orientation_well_formed(
    scale_range: tuple,
    yaw_range: tuple,
    normal: "Gf.Vec3d",
    seed: int,
    align_to_normal: bool,
    normal_blend: float,
    radius: float,
    density: float,
) -> None:
    """Every placement has scale within scale_range and a normalized orient.

    Validates: Requirements 4.4, 4.6
    """
    settings_obj = BrushSettings(
        radius=radius,
        density=density,
        spacing=0.0,
        position_jitter=1.0,
        yaw_range=yaw_range,
        scale_range=scale_range,
        align_to_normal=align_to_normal,
        normal_blend=normal_blend,
        seed=seed,
    )

    hit = SurfaceHit(
        point=Gf.Vec3d(0.0, 0.0, 0.0),
        normal=normal,
        prim_path="/World/surface",
    )

    palette = AssetPalette(stage=None)
    palette.add_asset("/World/asset_a", weight=1.0)
    palette.add_asset("/World/asset_b", weight=2.0)

    sampler = PlacementSampler(random.Random(seed))
    placements = sampler.sample(hit, settings_obj, palette)

    # With a non-empty palette (every entry pickable) the sampler always yields
    # the full count, so we should never get an empty result here.
    assert placements, "expected a non-empty palette to produce placements"

    scale_min, scale_max = scale_range
    for placement in placements:
        assert scale_min - _SCALE_EPS <= placement.scale <= scale_max + _SCALE_EPS, (
            f"scale {placement.scale} outside "
            f"[{scale_min}, {scale_max}]"
        )

        quat_length = Gf.Quatf(placement.orient).GetLength()
        assert math.isclose(quat_length, 1.0, abs_tol=NORMAL_TOLERANCE), (
            f"orient quaternion not normalized: length {quat_length} "
            f"(tolerance {NORMAL_TOLERANCE})"
        )
