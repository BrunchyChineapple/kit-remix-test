"""Property-based test for placement-sampler determinism (task 4.4).

Validates the design's **Property 4**: identical seed + settings + hit +
palette produce identical placements. ``PlacementSampler`` draws *all* of its
randomness (disc position, yaw, uniform scale, per-placement asset pick) from an
injected ``random.Random``. Two samplers built from ``random.Random(seed)`` with
the *same* seed must therefore reproduce a stroke exactly: same number of
placements and, element-wise, the same ``translate``, ``orient``, ``scale`` and
selected asset ``prim_path``.

The palette is built identically (same insertion order) for both runs with
``stage=None`` so path validation is skipped and every entry is pickable,
isolating determinism of the sampler + weighted pick from stage resolution.

Validates: Requirements 4.4
"""

from __future__ import annotations

import math
import random
from typing import List

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pxr import Gf

from lightspeed.trex.scatter.models import BrushSettings, SurfaceHit
from lightspeed.trex.scatter.palette import AssetPalette
from lightspeed.trex.scatter.sampler import PlacementSampler

# Same seed + same code path => bit-identical floats. A tiny tolerance is kept
# only as a safety margin against any platform float quirk; it is far below the
# model's NORMAL_TOLERANCE so it cannot mask a real divergence.
_EQ_TOL = 1e-9

# Insertion-ordered palette shared by both runs. Built once per example via the
# builder below so the two samplers see an identical palette.
_ASSET_PATHS = [
    "/World/Assets/grass_a",
    "/World/Assets/grass_b",
    "/World/Assets/rock_a",
    "/World/Assets/flower_a",
]


def _build_palette() -> AssetPalette:
    """Build the fixed palette in a fixed insertion order (no stage)."""
    palette = AssetPalette(stage=None)
    # Distinct weights so the weighted pick exercises its full branching, while
    # staying strictly positive (AssetRef requires weight > 0).
    for index, prim_path in enumerate(_ASSET_PATHS):
        palette.add_asset(prim_path, weight=float(index + 1))
    return palette


# Brush settings strategy. ``radius`` and ``density`` are bounded so the derived
# instance count -- max(1, round(density * pi * radius**2)) -- stays small
# (worst case here: round(2.0 * pi * 4.0) ~= 25), keeping examples fast while
# still drawing many placements per stroke.
_settings_strategy = st.builds(
    BrushSettings,
    radius=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
    density=st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False),
    spacing=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    position_jitter=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    yaw_range=st.tuples(
        st.floats(min_value=-360.0, max_value=360.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-360.0, max_value=360.0, allow_nan=False, allow_infinity=False),
    ).map(lambda pair: (min(pair), max(pair))),
    scale_range=st.tuples(
        st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
    ).map(lambda pair: (min(pair), max(pair))),
    align_to_normal=st.booleans(),
    normal_blend=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)


def _unit_normal(x: float, y: float, z: float) -> Gf.Vec3d:
    """Return a unit-length normal from raw components (fallback to +Z)."""
    vec = Gf.Vec3d(x, y, z)
    if vec.GetLength() <= 1e-6:
        return Gf.Vec3d(0.0, 0.0, 1.0)
    return vec.GetNormalized()


@st.composite
def _surface_hits(draw) -> SurfaceHit:
    """Random world-space hit with a guaranteed unit normal."""
    point = Gf.Vec3d(
        draw(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        draw(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        draw(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
    )
    normal = _unit_normal(
        draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
    )
    return SurfaceHit(point=point, normal=normal, prim_path="/World/Ground")


def _assert_vec_equal(a: Gf.Vec3d, b: Gf.Vec3d) -> None:
    for index in range(3):
        assert math.isclose(a[index], b[index], rel_tol=0.0, abs_tol=_EQ_TOL), (
            f"translate component {index} diverged: {a[index]!r} vs {b[index]!r}"
        )


def _assert_quat_equal(a: Gf.Quatf, b: Gf.Quatf) -> None:
    assert math.isclose(a.GetReal(), b.GetReal(), rel_tol=0.0, abs_tol=_EQ_TOL), (
        f"orient real diverged: {a.GetReal()!r} vs {b.GetReal()!r}"
    )
    ai, bi = a.GetImaginary(), b.GetImaginary()
    for index in range(3):
        assert math.isclose(ai[index], bi[index], rel_tol=0.0, abs_tol=_EQ_TOL), (
            f"orient imaginary component {index} diverged: {ai[index]!r} vs {bi[index]!r}"
        )


def _sample_with_seed(seed: int, hit: SurfaceHit, settings_obj: BrushSettings) -> List:
    """Run a sampler over a freshly built palette with a freshly seeded RNG."""
    sampler = PlacementSampler(random.Random(seed))
    return sampler.sample(hit, settings_obj, _build_palette())


@given(
    seed=st.integers(min_value=0, max_value=2**63 - 1),
    hit=_surface_hits(),
    settings_obj=_settings_strategy,
)
@settings(
    max_examples=75,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_identical_seed_produces_identical_placements(
    seed: int, hit: SurfaceHit, settings_obj: BrushSettings
) -> None:
    """Two equally-seeded samplers reproduce the same placement list.

    Validates: Requirements 4.4
    """
    first = _sample_with_seed(seed, hit, settings_obj)
    second = _sample_with_seed(seed, hit, settings_obj)

    assert len(first) == len(second), (
        f"placement count diverged: {len(first)} vs {len(second)}"
    )

    for index, (left, right) in enumerate(zip(first, second)):
        assert left.asset.prim_path == right.asset.prim_path, (
            f"placement {index} picked different assets: "
            f"{left.asset.prim_path!r} vs {right.asset.prim_path!r}"
        )
        _assert_vec_equal(left.translate, right.translate)
        _assert_quat_equal(left.orient, right.orient)
        assert math.isclose(left.scale, right.scale, rel_tol=0.0, abs_tol=_EQ_TOL), (
            f"placement {index} scale diverged: {left.scale!r} vs {right.scale!r}"
        )
