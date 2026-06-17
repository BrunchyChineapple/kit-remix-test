"""Property-based tests for weighted asset selection (task 2.2).

These tests validate the selection behaviour of ``AssetPalette.pick`` against
the design's **Property 4 (Determinism)** -- specifically its weighted-selection
component:

    - Over many seeded draws, each asset is selected with a frequency that
      matches its configured weight (weight_i / sum(weights)) within a
      statistical tolerance.
    - Two pick sequences drawn from two ``random.Random`` instances created with
      the *same* seed are identical (determinism / reproducibility).

The palette is constructed with ``stage=None`` so path validation is skipped and
every added entry is pickable (see ``AssetPalette`` docstring), isolating the
weighted-selection logic from stage resolution.

Validates: Requirements 2.5
"""

from __future__ import annotations

import random
from collections import Counter
from typing import List, Tuple

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lightspeed.trex.scatter.palette import AssetPalette

# A fixed, large draw count keeps the binomial sampling error tiny: at p = 0.5
# (worst case) the standard deviation of the empirical frequency is
# sqrt(0.25 / 20000) ~= 0.0035, so the 0.05 tolerance below is well over ten
# sigma and the frequency assertion is not flaky.
_DRAWS = 20000

# Absolute tolerance on |empirical_freq - expected_freq| for every asset.
_FREQ_TOLERANCE = 0.05


# Weights are constrained to a moderate, strictly-positive range so no single
# asset's probability collapses to a near-zero value that a finite sample could
# never represent reliably. 2..6 assets covers the multi-asset selection space
# (Requirement 2.5) without making each example needlessly expensive.
_weights_strategy = st.lists(
    st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    min_size=2,
    max_size=6,
)


def _build_palette(weights: List[float]) -> AssetPalette:
    """Build a palette (no stage) with one asset per weight at a unique path."""
    palette = AssetPalette(stage=None)
    for index, weight in enumerate(weights):
        palette.add_asset(f"/World/asset_{index}", weight=weight)
    return palette


@given(weights=_weights_strategy)
@settings(
    max_examples=25,
    deadline=None,  # large deterministic draw loop; the wall-clock varies
    suppress_health_check=[HealthCheck.too_slow],
)
def test_selection_frequency_matches_weights(weights: List[float]) -> None:
    """Empirical selection frequency tracks weight_i / sum(weights).

    Validates: Requirements 2.5
    """
    palette = _build_palette(weights)
    total_weight = sum(weights)

    rng = random.Random(0x5CA77E5)  # deterministic so the test never flakes
    counts: Counter = Counter()
    for _ in range(_DRAWS):
        asset = palette.pick(rng)
        assert asset is not None  # non-empty palette always yields a pick
        counts[asset.prim_path] += 1

    for index, weight in enumerate(weights):
        prim_path = f"/World/asset_{index}"
        empirical = counts[prim_path] / _DRAWS
        expected = weight / total_weight
        assert abs(empirical - expected) <= _FREQ_TOLERANCE, (
            f"asset {prim_path}: empirical freq {empirical:.4f} deviates from "
            f"expected {expected:.4f} by more than {_FREQ_TOLERANCE}"
        )


@given(
    weights=_weights_strategy,
    seed=st.integers(min_value=0, max_value=2**63 - 1),
    sequence_length=st.integers(min_value=1, max_value=200),
)
@settings(max_examples=50, deadline=None)
def test_identical_seed_reproduces_pick_sequence(
    weights: List[float], seed: int, sequence_length: int
) -> None:
    """Two RNGs seeded identically produce the same pick sequence.

    Validates: Requirements 2.5
    """
    palette = _build_palette(weights)

    def draw_sequence(rng_seed: int) -> List[Tuple[str, float]]:
        rng = random.Random(rng_seed)
        sequence: List[Tuple[str, float]] = []
        for _ in range(sequence_length):
            asset = palette.pick(rng)
            assert asset is not None
            sequence.append((asset.prim_path, asset.weight))
        return sequence

    first = draw_sequence(seed)
    second = draw_sequence(seed)
    assert first == second
