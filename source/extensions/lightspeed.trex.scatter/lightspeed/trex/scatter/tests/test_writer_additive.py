"""Property-based test for purely additive scatter authoring (task 5.3).

This module validates the design's **Property 9: Scattering is purely additive
(strokes accumulate)**:

    ``∀ stroke_n. prims_before(stroke_n) ⊆ prims_after(stroke_n)`` and
    ``prims_after(stroke_n) = prims_before(stroke_n) ∪ new_prims(stroke_n)``,
    where ``new_prims(stroke_n) ∩ prims_before(stroke_n) = ∅``.

Concretely, for a random sequence of strokes (each a random-length batch of
placements, drawn from a palette that varies stroke to stroke), every call to
``ScatterLayerWriter.author_placements`` must:

    * return only *new* prim paths (disjoint from the prims that existed before
      the stroke);
    * leave the set of scatter prims equal to the prior set unioned with the new
      paths -- nothing from a previous stroke is removed;
    * leave every prior prim's authored data (its transform-op stack and its
      source reference) byte-for-byte identical -- nothing is modified.

The test mirrors ``test_writer.py``: it authors into a real, in-memory
``Usd.Stage`` whose anonymous root layer doubles as the mod layer, so the
authoring logic runs against a genuine USD runtime.

Validates: Requirements 14.1, 14.5
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import pytest

pytest.importorskip("pxr", reason="ScatterLayerWriter authoring requires USD (pxr)")

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: E402

from lightspeed.trex.scatter.models import AssetRef, Placement  # noqa: E402
from lightspeed.trex.scatter.writer import (  # noqa: E402
    SCATTER_ROOT_PATH,
    ScatterLayerWriter,
)

# A pool of referenceable source prims. Each stroke draws from a (varying)
# slice of this pool so successive strokes use different "palettes" and asset
# types, exercising the multi-asset accumulation clause of Property 9.
_NUM_SOURCE_ASSETS = 5
_SOURCE_PRIM_PATHS = [f"/World/Asset_{i}" for i in range(_NUM_SOURCE_ASSETS)]

# Identity orientation (unit quaternion) satisfies Placement's normalisation
# check; the transform values themselves are irrelevant to the additive
# property, only that prior prims keep whatever they were authored with.
_IDENTITY_ORIENT = Gf.Quatf(1.0, 0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #

# A single placement is described by which source asset it references, a uniform
# scale (> 0) and an integer translate triple (kept as ints so distinct
# placements get distinct, exactly-comparable transforms).
_placement_spec = st.tuples(
    st.integers(min_value=0, max_value=_NUM_SOURCE_ASSETS - 1),  # asset index
    st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    st.tuples(
        st.integers(min_value=-50, max_value=50),
        st.integers(min_value=-50, max_value=50),
        st.integers(min_value=-50, max_value=50),
    ),
)

# A stroke is a (possibly empty) batch of placements; an empty stroke models a
# miss / capped stroke that authors nothing.
_stroke = st.lists(_placement_spec, min_size=0, max_size=8)

# A sequence of strokes painted one after another into the same writer.
_stroke_sequence = st.lists(_stroke, min_size=1, max_size=6)


def _build_placements(stroke_spec: List[Tuple[int, float, Tuple[int, int, int]]]) -> List[Placement]:
    """Turn a generated stroke spec into validated ``Placement`` objects."""
    placements: List[Placement] = []
    for asset_index, scale, (tx, ty, tz) in stroke_spec:
        prim_path = _SOURCE_PRIM_PATHS[asset_index]
        placements.append(
            Placement(
                asset=AssetRef(prim_path=prim_path, layer_id="anon"),
                translate=Gf.Vec3d(float(tx), float(ty), float(tz)),
                orient=_IDENTITY_ORIENT,
                scale=float(scale),
            )
        )
    return placements


# --------------------------------------------------------------------------- #
# Snapshot helpers
# --------------------------------------------------------------------------- #


def _scatter_prim_paths(stage: "Usd.Stage") -> Set[str]:
    """Return the set of authored scatter prim paths (leaves under the root)."""
    root = stage.GetPrimAtPath(SCATTER_ROOT_PATH)
    if not root.IsValid():
        return set()
    # Every placement is a direct Xform child of the scatter root scope.
    return {str(child.GetPath()) for child in root.GetChildren()}


def _reference_target_paths(prim_spec) -> List[str]:
    """Collect the source primPaths from a prim spec's reference list."""
    ref_list = prim_spec.referenceList
    items = (
        list(ref_list.prependedItems)
        + list(ref_list.appendedItems)
        + list(ref_list.explicitItems)
    )
    return sorted(str(ref.primPath) for ref in items)


def _snapshot_specs(
    scatter_layer: "Sdf.Layer", paths: Set[str]
) -> Dict[str, Tuple[Tuple[Tuple[str, object], ...], Tuple[str, ...]]]:
    """Capture the authored data (attrs + references) for ``paths``.

    For each path we record:
        * every authored attribute's name and default value (this includes the
          ``xformOp:*`` op values and ``xformOpOrder``), and
        * the sorted list of reference target prim paths.

    Two snapshots compare equal iff none of the prior prims were modified.
    """
    snapshot: Dict[str, Tuple[Tuple[Tuple[str, object], ...], Tuple[str, ...]]] = {}
    for path in paths:
        spec = scatter_layer.GetPrimAtPath(Sdf.Path(path))
        assert spec is not None, f"expected a prim spec at {path} in the scatter layer"
        attrs = tuple(
            sorted((attr.name, attr.default) for attr in spec.attributes)
        )
        refs = tuple(_reference_target_paths(spec))
        snapshot[path] = (attrs, refs)
    return snapshot


def _new_stage_and_writer() -> Tuple["Usd.Stage", ScatterLayerWriter]:
    """Build a fresh in-memory stage (root layer = mod layer) and a writer."""
    stage = Usd.Stage.CreateInMemory()
    for prim_path in _SOURCE_PRIM_PATHS:
        UsdGeom.Xform.Define(stage, prim_path)
    writer = ScatterLayerWriter(stage, stage.GetRootLayer())
    return stage, writer


# --------------------------------------------------------------------------- #
# Property 9
# --------------------------------------------------------------------------- #


@given(strokes=_stroke_sequence)
@settings(
    max_examples=75,
    deadline=None,  # USD authoring wall-clock varies; bound by max_examples
    suppress_health_check=[HealthCheck.too_slow],
)
def test_authoring_is_purely_additive(
    strokes: List[List[Tuple[int, float, Tuple[int, int, int]]]],
) -> None:
    """Every stroke only defines new prims; prior strokes are never touched.

    For each stroke in a random sequence (each drawing from a varying palette):

        * ``new_prims`` is disjoint from ``prims_before``;
        * ``prims_after == prims_before ∪ new_prims``;
        * every prim from ``prims_before`` still exists afterwards with identical
          authored transform-op values and reference targets.

    Validates: Requirements 14.1, 14.5
    """
    stage, writer = _new_stage_and_writer()
    # Ensure the scatter layer exists up front so we can snapshot specs even
    # before the first authoring call (and so the layer identity is stable).
    scatter_layer = writer.ensure_scatter_layer()

    for stroke_spec in strokes:
        placements = _build_placements(stroke_spec)

        prims_before = _scatter_prim_paths(stage)
        before_snapshot = _snapshot_specs(scatter_layer, prims_before)

        new_prims = writer.author_placements(placements)

        prims_after = _scatter_prim_paths(stage)
        new_set = set(new_prims)

        # One new prim path per placement, all returned paths are unique.
        assert len(new_prims) == len(placements)
        assert len(new_set) == len(new_prims)

        # new_prims ∩ prims_before = ∅  (nothing reuses a prior path).
        assert new_set.isdisjoint(prims_before), (
            f"new prims {new_set & prims_before} collide with prior stroke prims"
        )

        # prims_after = prims_before ∪ new_prims  (additive, nothing removed).
        assert prims_after == prims_before | new_set, (
            "scatter prim set after the stroke is not the prior set unioned with "
            f"the new prims (removed: {prims_before - prims_after}, "
            f"unexpected: {prims_after - prims_before - new_set})"
        )

        # prims_before ⊆ prims_after  (explicit subset clause of Property 9).
        assert prims_before <= prims_after

        # Every prior prim is unchanged: same authored attrs + references.
        after_snapshot = _snapshot_specs(scatter_layer, prims_before)
        assert after_snapshot == before_snapshot, (
            "a prim authored by a previous stroke was modified by this stroke"
        )

        # Every prior prim still resolves to a valid prim on the composed stage.
        for path in prims_before:
            assert stage.GetPrimAtPath(path).IsValid(), (
                f"prior stroke prim {path} disappeared after a later stroke"
            )


@given(strokes=_stroke_sequence)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_accumulated_prims_equal_total_authored(
    strokes: List[List[Tuple[int, float, Tuple[int, int, int]]]],
) -> None:
    """The final scatter set is exactly the union of every stroke's new prims.

    A direct corollary of Property 9 across a whole sequence: because each stroke
    only adds disjoint new prims and never removes any, the prims present at the
    end equal the disjoint union of all per-stroke additions, and their count
    equals the total number of placements authored.

    Validates: Requirements 14.1, 14.5
    """
    stage, writer = _new_stage_and_writer()

    all_new: List[str] = []
    total_placements = 0
    for stroke_spec in strokes:
        placements = _build_placements(stroke_spec)
        total_placements += len(placements)
        all_new.extend(writer.author_placements(placements))

    # No path was ever reused across the whole sequence.
    assert len(all_new) == len(set(all_new))
    assert len(all_new) == total_placements

    # The live scatter set equals the union of every stroke's additions.
    assert _scatter_prim_paths(stage) == set(all_new)
