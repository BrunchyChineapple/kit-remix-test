"""Unit tests for AssetPalette edge cases (task 2.3).

These example-based tests exercise the boundary behaviour of ``palette.py``:

    * an empty palette is empty and has no valid ``pick`` (Property 8 at the
      palette level -- the "no valid asset" precondition the stroke orchestrator
      relies on);
    * ``add_asset`` rejects a path that does not resolve to a referenceable prim
      and leaves the palette unchanged;
    * an entry whose prim later becomes invalid/missing is silently skipped by
      ``pick`` (and ``pick`` returns ``None`` once *every* entry is invalid);
    * the enabled set is capped at ``MAX_ENABLED_ASSETS`` (64).

The palette talks to the stage through a tiny duck-typed surface
(``GetPrimAtPath``/``Traverse``), so these tests inject fakes rather than
requiring USD to be installed.

Requirements covered:
    2.3 -- unresolved/invalid asset paths are rejected
    2.6 -- an empty enabled set yields nothing to paint (no valid pick)
    9.2 -- assets that no longer resolve are excluded from selection
    9.7 -- when every enabled asset fails validation there is no valid pick
"""

from __future__ import annotations

import random

import pytest

from lightspeed.trex.scatter.models import AssetRef
from lightspeed.trex.scatter.palette import MAX_ENABLED_ASSETS, AssetPalette


# --------------------------------------------------------------------------- #
# Fakes: a minimal duck-typed stage + prim
# --------------------------------------------------------------------------- #


class FakePrim:
    """A minimal stand-in for ``pxr.Usd.Prim``.

    Exposes only what the palette consults: ``IsValid`` (referenceability) and
    ``GetPath`` (for discovery/identity). ``valid`` is mutable so a test can flip
    a previously-valid prim to invalid to model an asset that disappears.
    """

    def __init__(self, path: str, valid: bool = True, has_refs: bool = True) -> None:
        self._path = path
        self.valid = valid
        self.has_refs = has_refs

    def IsValid(self) -> bool:  # noqa: N802 - mirrors pxr API
        return self.valid

    def GetPath(self) -> str:  # noqa: N802 - mirrors pxr API
        return self._path

    def HasAuthoredReferences(self) -> bool:  # noqa: N802 - mirrors pxr API
        return self.has_refs


class FakeStage:
    """A duck-typed stage backed by an explicit ``path -> prim`` map.

    ``GetPrimAtPath`` returns the registered prim for a known path or an
    *invalid* prim for anything unknown (mirroring USD, which hands back an
    invalid prim rather than ``None`` for an absent path). The same prim object
    is returned for repeat lookups so a test can mutate it after ``add_asset``.
    """

    def __init__(self, prims: dict | None = None) -> None:
        self._prims = dict(prims or {})

    def register(self, path: str, prim: FakePrim) -> None:
        self._prims[path] = prim

    def GetPrimAtPath(self, path: str):  # noqa: N802 - mirrors pxr API
        if path in self._prims:
            return self._prims[path]
        return FakePrim(path, valid=False)

    def Traverse(self):  # noqa: N802 - mirrors pxr API
        return list(self._prims.values())


class AllValidStage:
    """A stage that resolves *any* path to a fresh valid prim.

    Useful for the cap test where the path identities are irrelevant and we only
    care that validation always succeeds.
    """

    def GetPrimAtPath(self, path: str):  # noqa: N802 - mirrors pxr API
        return FakePrim(path, valid=True)


def _rng() -> random.Random:
    return random.Random(1234)


# --------------------------------------------------------------------------- #
# Empty palette: is_empty + pick == None (Property 8, Requirements 2.6/9.7)
# --------------------------------------------------------------------------- #


def test_empty_palette_is_empty_no_stage():
    palette = AssetPalette()
    assert palette.is_empty() is True
    assert len(palette) == 0
    assert palette.list_assets() == []


def test_empty_palette_pick_returns_none_no_stage():
    # Property 8 (palette level): an empty palette has no valid pick.
    palette = AssetPalette()
    assert palette.pick(_rng()) is None


def test_empty_palette_pick_returns_none_with_stage():
    # Same property holds when a stage is attached -- still nothing to pick.
    palette = AssetPalette(stage=FakeStage())
    assert palette.is_empty() is True
    assert palette.pick(_rng()) is None


def test_pick_on_empty_palette_is_stable_across_many_seeds():
    # No matter the RNG draw, an empty palette can never produce an asset.
    palette = AssetPalette(stage=FakeStage())
    for seed in range(50):
        assert palette.pick(random.Random(seed)) is None


# --------------------------------------------------------------------------- #
# add_asset rejects unresolved paths and leaves the palette unchanged (Req 2.3)
# --------------------------------------------------------------------------- #


def test_add_asset_rejects_unresolved_path():
    # The stage knows nothing about this path -> invalid prim -> rejected.
    palette = AssetPalette(stage=FakeStage())
    with pytest.raises(ValueError, match="does not resolve"):
        palette.add_asset("/World/missing")


def test_add_asset_rejects_explicitly_invalid_prim():
    stage = FakeStage({"/World/ghost": FakePrim("/World/ghost", valid=False)})
    palette = AssetPalette(stage=stage)
    with pytest.raises(ValueError, match="does not resolve"):
        palette.add_asset("/World/ghost")


def test_rejected_add_leaves_palette_unchanged():
    # A good asset is registered first; a failed add must not perturb it.
    stage = FakeStage({"/World/grass": FakePrim("/World/grass", valid=True)})
    palette = AssetPalette(stage=stage)
    palette.add_asset("/World/grass")
    before = palette.list_assets()
    assert len(before) == 1

    with pytest.raises(ValueError):
        palette.add_asset("/World/nope")

    after = palette.list_assets()
    assert len(after) == 1
    assert after[0].prim_path == "/World/grass"
    assert [a.prim_path for a in after] == [a.prim_path for a in before]


def test_add_asset_rejects_empty_path():
    palette = AssetPalette(stage=FakeStage())
    with pytest.raises(ValueError, match="non-empty string"):
        palette.add_asset("")
    assert palette.is_empty() is True


def test_add_asset_without_stage_skips_validation():
    # With no stage attached, validation cannot run, so adds are accepted.
    palette = AssetPalette()
    palette.add_asset("/World/grass")
    assert len(palette) == 1
    assert palette.list_assets()[0].prim_path == "/World/grass"


# --------------------------------------------------------------------------- #
# Entries that stop resolving are excluded by pick (Requirements 9.2, 9.7)
# --------------------------------------------------------------------------- #


def test_pick_excludes_entry_that_became_invalid():
    good = FakePrim("/World/good", valid=True)
    gone = FakePrim("/World/gone", valid=True)
    stage = FakeStage({"/World/good": good, "/World/gone": gone})
    palette = AssetPalette(stage=stage)
    palette.add_asset("/World/good")
    palette.add_asset("/World/gone")
    assert len(palette) == 2

    # The asset file/prim disappears after being added.
    gone.valid = False

    # Every pick must now return the one still-valid asset, never the stale one.
    for seed in range(50):
        picked = palette.pick(random.Random(seed))
        assert picked is not None
        assert picked.prim_path == "/World/good"


def test_pick_returns_none_when_all_entries_invalid():
    # Requirement 9.7: if every enabled asset fails validation, there is no
    # valid pick (the stroke orchestrator turns this into a user-facing warning).
    a = FakePrim("/World/a", valid=True)
    b = FakePrim("/World/b", valid=True)
    stage = FakeStage({"/World/a": a, "/World/b": b})
    palette = AssetPalette(stage=stage)
    palette.add_asset("/World/a")
    palette.add_asset("/World/b")

    a.valid = False
    b.valid = False

    # Entries remain in the palette (not auto-removed), but none is selectable.
    assert len(palette) == 2
    assert palette.is_empty() is False
    for seed in range(50):
        assert palette.pick(random.Random(seed)) is None


def test_pick_returns_valid_asset_for_all_valid_palette():
    a = FakePrim("/World/a", valid=True)
    stage = FakeStage({"/World/a": a})
    palette = AssetPalette(stage=stage)
    palette.add_asset("/World/a")
    picked = palette.pick(_rng())
    assert isinstance(picked, AssetRef)
    assert picked.prim_path == "/World/a"


# --------------------------------------------------------------------------- #
# MAX_ENABLED_ASSETS cap (Requirement 2.4 boundary touched by 2.3 validation)
# --------------------------------------------------------------------------- #


def test_adding_new_asset_beyond_cap_is_rejected():
    palette = AssetPalette(stage=AllValidStage())
    for i in range(MAX_ENABLED_ASSETS):
        palette.add_asset(f"/World/asset_{i}")
    assert len(palette) == MAX_ENABLED_ASSETS

    with pytest.raises(ValueError, match="cannot enable more than"):
        palette.add_asset(f"/World/asset_{MAX_ENABLED_ASSETS}")

    # The cap holds: the palette is unchanged after the rejected add.
    assert len(palette) == MAX_ENABLED_ASSETS


def test_reweighting_existing_asset_at_cap_is_allowed():
    # Re-adding an existing path updates its weight rather than counting as a
    # new entry, so it must be permitted even when the palette is full.
    palette = AssetPalette(stage=AllValidStage())
    for i in range(MAX_ENABLED_ASSETS):
        palette.add_asset(f"/World/asset_{i}")

    palette.add_asset("/World/asset_0", weight=5.0)
    assert len(palette) == MAX_ENABLED_ASSETS
    asset0 = next(a for a in palette.list_assets() if a.prim_path == "/World/asset_0")
    assert asset0.weight == 5.0
