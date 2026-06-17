"""Unit tests for the ``BrushPanel`` settings core (task 8.2).

These tests exercise the UI-free settings/snapshot logic of ``BrushPanel``
without ``omni.ui`` (the import is guarded by ``HAS_OMNI_UI``). They assert that:

    - A valid control change produces an accepted, valid ``BrushSettings``
      snapshot and emits exactly one snapshot to a registered callback
      (Requirement 4.2).
    - Out-of-range control entries are rejected without changing the last valid
      value, record a non-``None`` error, and emit no snapshot (Requirement 4.2).
    - An inverted scale range is rejected and the last valid scale is retained
      (Requirement 4.5).
    - A fresh panel exposes valid defaults (``instancing_mode`` defaults to
      "Point Instancing").
    - Active brush-asset selection rejects non-USD files and retains the prior
      selection (Requirements 2.2, 2.3) while accepting a valid ``.usda`` path.

Validates: Requirements 4.2, 4.5
"""

from __future__ import annotations

import pytest

from lightspeed.trex.scatter.models import POINT_INSTANCING, BrushSettings
from lightspeed.trex.scatter.panel import (
    DENSITY_MAX,
    DENSITY_MIN,
    RADIUS_MAX,
    RADIUS_MIN,
    YAW_MAX,
    YAW_MIN,
    BrushPanel,
)


def _recording_panel():
    """Return a fresh panel and a list that captures every emitted snapshot."""
    panel = BrushPanel()
    snapshots = []
    panel.on_settings_changed(snapshots.append)
    return panel, snapshots


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
def test_fresh_panel_defaults_are_valid():
    panel = BrushPanel()
    settings = panel.get_settings()

    assert isinstance(settings, BrushSettings)
    # Constructing the snapshot already ran BrushSettings.validate(); re-running
    # it confirms the defaults satisfy every model invariant.
    settings.validate()
    assert settings.instancing_mode == POINT_INSTANCING
    assert panel.error is None


# ---------------------------------------------------------------------------
# Accepted control change (Requirement 4.2)
# ---------------------------------------------------------------------------
def test_valid_radius_change_emits_single_valid_snapshot():
    panel, snapshots = _recording_panel()

    accepted = panel.set_radius(5.0)

    assert accepted is True
    # Exactly one snapshot emitted for the single accepted change.
    assert len(snapshots) == 1
    emitted = snapshots[0]
    assert isinstance(emitted, BrushSettings)
    emitted.validate()
    assert emitted.radius == 5.0

    # get_settings() reflects the change and returns the same valid snapshot.
    current = panel.get_settings()
    assert current is emitted
    assert current.radius == 5.0
    assert panel.get_value("radius") == 5.0
    assert panel.error is None


def test_accepted_change_at_range_boundaries():
    panel, snapshots = _recording_panel()

    assert panel.set_radius(RADIUS_MIN) is True
    assert panel.set_radius(RADIUS_MAX) is True
    assert panel.set_density(DENSITY_MIN) is True
    assert panel.set_density(DENSITY_MAX) is True
    assert panel.set_yaw_jitter(YAW_MIN) is True
    assert panel.set_yaw_jitter(YAW_MAX) is True

    # One snapshot per accepted change.
    assert len(snapshots) == 6
    panel.get_settings().validate()


# ---------------------------------------------------------------------------
# Rejected out-of-range control changes (Requirement 4.2)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "setter_name, bad_value, field",
    [
        ("set_radius", RADIUS_MIN - 0.001, "radius"),
        ("set_radius", RADIUS_MAX + 1.0, "radius"),
        ("set_density", DENSITY_MIN - 0.001, "density"),
        ("set_density", DENSITY_MAX + 1.0, "density"),
        ("set_yaw_jitter", YAW_MIN - 1.0, "yaw_jitter"),
        ("set_yaw_jitter", YAW_MAX + 1.0, "yaw_jitter"),
    ],
)
def test_out_of_range_change_is_rejected_without_mutating_state(
    setter_name, bad_value, field
):
    panel, snapshots = _recording_panel()
    before_settings = panel.get_settings()
    before_value = panel.get_value(field)

    accepted = getattr(panel, setter_name)(bad_value)

    assert accepted is False
    # Last valid value/snapshot retained.
    assert panel.get_value(field) == before_value
    assert panel.get_settings() is before_settings
    # Non-None error recorded for the rejected change.
    assert panel.error is not None
    # No snapshot emitted for a rejected change.
    assert snapshots == []


def test_error_clears_after_a_subsequent_valid_change():
    panel, snapshots = _recording_panel()

    assert panel.set_radius(RADIUS_MAX + 100.0) is False
    assert panel.error is not None

    assert panel.set_radius(2.0) is True
    assert panel.error is None
    assert len(snapshots) == 1


# ---------------------------------------------------------------------------
# Inverted scale range rejection (Requirement 4.5)
# ---------------------------------------------------------------------------
def test_scale_min_above_current_max_is_rejected_and_retains_last_valid():
    panel, snapshots = _recording_panel()
    # Defaults: scale_min == scale_max == 1.0.
    before = panel.get_settings().scale_range

    # Pushing min above the current max inverts the range -> rejected.
    accepted = panel.set_scale_min(5.0)

    assert accepted is False
    assert panel.get_settings().scale_range == before
    assert panel.get_value("scale_min") == before[0]
    assert panel.error is not None
    assert snapshots == []


def test_set_scale_range_with_min_greater_than_max_is_rejected():
    panel, snapshots = _recording_panel()
    before = panel.get_settings().scale_range

    accepted = panel.set_scale_range(5.0, 2.0)

    assert accepted is False
    assert panel.get_settings().scale_range == before
    assert panel.error is not None
    assert snapshots == []


def test_valid_scale_range_widening_is_accepted():
    panel, snapshots = _recording_panel()

    accepted = panel.set_scale_range(0.5, 3.0)

    assert accepted is True
    assert panel.get_settings().scale_range == (0.5, 3.0)
    assert len(snapshots) == 1
    panel.get_settings().validate()


# ---------------------------------------------------------------------------
# Active brush-asset selection (Requirements 2.2, 2.3)
# ---------------------------------------------------------------------------
def test_select_asset_rejects_non_usd_extension_and_retains_previous():
    panel = BrushPanel()
    # Establish a valid prior selection first.
    assert panel.select_asset("C:/assets/tree.usda") is True
    assert panel.selected_asset_path == "C:/assets/tree.usda"
    assert panel.asset_error is None

    # A non-USD file is rejected; the previous selection is retained.
    accepted = panel.select_asset("C:/assets/texture.png")

    assert accepted is False
    assert panel.selected_asset_path == "C:/assets/tree.usda"
    assert panel.asset_error is not None


def test_select_asset_rejection_from_empty_keeps_none_selected():
    panel = BrushPanel()

    accepted = panel.select_asset("C:/assets/model.fbx")

    assert accepted is False
    assert panel.selected_asset_path is None
    assert panel.asset_error is not None


@pytest.mark.parametrize("path", ["C:/a/rock.usd", "C:/a/rock.usda", "C:/a/rock.usdc", "C:/a/rock.usdz"])
def test_select_asset_accepts_valid_usd_extensions(path):
    panel = BrushPanel()

    accepted = panel.select_asset(path)

    assert accepted is True
    assert panel.selected_asset_path == path
    assert panel.asset_error is None


def test_select_asset_respects_asset_validator():
    # Validator rejects everything -> a well-formed .usda path is still rejected.
    panel = BrushPanel(asset_validator=lambda _p: False)

    accepted = panel.select_asset("C:/assets/tree.usda")

    assert accepted is False
    assert panel.selected_asset_path is None
    assert panel.asset_error is not None
