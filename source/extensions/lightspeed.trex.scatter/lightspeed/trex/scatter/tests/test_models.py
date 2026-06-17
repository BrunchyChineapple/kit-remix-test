"""Unit tests for scatter brush data-model validation (task 1.3).

These example-based tests exercise the construction-time validation in
``models.py``: every invalid ``BrushSettings`` must raise ``ValueError``, valid
settings must construct, and the documented defaults must match the design.

Requirements covered:
    4.2  -- out-of-range brush parameters are rejected
    4.5  -- min scale must not exceed max scale
    4.11 -- instancing-mode options and "Point Instancing" default
"""

from __future__ import annotations

import pytest

from lightspeed.trex.scatter.models import (
    DEFAULT_INSTANCING_MODE,
    INSTANCING_NONE,
    POINT_INSTANCING,
    BrushSettings,
)

# A baseline set of valid keyword arguments. Individual tests override one field
# at a time so each assertion isolates a single validation rule.
_VALID_KWARGS = dict(
    radius=1.0,
    density=2.0,
    spacing=0.5,
    position_jitter=0.5,
    yaw_range=(0.0, 90.0),
    scale_range=(0.5, 2.0),
)


def _make(**overrides) -> BrushSettings:
    kwargs = dict(_VALID_KWARGS)
    kwargs.update(overrides)
    return BrushSettings(**kwargs)


# --------------------------------------------------------------------------- #
# Valid construction + defaults (Requirements 4.11)
# --------------------------------------------------------------------------- #


def test_valid_settings_construct():
    settings = _make()
    assert settings.radius == 1.0
    assert settings.density == 2.0
    assert settings.spacing == 0.5
    assert settings.position_jitter == 0.5
    assert settings.yaw_range == (0.0, 90.0)
    assert settings.scale_range == (0.5, 2.0)


def test_defaults_match_design():
    settings = _make()
    # Defaults documented in the design "Model: BrushSettings".
    assert settings.align_to_normal is True
    assert settings.normal_blend == 1.0
    assert settings.seed == 0
    assert settings.max_instances_per_stroke == 10000
    assert settings.instancing_mode == "Point Instancing"
    assert settings.instancing_mode == DEFAULT_INSTANCING_MODE
    assert DEFAULT_INSTANCING_MODE == POINT_INSTANCING


def test_boundary_values_construct():
    # Inclusive boundaries should be accepted: jitter/normal_blend at 0 and 1,
    # equal yaw bounds, and equal scale bounds.
    settings = _make(
        position_jitter=0.0,
        normal_blend=0.0,
        yaw_range=(45.0, 45.0),
        scale_range=(1.0, 1.0),
    )
    assert settings.position_jitter == 0.0
    assert settings.normal_blend == 0.0
    assert settings.yaw_range == (45.0, 45.0)
    assert settings.scale_range == (1.0, 1.0)

    settings_high = _make(position_jitter=1.0, normal_blend=1.0)
    assert settings_high.position_jitter == 1.0
    assert settings_high.normal_blend == 1.0


def test_spacing_zero_is_valid():
    assert _make(spacing=0.0).spacing == 0.0


def test_both_instancing_modes_construct():
    assert _make(instancing_mode=POINT_INSTANCING).instancing_mode == POINT_INSTANCING
    assert _make(instancing_mode=INSTANCING_NONE).instancing_mode == INSTANCING_NONE


# --------------------------------------------------------------------------- #
# Rejection of out-of-range radius / density / spacing (Requirement 4.2)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("radius", [0.0, -0.01, -100.0])
def test_non_positive_radius_rejected(radius):
    with pytest.raises(ValueError, match="radius"):
        _make(radius=radius)


@pytest.mark.parametrize("density", [0.0, -0.01, -50.0])
def test_non_positive_density_rejected(density):
    with pytest.raises(ValueError, match="density"):
        _make(density=density)


def test_negative_spacing_rejected():
    with pytest.raises(ValueError, match="spacing"):
        _make(spacing=-0.001)


# --------------------------------------------------------------------------- #
# Rejection of out-of-range jitter / normal_blend (Requirement 4.2)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("jitter", [-0.0001, 1.0001, -1.0, 2.0])
def test_position_jitter_out_of_range_rejected(jitter):
    with pytest.raises(ValueError, match="position_jitter"):
        _make(position_jitter=jitter)


@pytest.mark.parametrize("blend", [-0.0001, 1.0001, -1.0, 2.0])
def test_normal_blend_out_of_range_rejected(blend):
    with pytest.raises(ValueError, match="normal_blend"):
        _make(normal_blend=blend)


# --------------------------------------------------------------------------- #
# Rejection of invalid scale_range (Requirements 4.2, 4.5)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scale_min", [0.0, -0.01, -10.0])
def test_non_positive_scale_min_rejected(scale_min):
    with pytest.raises(ValueError, match="scale_range min"):
        _make(scale_range=(scale_min, 2.0))


@pytest.mark.parametrize("scale_range", [(2.0, 1.0), (5.0, 0.5), (1.001, 1.0)])
def test_inverted_scale_range_rejected(scale_range):
    with pytest.raises(ValueError, match="scale_range must be ordered"):
        _make(scale_range=scale_range)


# --------------------------------------------------------------------------- #
# Rejection of inverted yaw_range (Requirement 4.2)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("yaw_range", [(90.0, 0.0), (1.0, -1.0), (360.0, 0.0)])
def test_inverted_yaw_range_rejected(yaw_range):
    with pytest.raises(ValueError, match="yaw_range must be ordered"):
        _make(yaw_range=yaw_range)


# --------------------------------------------------------------------------- #
# Rejection of non-positive max_instances_per_stroke (Requirement 4.2)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cap", [0, -1, -10000])
def test_non_positive_max_instances_rejected(cap):
    with pytest.raises(ValueError, match="max_instances_per_stroke"):
        _make(max_instances_per_stroke=cap)


# --------------------------------------------------------------------------- #
# Rejection of invalid instancing_mode (Requirement 4.11)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "mode",
    ["", "point instancing", "Instancing", "none", "PointInstancing", "Bogus"],
)
def test_invalid_instancing_mode_rejected(mode):
    with pytest.raises(ValueError, match="instancing_mode"):
        _make(instancing_mode=mode)
