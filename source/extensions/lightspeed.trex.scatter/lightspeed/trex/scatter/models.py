"""Validated data models for the scatter brush.

These frozen dataclasses are the value objects passed between the scatter
brush components (panel -> brush -> sampler -> writer). Each model enforces the
validation rules defined in the design document at construction time via
``__post_init__`` (which delegates to an explicit ``validate()`` method), so an
invalid model can never be constructed.

The geometric types (``point``, ``normal``, ``translate``, ``orient``) are USD
``pxr.Gf`` values at runtime. ``pxr`` import is guarded so this module remains
importable -- and therefore unit-testable -- in environments where USD is not
installed. When ``pxr`` is unavailable, callers may pass any object exposing the
same length/normalisation surface (or a plain sequence of floats); the length
helpers below adapt to both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

try:  # pragma: no cover - import guard exercised only by environment
    from pxr import Gf  # type: ignore

    HAS_PXR = True
except ImportError:  # pragma: no cover - exercised only without USD installed
    Gf = None  # type: ignore
    HAS_PXR = False

__all__ = [
    "HAS_PXR",
    "InstancingMode",
    "POINT_INSTANCING",
    "INSTANCING_NONE",
    "DEFAULT_INSTANCING_MODE",
    "NORMAL_TOLERANCE",
    "BrushSettings",
    "SurfaceHit",
    "AssetRef",
    "Placement",
]

# Instancing-mode options (Requirement 4.11). The brush authors USD
# ``PointInstancer`` prims by default; "None" authors individual reference
# Xforms instead.
POINT_INSTANCING = "Point Instancing"
INSTANCING_NONE = "None"
DEFAULT_INSTANCING_MODE = POINT_INSTANCING
InstancingMode = str
_VALID_INSTANCING_MODES = (POINT_INSTANCING, INSTANCING_NONE)

# Tolerance used when checking that a normal is unit length and that a
# quaternion is normalised. Sampling and raycast math accumulate float error,
# so an exact == 1.0 check would be too strict.
NORMAL_TOLERANCE = 1e-4


def _vector_length(vec: object) -> float:
    """Return the Euclidean length of a vector-like value.

    Accepts a ``pxr.Gf.Vec*`` (which exposes ``GetLength``) or any finite-length
    iterable of numbers (e.g. a ``tuple`` used in tests without USD installed).
    """
    get_length = getattr(vec, "GetLength", None)
    if callable(get_length):
        return float(get_length())
    try:
        components = list(vec)  # type: ignore[arg-type]
    except TypeError as exc:  # not iterable and no GetLength -> unusable
        raise TypeError(f"cannot measure length of {vec!r}") from exc
    return math.sqrt(sum(float(c) * float(c) for c in components))


def _quaternion_length(quat: object) -> float:
    """Return the norm of a quaternion-like value.

    Accepts a ``pxr.Gf.Quat*`` (via ``GetLength``) or, as a fallback, an object
    exposing ``GetReal``/``GetImaginary`` or a 4-element iterable ``(w, x, y, z)``.
    """
    get_length = getattr(quat, "GetLength", None)
    if callable(get_length):
        return float(get_length())

    get_real = getattr(quat, "GetReal", None)
    get_imaginary = getattr(quat, "GetImaginary", None)
    if callable(get_real) and callable(get_imaginary):
        real = float(get_real())
        imaginary = get_imaginary()
        return math.sqrt(real * real + _vector_length(imaginary) ** 2)

    try:
        components = [float(c) for c in quat]  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"cannot measure length of quaternion {quat!r}") from exc
    return math.sqrt(sum(c * c for c in components))


@dataclass(frozen=True)
class BrushSettings:
    """Immutable snapshot of the brush controls for one stroke.

    Validation rules (design "Model: BrushSettings"):
        - ``radius > 0``, ``density > 0``, ``spacing >= 0``
        - ``0 <= position_jitter <= 1``, ``0 <= normal_blend <= 1``
        - ``yaw_range[0] <= yaw_range[1]``
        - ``0 < scale_range[0] <= scale_range[1]``
        - ``max_instances_per_stroke > 0``
        - ``instancing_mode`` is one of the allowed options (default
          "Point Instancing", Requirement 4.11)
    """

    radius: float  # world units, brush disc radius
    density: float  # instances per square world unit
    spacing: float  # min world distance between stroke samples
    position_jitter: float  # 0..1 fraction of radius
    yaw_range: Tuple[float, float]  # degrees, min <= max
    scale_range: Tuple[float, float]  # uniform scale multiplier, 0 < min <= max
    align_to_normal: bool = True  # DEFAULT ON: orient up-axis to the surface normal
    normal_blend: float = 1.0  # 0..1 blend between world-up and surface normal
    seed: int = 0  # RNG seed for reproducibility
    max_instances_per_stroke: int = 10000  # safety cap
    instancing_mode: InstancingMode = DEFAULT_INSTANCING_MODE

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.radius > 0:
            raise ValueError(f"radius must be > 0, got {self.radius}")
        if not self.density > 0:
            raise ValueError(f"density must be > 0, got {self.density}")
        if not self.spacing >= 0:
            raise ValueError(f"spacing must be >= 0, got {self.spacing}")
        if not 0.0 <= self.position_jitter <= 1.0:
            raise ValueError(
                f"position_jitter must be in [0, 1], got {self.position_jitter}"
            )
        if not 0.0 <= self.normal_blend <= 1.0:
            raise ValueError(
                f"normal_blend must be in [0, 1], got {self.normal_blend}"
            )

        yaw_min, yaw_max = self.yaw_range
        if not yaw_min <= yaw_max:
            raise ValueError(
                f"yaw_range must be ordered (min <= max), got {self.yaw_range}"
            )

        scale_min, scale_max = self.scale_range
        if not scale_min > 0:
            raise ValueError(
                f"scale_range min must be > 0, got {scale_min}"
            )
        if not scale_min <= scale_max:
            raise ValueError(
                f"scale_range must be ordered (min <= max), got {self.scale_range}"
            )

        if not self.max_instances_per_stroke > 0:
            raise ValueError(
                f"max_instances_per_stroke must be > 0, got {self.max_instances_per_stroke}"
            )

        if self.instancing_mode not in _VALID_INSTANCING_MODES:
            raise ValueError(
                f"instancing_mode must be one of {_VALID_INSTANCING_MODES}, "
                f"got {self.instancing_mode!r}"
            )


@dataclass(frozen=True)
class SurfaceHit:
    """A world-space ray-surface intersection result.

    Validation rules (design "Model: SurfaceHit"):
        - ``normal`` is unit length (within ``NORMAL_TOLERANCE``)
        - ``prim_path`` is a non-empty path string

    (Stage-level validity of ``prim_path`` -- that it resolves to an imageable
    prim -- is checked by the raycaster against a live stage, not here.)
    """

    point: object  # Gf.Vec3d: world-space hit position
    normal: object  # Gf.Vec3d: world-space unit surface normal
    prim_path: str  # path of the hit prim

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.prim_path, str) or not self.prim_path:
            raise ValueError(f"prim_path must be a non-empty string, got {self.prim_path!r}")
        length = _vector_length(self.normal)
        if abs(length - 1.0) > NORMAL_TOLERANCE:
            raise ValueError(
                f"normal must be unit length (within {NORMAL_TOLERANCE}), "
                f"got length {length}"
            )


@dataclass(frozen=True)
class AssetRef:
    """A reference to a source asset prim that can be scattered.

    Validation rules (design "Model: AssetRef"):
        - ``prim_path`` is a non-empty path string
        - ``layer_id`` is a string
        - ``weight > 0`` (selection weight)
    """

    prim_path: str  # source prim to reference
    layer_id: str  # identifier of the layer providing the source
    weight: float = 1.0  # selection weight, > 0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.prim_path, str) or not self.prim_path:
            raise ValueError(f"prim_path must be a non-empty string, got {self.prim_path!r}")
        if not isinstance(self.layer_id, str):
            raise ValueError(f"layer_id must be a string, got {self.layer_id!r}")
        if not self.weight > 0:
            raise ValueError(f"weight must be > 0, got {self.weight}")


@dataclass(frozen=True)
class Placement:
    """A single concrete instance to author into the scatter layer.

    Validation rules (design "Model: Placement"):
        - ``scale > 0``
        - ``orient`` is a normalized quaternion (within ``NORMAL_TOLERANCE``)
    """

    asset: AssetRef
    translate: object  # Gf.Vec3d: world-space position
    orient: object  # Gf.Quatf: world-space orientation
    scale: float  # uniform scale multiplier

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.asset, AssetRef):
            raise ValueError(f"asset must be an AssetRef, got {self.asset!r}")
        if not self.scale > 0:
            raise ValueError(f"scale must be > 0, got {self.scale}")
        length = _quaternion_length(self.orient)
        if abs(length - 1.0) > NORMAL_TOLERANCE:
            raise ValueError(
                f"orient must be a normalized quaternion (within {NORMAL_TOLERANCE}), "
                f"got norm {length}"
            )
