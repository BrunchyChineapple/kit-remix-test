"""Cursor-to-world-space surface raycasting for the scatter brush.

``SceneRaycaster`` turns a normalized viewport position into a world-space
surface hit (``SurfaceHit``) by building a ray from the active camera and
querying the scene for the closest intersection. It is the read-only "where is
the cursor pointing in the world" service consumed by ``ScatterBrush`` during a
stroke (design "Component 4: SceneRaycaster").

Design contract (design "Function: SceneRaycaster.raycast()"):
    Preconditions
        - ``screen_pos`` components are in normalized viewport range ``[0, 1]``.
        - An active camera and stage exist.
    Postconditions
        - Returns ``None`` if the ray hits no imageable geometry.
        - On hit, ``result.normal`` is unit length and ``result.prim_path`` is
          valid.
        - No side effects: the stage is not mutated.

Testability / dependency injection
    The constructor accepts an injectable ``viewport_api`` (the source of the
    active camera and, optionally, a ray-building helper) and an injectable
    ``scene_query`` callable (the closest-hit query). This lets unit tests
    (task 3.2) drive the raycaster with a fake viewport/camera and a fake scene
    query without a live Kit runtime. When ``scene_query`` is omitted, a default
    backed by the PhysX scene-query interface is constructed lazily on first use.

``pxr`` and ``omni`` imports are guarded (mirroring the ``HAS_PXR`` pattern in
``models.py``) so this module remains importable -- and the ray/hit plumbing
unit-testable -- in environments where USD / Kit are not installed.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

from .models import HAS_PXR, SurfaceHit

try:  # pragma: no cover - import guard exercised only by environment
    from pxr import Gf  # type: ignore
except ImportError:  # pragma: no cover - exercised only without USD installed
    Gf = None  # type: ignore

__all__ = ["SceneRaycaster"]

# Keys under which a scene-query result may expose its fields. Different Kit
# backends (PhysX scene query, omni.kit.raycast.query) name these differently,
# so we probe a small set of aliases and also fall back to attribute access.
_POSITION_KEYS = ("position", "world_pos", "hit_position", "point")
_NORMAL_KEYS = ("normal", "world_normal", "hit_normal")
_PRIM_PATH_KEYS = ("rigidBody", "collision", "primPath", "prim_path", "path")
_HIT_FLAG_KEYS = ("hit", "valid", "is_hit")


def _field(result: object, keys: Sequence[str], default=None):
    """Read the first present field of ``result`` from ``keys``.

    Supports both mapping-style results (``dict``) and attribute-style results
    (objects returned by some Kit query interfaces).
    """
    for key in keys:
        if isinstance(result, dict):
            if key in result and result[key] is not None:
                return result[key]
        elif hasattr(result, key):
            value = getattr(result, key)
            if value is not None:
                return value
    return default


def _as_vec3d(value: object):
    """Coerce a 3-component value into a ``Gf.Vec3d`` when USD is available.

    Without USD installed (tests), the value is returned unchanged so the model
    layer's length helpers can still operate on plain sequences.
    """
    if value is None:
        return None
    if HAS_PXR and Gf is not None and not isinstance(value, Gf.Vec3d):
        try:
            return Gf.Vec3d(float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, IndexError, ValueError):
            return value
    return value


def _length(vec: object) -> float:
    """Euclidean length of a ``Gf.Vec3d`` or a plain 3-sequence."""
    get_length = getattr(vec, "GetLength", None)
    if callable(get_length):
        return float(get_length())
    return float(sum(float(c) * float(c) for c in vec)) ** 0.5


def _normalized(vec: object, fallback: object):
    """Return ``vec`` scaled to unit length.

    If ``vec`` is degenerate (near-zero length) the ``fallback`` direction is
    used instead, so a hit always yields a unit-length normal as required by the
    ``SurfaceHit`` contract.
    """
    length = _length(vec)
    if length <= 1e-9:
        vec = fallback
        length = _length(vec)
        if length <= 1e-9:  # both degenerate -- cannot form a valid normal
            return None

    get_normalized = getattr(vec, "GetNormalized", None)
    if callable(get_normalized):
        return get_normalized()
    # Plain-sequence fallback (no USD installed).
    return type(vec)(c / length for c in vec) if not isinstance(vec, (list, tuple)) else [c / length for c in vec]


class SceneRaycaster:
    """Resolve a normalized cursor position to a world-space ``SurfaceHit``.

    Args:
        viewport_api: The active viewport. Used to build the world-space ray
            from the active camera. If it exposes a ``compute_ray(screen_pos)``
            method returning ``(origin, direction)``, that is used directly;
            otherwise the ray is unprojected from the viewport's ``transform``
            (camera-to-world) and ``projection`` matrices.
        scene_query: Optional callable ``(origin, direction) -> result`` that
            returns the closest scene hit (or a falsy/empty result on miss).
            When omitted, a PhysX-backed query is built lazily on first use.
    """

    def __init__(
        self,
        viewport_api: object,
        scene_query: Optional[Callable[[object, object], object]] = None,
    ) -> None:
        self._viewport_api = viewport_api
        self._scene_query = scene_query

    def raycast(self, screen_pos: "object") -> Optional[SurfaceHit]:
        """Cast a ray through ``screen_pos`` and return the closest surface hit.

        Args:
            screen_pos: A 2-component value (``Vec2``, ``(x, y)`` tuple, or an
                object exposing ``x``/``y``) in normalized viewport range
                ``[0, 1]``, with ``(0, 0)`` at the top-left.

        Returns:
            A ``SurfaceHit`` with a unit-length ``normal`` and a valid
            ``prim_path`` on hit, or ``None`` when the ray misses all geometry.

        Raises:
            ValueError: If ``screen_pos`` is out of the normalized ``[0, 1]``
                range, or if no active camera/viewport is available
                (precondition violations).
        """
        sx, sy = self._screen_components(screen_pos)
        if self._viewport_api is None:
            raise ValueError("no active viewport/camera available for raycast")

        ray = self._build_ray(sx, sy)
        if ray is None:
            return None
        origin, direction = ray

        query = self._scene_query or self._default_scene_query()
        result = query(origin, direction)
        if not self._is_hit(result):
            return None

        prim_path = _field(result, _PRIM_PATH_KEYS, default="")
        prim_path = str(prim_path) if prim_path is not None else ""
        if not prim_path or not self._prim_is_valid(prim_path):
            # A hit without a resolvable imageable prim is treated as a miss so
            # the postcondition "prim_path is valid on hit" always holds.
            return None

        point = _as_vec3d(_field(result, _POSITION_KEYS))
        if point is None:
            return None

        raw_normal = _as_vec3d(_field(result, _NORMAL_KEYS))
        # Fall back to a normal facing back along the ray when the backend does
        # not supply one (or supplies a degenerate one).
        fallback = self._reverse_direction(direction)
        normal = _normalized(raw_normal if raw_normal is not None else fallback, fallback)
        if normal is None:
            return None

        return SurfaceHit(point=point, normal=normal, prim_path=prim_path)

    # -- ray construction --------------------------------------------------

    def _build_ray(self, sx: float, sy: float):
        """Build a world-space ``(origin, direction)`` ray for normalized coords.

        Prefers a ``compute_ray`` hook on the viewport when present (the seam
        used by fakes); otherwise unprojects the near/far NDC points through the
        inverse projection and camera-to-world transforms.
        """
        compute_ray = getattr(self._viewport_api, "compute_ray", None)
        if callable(compute_ray):
            ray = compute_ray((sx, sy))
            if ray is None:
                return None
            origin, direction = ray
            return _as_vec3d(origin), _as_vec3d(direction)

        if not HAS_PXR or Gf is None:
            raise ValueError(
                "viewport_api has no compute_ray and USD (pxr) is unavailable; "
                "cannot build a world-space ray"
            )

        projection = getattr(self._viewport_api, "projection", None)
        transform = getattr(self._viewport_api, "transform", None)
        if projection is None or transform is None:
            raise ValueError(
                "viewport_api must expose 'projection' and 'transform' matrices "
                "(or a compute_ray helper) to build a ray"
            )

        projection = Gf.Matrix4d(projection)
        cam_to_world = Gf.Matrix4d(transform)
        inv_proj = projection.GetInverse()

        # Normalized viewport (top-left origin) -> NDC (centre origin, y up).
        ndc_x = sx * 2.0 - 1.0
        ndc_y = 1.0 - sy * 2.0

        near_world = self._unproject(ndc_x, ndc_y, -1.0, inv_proj, cam_to_world)
        far_world = self._unproject(ndc_x, ndc_y, 1.0, inv_proj, cam_to_world)

        direction = (far_world - near_world).GetNormalized()
        return near_world, direction

    @staticmethod
    def _unproject(ndc_x, ndc_y, ndc_z, inv_proj, cam_to_world):
        """Unproject a single NDC point to world space (row-vector convention)."""
        clip = Gf.Vec4d(ndc_x, ndc_y, ndc_z, 1.0)
        view = clip * inv_proj
        if view[3] != 0.0:
            view = view / view[3]
        return cam_to_world.Transform(Gf.Vec3d(view[0], view[1], view[2]))

    @staticmethod
    def _reverse_direction(direction: object):
        """Return ``-direction`` as a unit vector, for use as a fallback normal."""
        get_normalized = getattr(direction, "GetNormalized", None)
        if callable(get_normalized):
            neg = direction * -1.0
            return neg.GetNormalized()
        length = _length(direction) or 1.0
        return [-c / length for c in direction]

    # -- scene query -------------------------------------------------------

    def _default_scene_query(self) -> Callable[[object, object], object]:
        """Build (and cache) the default PhysX-backed closest-hit query.

        Imported lazily and guarded so the module imports without ``omni.physx``
        and so tests that inject their own ``scene_query`` never touch PhysX.
        """
        from omni.physx import get_physx_scene_query_interface  # type: ignore

        scene_query_iface = get_physx_scene_query_interface()

        def _query(origin: object, direction: object) -> object:
            origin_t = (float(origin[0]), float(origin[1]), float(origin[2]))
            dir_t = (float(direction[0]), float(direction[1]), float(direction[2]))
            return scene_query_iface.raycast_closest(origin_t, dir_t, 1.0e8)

        self._scene_query = _query
        return _query

    @staticmethod
    def _is_hit(result: object) -> bool:
        """Return ``True`` when a query result represents a real intersection."""
        if not result:
            return False
        flag = _field(result, _HIT_FLAG_KEYS, default=None)
        if flag is not None:
            return bool(flag)
        # No explicit hit flag: treat a result carrying a position as a hit.
        return _field(result, _POSITION_KEYS) is not None

    # -- stage validation --------------------------------------------------

    def _prim_is_valid(self, prim_path: str) -> bool:
        """Best-effort check that ``prim_path`` resolves to a prim on the stage.

        Reads the stage from the viewport (or the USD context) without mutating
        it. When no stage is reachable (e.g. tests with a fake viewport), the
        path is accepted -- the backend already reported it as the hit prim.
        """
        stage = self._get_stage()
        if stage is None:
            return True
        get_prim = getattr(stage, "GetPrimAtPath", None)
        if not callable(get_prim):
            return True
        prim = get_prim(prim_path)
        is_valid = getattr(prim, "IsValid", None)
        return bool(is_valid()) if callable(is_valid) else bool(prim)

    def _get_stage(self):
        """Resolve the active USD stage read-only, or ``None`` if unavailable."""
        stage = getattr(self._viewport_api, "stage", None)
        if stage is not None:
            return stage
        usd_context = getattr(self._viewport_api, "usd_context", None)
        get_stage = getattr(usd_context, "get_stage", None)
        if callable(get_stage):
            return get_stage()
        return None

    # -- input helpers -----------------------------------------------------

    @staticmethod
    def _screen_components(screen_pos: object) -> Tuple[float, float]:
        """Extract ``(x, y)`` floats from a Vec2 / tuple / object, validated.

        Enforces the precondition that both components lie in ``[0, 1]``.
        """
        x_attr = getattr(screen_pos, "x", None)
        y_attr = getattr(screen_pos, "y", None)
        if x_attr is not None and y_attr is not None:
            sx, sy = float(x_attr), float(y_attr)
        else:
            try:
                sx, sy = float(screen_pos[0]), float(screen_pos[1])
            except (TypeError, IndexError, ValueError) as exc:
                raise ValueError(
                    f"screen_pos must be a 2-component value, got {screen_pos!r}"
                ) from exc

        if not (0.0 <= sx <= 1.0 and 0.0 <= sy <= 1.0):
            raise ValueError(
                f"screen_pos components must be in normalized [0, 1], got ({sx}, {sy})"
            )
        return sx, sy
