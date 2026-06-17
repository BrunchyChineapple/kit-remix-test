"""Native RTX Remix Toolkit scatter brush extension.

This package reimplements E-Man's "Scatter Brush" as a self-contained Kit
extension for the RTX Remix Toolkit (Lightspeed Trex). It paints grass, rocks,
and vegetation onto captured meshes by authoring USD point instancers into the
mod layer.

The extension is intentionally free of any ``omni.paint.*`` dependency
(Requirement 1.2); all scatter distribution, viewport raycasting, and USD
authoring are implemented here on top of host-available Kit services.

Submodules (added by later tasks):
    models      -- validated data models (BrushSettings, SurfaceHit, ...)
    palette     -- AssetPalette: source asset discovery + weighted selection
    raycaster   -- SceneRaycaster: cursor -> world-space surface hit
    sampler     -- PlacementSampler: density/jitter/scale sampling
    writer      -- ScatterLayerWriter: undoable, non-destructive USD authoring
    brush       -- ScatterBrush: stroke orchestration
    panel       -- BrushPanel: omni.ui controls + asset palette
    input_handler -- ViewportInputHandler: pointer events -> stroke lifecycle
    extension   -- omni.ext.IExt entry point that wires everything together
"""

__all__: list[str] = []
