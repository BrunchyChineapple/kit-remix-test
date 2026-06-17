# Changelog

All notable changes to the `lightspeed.trex.scatter` extension are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0]

### Added

- Initial experimental release of the native scatter/paint brush.
- Paint grass, rocks, and vegetation onto captured meshes as referencing USD
  prims authored into a dedicated `scatter.usda` sublayer (non-destructive,
  additive, undoable).
- Components: validated data models, evolving weighted `AssetPalette`,
  `SceneRaycaster`, `PlacementSampler` (density/jitter/scale, surface-normal
  alignment), `ScatterLayerWriter`, `ScatterBrush` stroke orchestrator,
  `BrushPanel` (omni.ui), `ViewportInputHandler`, and the `omni.ext` entry
  point with stage-event recovery and read-only/missing mod-layer handling.
