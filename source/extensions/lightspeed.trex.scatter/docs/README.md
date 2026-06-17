# lightspeed.trex.scatter

Native RTX Remix Toolkit scatter brush. Paint grass, rocks, and vegetation onto
captured `mesh_HASH` prims as USD point instancers, authored non-destructively
into the mod layer and rendered in-game via the RTX Remix runtime instancer.

This extension is self-contained: it implements scatter distribution, viewport
raycasting, and USD point-instancing authoring without importing any
`omni.paint.*` module. It depends only on host-available toolkit services
(`omni.ui`, `omni.usd`, `omni.kit.viewport.utility`, `omni.kit.undo`, and the
`pxr` USD libraries provided by `omni.usd.libs`).

Status: experimental. Delivered on a dedicated experimental branch of the
toolkit-remix fork.
