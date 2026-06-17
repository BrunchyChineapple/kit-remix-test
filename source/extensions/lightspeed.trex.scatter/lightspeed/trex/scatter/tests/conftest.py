"""Pytest configuration for the scatter extension tests.

The extension follows the Kit layout ``<ext-root>/lightspeed/trex/scatter`` where
``lightspeed`` and ``lightspeed.trex`` are implicit namespace packages. When the
tests are run with a plain ``pytest`` invocation (outside the Kit runtime) the
extension root is not on ``sys.path``. This adds it so ``lightspeed.trex.scatter``
imports resolve to the in-tree package.
"""

from __future__ import annotations

import sys
from pathlib import Path

# tests/ -> scatter/ -> trex/ -> lightspeed/ -> <extension root>
_EXTENSION_ROOT = Path(__file__).resolve().parents[4]

if str(_EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_ROOT))
