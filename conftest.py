from __future__ import annotations

import os
import sys

# Ensure the project root (where the bot's modules live as plain top-level
# files, not a package) is importable from tests/ regardless of how pytest
# is invoked.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
