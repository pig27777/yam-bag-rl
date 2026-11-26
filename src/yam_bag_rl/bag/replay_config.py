"""The locally extracted Monday physical parameters and two reference regimes."""

import json

from ..paths import CONFIGS

SURROGATE = CONFIGS / "surrogate"
MONDAY = json.loads((SURROGATE / "monday.json").read_text())
PAYLOAD = json.loads((SURROGATE / "payload.json").read_text())
