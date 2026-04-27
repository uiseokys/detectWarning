from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys


_APP_DIR = Path(__file__).resolve().parent / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

_module = import_module("app.training_dashboard")
for _name in dir(_module):
    if _name.startswith("__") and _name.endswith("__"):
        continue
    globals()[_name] = getattr(_module, _name)

__all__ = [name for name in globals() if not name.startswith("_")]

if __name__ == "__main__":
    _module.main()
