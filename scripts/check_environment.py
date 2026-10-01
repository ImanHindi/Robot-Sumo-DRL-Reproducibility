from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = [
    "numpy", "yaml", "gymnasium", "stable_baselines3", "sb3_contrib", "torch",
    "pandas", "scipy", "matplotlib", "psutil",
]

print(f"Python executable: {sys.executable}")
print(f"Python version:    {sys.version.split()[0]}")
failed = []
for name in MODULES:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "installed")
        print(f"[OK] {name:20s} {version}")
    except Exception as exc:
        failed.append((name, repr(exc)))
        print(f"[FAIL] {name:18s} {exc}")

if failed:
    print("\nEnvironment is NOT ready.")
    print(f'Install with: "{sys.executable}" -m pip install -r "{ROOT / "requirements.txt"}"')
    raise SystemExit(1)

print("\nEnvironment is ready for Webots RL controllers.")
