"""Thin shim so ``python src/main.py ...`` works as documented in README 4.4 (see D-001)."""

import sys

from foveamap.cli import main

if __name__ == "__main__":
    sys.exit(main())
