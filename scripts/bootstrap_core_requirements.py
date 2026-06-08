#!/usr/bin/env python3
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from ballontranslator.utils.core_requirements import ensure_core_requirements
    ensure_core_requirements(repo_root=str(repo_root), force=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
