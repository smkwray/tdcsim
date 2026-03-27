#!/usr/bin/env python3
"""Entry point for TDCsim. Delegates to simulation_core.main()."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from simulation_core import main

if __name__ == '__main__':
    main()
