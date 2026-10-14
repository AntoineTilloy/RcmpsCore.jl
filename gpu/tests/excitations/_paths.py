"""Locations of the data the tests read outside this directory.

FIXTURES: the fixtures shared with the Julia package, test/fixtures of the repository
(override with RCMPS_TEST_FIXTURES).
PHI4_Z2: the companion repository rcmps-phi4-z2 with the stored Julia references of the
phi^4 mass scan, by default next to this repository (override with RCMPS_PHI4_Z2).
"""
import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
FIXTURES = Path(os.environ.get("RCMPS_TEST_FIXTURES") or REPOSITORY / "test" / "fixtures")
PHI4_Z2 = Path(os.environ.get("RCMPS_PHI4_Z2") or REPOSITORY.parent / "rcmps-phi4-z2")
