"""Run deterministic polling races against the actual page JavaScript."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_polling_races():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js is required for the frontend regression tests')
    subprocess.run([node, '--test', str(Path(__file__).with_name('frontend.test.cjs'))],
                   check=True, capture_output=True, text=True)
