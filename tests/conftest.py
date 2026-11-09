"""Ensures the project root is importable when running pytest from anywhere.

This lives in tests/ rather than beside pytest.ini because pytest loads every
conftest.py on the path from the rootdir down to each test file. A conftest.py
one level up, at the repository root, would cover the tests too - but it would
also be a stray .py file in the root, and `python -m pytest` already happens to
put the working directory on sys.path, which masks the difference. The bare
`pytest` command does not, and this file is what keeps that command working.
"""

import os
import sys

# The repository root, one level above this file.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
