# Copyright 2026 Andrei Berteanu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test suite verifying PEP 257 docstring compliance."""

import os

import pytest

try:
    from ament_pep257.main import main

    HAVE_AMENT_PEP257 = True
except ImportError:
    HAVE_AMENT_PEP257 = False
    import pydocstyle


@pytest.mark.linter
@pytest.mark.pep257
def test_pep257() -> None:
    """Validate docstrings across package python files."""
    pkg_dir = os.path.dirname(os.path.dirname(__file__))

    if HAVE_AMENT_PEP257:
        rc = main(argv=[pkg_dir])
        assert rc == 0, "Found PEP 257 code style errors / warnings"
    else:
        py_files = [
            os.path.join(root, f)
            for root, _, files in os.walk(pkg_dir)
            for f in files
            if f.endswith(".py")
        ]
        errors = list(pydocstyle.check(py_files))
        assert len(errors) == 0, f"Found {len(errors)} PEP 257 errors: {errors}"
