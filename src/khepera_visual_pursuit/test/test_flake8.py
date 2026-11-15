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

"""Test suite verifying flake8 code style standards."""

import os

import pytest

try:
    from ament_flake8.main import main_with_errors

    HAVE_AMENT_FLAKE8 = True
except ImportError:
    HAVE_AMENT_FLAKE8 = False
    import flake8.api.legacy as flake8


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8() -> None:
    """Validate PEP 8 code formatting using flake8."""
    pkg_dir = os.path.dirname(os.path.dirname(__file__))
    config_file = os.path.join(pkg_dir, "setup.cfg")

    if HAVE_AMENT_FLAKE8:
        rc, errors = main_with_errors(argv=[pkg_dir, "--config", config_file])
        assert rc == 0, f"Found {len(errors)} code style errors:\n" + "\n".join(errors)
    else:
        style_guide = flake8.get_style_guide(
            config_file=config_file, max_line_length=100
        )
        report = style_guide.check_files([pkg_dir])
        assert report.total_errors == 0, f"Found {report.total_errors} flake8 errors."
