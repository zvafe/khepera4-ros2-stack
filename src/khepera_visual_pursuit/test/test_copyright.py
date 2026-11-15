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

"""Test suite verifying copyright headers."""

import os

import pytest

try:
    from ament_copyright.main import main

    HAVE_AMENT_COPYRIGHT = True
except ImportError:
    HAVE_AMENT_COPYRIGHT = False


@pytest.mark.skipif(not HAVE_AMENT_COPYRIGHT, reason="ament_copyright not available")
@pytest.mark.copyright
@pytest.mark.linter
def test_copyright() -> None:
    """Validate license and copyright headers."""
    pkg_dir = os.path.dirname(os.path.dirname(__file__))
    rc = main(argv=[pkg_dir])
    assert rc == 0, "Found copyright errors"
