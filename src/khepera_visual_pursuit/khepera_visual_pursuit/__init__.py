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

"""Hybrid visual pursuit and tracking module for Khepera IV mobile robot."""

from khepera_visual_pursuit.dinov2_verifier import DINOv2Verifier, VerificationResult
from khepera_visual_pursuit.ibvs_controller import IBVSController, IBVSGains, TwistCommand
from khepera_visual_pursuit.superpoint_lightglue_engine import (
    SuperPointLightGlueTracker,
    TrackingOutput,
)
from khepera_visual_pursuit.yolo_world_engine import BoundingBox, YoloWorldEngine

__all__ = [
    'BoundingBox',
    'DINOv2Verifier',
    'IBVSController',
    'IBVSGains',
    'SuperPointLightGlueTracker',
    'TrackingOutput',
    'TwistCommand',
    'VerificationResult',
    'YoloWorldEngine',
]
