"""Unit tests for SuperSplat binary PLY exporter."""

import os
import tempfile
import numpy as np
from plyfile import PlyData
import pytest

from khepera_3dgs.ply_exporter import SuperSplatPlyExporter


def test_ply_header_and_structure():
    """Verify PLY header properties match Inria / SuperSplat specification."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_ply = os.path.join(tmp_dir, "test_splat.ply")

        n = 50
        means = np.random.randn(n, 3).astype(np.float32)
        scales = np.random.uniform(0.01, 0.05, (n, 3)).astype(np.float32)
        quats = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32)
        opacities = np.full((n, 1), 0.8, dtype=np.float32)
        sh_dc = np.random.uniform(0.0, 1.0, (n, 3)).astype(np.float32)

        file_path = SuperSplatPlyExporter.export_ply(
            output_path=out_ply,
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh_dc=sh_dc,
            is_log_scale=False,  # Should convert to log
            is_logit_opacity=False,  # Should convert to logit
        )

        assert os.path.exists(file_path)
        assert os.path.getsize(file_path) > 0

        # Read back with plyfile to verify schema
        ply_data = PlyData.read(file_path)
        assert "vertex" in ply_data
        vertex = ply_data["vertex"]
        assert len(vertex) == n

        expected_props = [
            "x", "y", "z",
            "nx", "ny", "nz",
            "f_dc_0", "f_dc_1", "f_dc_2",
            "opacity",
            "scale_0", "scale_1", "scale_2",
            "rot_0", "rot_1", "rot_2", "rot_3",
        ]
        prop_names = [p.name for p in vertex.properties]
        assert prop_names == expected_props

        # Check values
        assert np.allclose(vertex["x"], means[:, 0])
        assert np.allclose(vertex["y"], means[:, 1])
        assert np.allclose(vertex["z"], means[:, 2])

        # Verify normal components are 0.0
        assert np.all(vertex["nx"] == 0.0)
        assert np.all(vertex["ny"] == 0.0)
        assert np.all(vertex["nz"] == 0.0)

        # Verify quaternion rot_0 is normalized (approx 1.0)
        assert np.allclose(vertex["rot_0"], 1.0, atol=1e-4)


def test_empty_points_error():
    """Verify exporting 0 points raises ValueError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_ply = os.path.join(tmp_dir, "empty.ply")
        with pytest.raises(ValueError):
            SuperSplatPlyExporter.export_ply(
                output_path=out_ply,
                means=np.empty((0, 3)),
                scales=np.empty((0, 3)),
                quats=np.empty((0, 4)),
                opacities=np.empty((0, 1)),
                sh_dc=np.empty((0, 3)),
            )


def test_supersplat_y_up_alignment():
    """Verify conversion from ROS Z-up to SuperSplat Y-up."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_ply = os.path.join(tmp_dir, "aligned.ply")
        means = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        scales = np.array([[0.05, 0.05, 0.05]], dtype=np.float32)
        quats = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        opacities = np.array([[0.8]], dtype=np.float32)
        sh_dc = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)

        SuperSplatPlyExporter.export_ply(
            output_path=out_ply,
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh_dc=sh_dc,
            align_supersplat_y_up=True,
        )

        ply_data = PlyData.read(out_ply)
        v = ply_data["vertex"]
        # In pre-rotated PLY coordinates: X_ply = -1.0, Y_ply = -3.0, Z_ply = -2.0
        assert np.isclose(v["x"][0], -1.0, atol=1e-4)
        assert np.isclose(v["y"][0], -3.0, atol=1e-4)
        assert np.isclose(v["z"][0], -2.0, atol=1e-4)

        # Upon import, SuperSplat automatically applies [0, 0, 180] Euler rotation.
        # Confirm that after SuperSplat default rotation, the scene is upright:
        # X_view = 1.0 (X_ros), Y_view = 3.0 (Z_ros height / Up), Z_view = -2.0 (-Y_ros forward)
        from scipy.spatial.transform import Rotation as ScipyRotation
        r_ss = ScipyRotation.from_euler("z", 180, degrees=True)
        v_ss = r_ss.apply(np.array([v["x"][0], v["y"][0], v["z"][0]]))
        assert np.isclose(v_ss[0], 1.0, atol=1e-4)
        assert np.isclose(v_ss[1], 3.0, atol=1e-4)
        assert np.isclose(v_ss[2], -2.0, atol=1e-4)


