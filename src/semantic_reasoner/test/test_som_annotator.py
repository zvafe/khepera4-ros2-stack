"""Unit tests for Set-of-Marks visual annotator."""

import numpy as np
from semantic_reasoner.som_annotator import SetOfMarksAnnotator


def test_som_annotator_basic() -> None:
    """Verify that markers are drawn on image without errors."""
    annotator = SetOfMarksAnnotator(circle_radius=12)
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    markers = [("A", (100, 150)), ("B", (300, 250))]
    result = annotator.annotate(img, markers)

    assert result.shape == img.shape
    # Check that pixel values at marker location changed from pure black
    assert not np.array_equal(result[150, 100], [0, 0, 0])
    assert not np.array_equal(result[250, 300], [0, 0, 0])


def test_som_annotator_boundary_clamping() -> None:
    """Verify that markers outside or at the border of image are clamped safely."""
    annotator = SetOfMarksAnnotator(circle_radius=15)
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Edge and out-of-bound coords
    markers = [("C", (-10, -5)), ("D", (700, 500))]
    result = annotator.annotate(img, markers)

    assert result.shape == img.shape
    assert result is not None
