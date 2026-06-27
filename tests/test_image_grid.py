"""Tests for the reusable contact-sheet / comparison-grid utility.

Comparison grids (rows = prompts/variants, cols = arms/methods) recur across the experiment
validators; this is the one tested implementation they should all call.
"""
from pathlib import Path

from PIL import Image

from krea2_explorations.image_grid import build_contact_sheet


def _png(path: Path, size=(32, 24), color=(10, 120, 200)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def test_dims_with_col_labels_no_row_labels(tmp_path):
    grid = [[_png(tmp_path / f"c{r}{c}.png") for c in range(3)] for r in range(2)]
    out = build_contact_sheet(grid, tmp_path / "g.png", col_labels=["a", "b", "c"],
                              cell_px=100, label_px=20)
    assert out.exists()
    # 3 cols x 100 wide; 2 rows x 100 tall + 20 header band
    assert Image.open(out).size == (300, 220)


def test_row_labels_add_left_band(tmp_path):
    grid = [[_png(tmp_path / f"r{r}{c}.png") for c in range(2)] for r in range(2)]
    out = build_contact_sheet(grid, tmp_path / "g.png",
                              col_labels=["x", "y"], row_labels=["p0", "p1"],
                              cell_px=100, label_px=20, row_label_w=150)
    assert Image.open(out).size == (150 + 200, 20 + 200)


def test_no_labels_is_bare_grid(tmp_path):
    grid = [[_png(tmp_path / f"n{r}{c}.png") for c in range(2)] for r in range(3)]
    out = build_contact_sheet(grid, tmp_path / "g.png", cell_px=64)
    assert Image.open(out).size == (128, 192)


def test_none_and_missing_cells_do_not_crash(tmp_path):
    grid = [
        [_png(tmp_path / "ok.png"), None],
        [tmp_path / "does_not_exist.png"],  # ragged + missing path
    ]
    out = build_contact_sheet(grid, tmp_path / "g.png", cell_px=50)
    # 2 cols (max width), 2 rows
    assert Image.open(out).size == (100, 100)


def test_accepts_pil_images_and_paths(tmp_path):
    grid = [[Image.new("RGB", (8, 8), (0, 0, 0)), str(_png(tmp_path / "p.png"))]]
    out = build_contact_sheet(grid, tmp_path / "g.png", cell_px=40)
    assert isinstance(out, Path) and out.exists()
    assert Image.open(out).size == (80, 40)
