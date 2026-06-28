"""Tests for the reusable contact-sheet / comparison-grid utility.

Comparison grids (rows = prompts/variants, cols = arms/methods) recur across the experiment
validators; this is the one tested implementation they should all call.
"""
from pathlib import Path

from PIL import Image

import pytest

from krea2_explorations.image_grid import build_contact_sheet, build_doc_figure


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


def test_title_adds_top_band_above_col_header(tmp_path):
    grid = [[_png(tmp_path / f"t{r}{c}.png") for c in range(2)] for r in range(2)]
    out = build_contact_sheet(grid, tmp_path / "g.png", col_labels=["a", "b"],
                              cell_px=100, label_px=20, title="prompt + settings", title_line_px=30)
    # 2 cols x 100 wide; title band 30 + col header 20 + 2 rows x 100 tall
    assert Image.open(out).size == (200, 30 + 20 + 200)


def test_multiline_title_band_scales_with_explicit_lines(tmp_path):
    grid = [[_png(tmp_path / "m.png")]]
    out = build_contact_sheet(grid, tmp_path / "g.png", cell_px=80,
                              title="line one\nline two\nline three", title_line_px=25)
    # no col/row labels; title 3 lines x 25 + 1 row x 80
    assert Image.open(out).size == (80, 75 + 80)


def test_doc_figure_requires_title_and_labels():
    # title / col_labels / row_labels are required kwargs: a committed figure can't omit its legend.
    grid = [[None, None], [None, None]]
    with pytest.raises(TypeError):
        build_doc_figure(grid, "x.png", col_labels=["a", "b"], row_labels=["p", "q"])  # no title
    with pytest.raises(TypeError):
        build_doc_figure(grid, "x.png", title="t", row_labels=["p", "q"])  # no col_labels
    with pytest.raises(TypeError):
        build_doc_figure(grid, "x.png", title="t", col_labels=["a", "b"])  # no row_labels


def test_doc_figure_rejects_blank_or_mismatched_labels(tmp_path):
    grid = [[_png(tmp_path / f"d{r}{c}.png") for c in range(2)] for r in range(2)]
    with pytest.raises(ValueError):  # blank title
        build_doc_figure(grid, tmp_path / "g.png", title="  ", col_labels=["a", "b"], row_labels=["p", "q"])
    with pytest.raises(ValueError):  # wrong number of col labels
        build_doc_figure(grid, tmp_path / "g.png", title="t", col_labels=["a"], row_labels=["p", "q"])
    with pytest.raises(ValueError):  # wrong number of row labels
        build_doc_figure(grid, tmp_path / "g.png", title="t", col_labels=["a", "b"], row_labels=["p"])


def test_doc_figure_builds_with_full_legend(tmp_path):
    grid = [[_png(tmp_path / f"f{r}{c}.png") for c in range(2)] for r in range(2)]
    out = build_doc_figure(grid, tmp_path / "g.png", title="prompt: 'a fox'\nsettings: 8 steps",
                           col_labels=["seed 42", "seed 123"], row_labels=["k=1", "k=2"],
                           cell_px=100, label_px=20, title_line_px=30)
    # title 2 lines x 30 + col header 20 + 2 rows x 100; row band default 160 + 2 cols x 100
    assert Image.open(out).size == (160 + 200, 60 + 20 + 200)
