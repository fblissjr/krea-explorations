"""Reusable comparison-grid / contact-sheet builder for experiment figures.

Rows = prompts/variants, columns = arms/methods. One tested implementation so the validators
(`internal/training/validate_*.py`) and any figure script call the same code instead of re-inlining
PIL. Depends only on Pillow + stdlib, so it imports cleanly from the isolated training venv too
(add `<repo>/src` to sys.path).

    from krea2_explorations.image_grid import build_contact_sheet
    build_contact_sheet(
        grid=[[img_paths_for_each_arm] for each prompt],
        out_path="grid.png",
        col_labels=["base", "without_projector", "with_projector"],
        row_labels=[short_prompt_0, short_prompt_1, ...],
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

from PIL import Image, ImageDraw, ImageFont

Cell = Union[str, Path, Image.Image, None]

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _load_font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, px)
        except Exception:
            continue
    return ImageFont.load_default()


def _open_cell(cell: Cell) -> Image.Image | None:
    if cell is None:
        return None
    if isinstance(cell, Image.Image):
        return cell.convert("RGB")
    try:
        return Image.open(cell).convert("RGB")
    except Exception:
        return None  # missing/unreadable -> placeholder, never crash a long run's figure


def _fit(img: Image.Image, box: int, mode: str) -> Image.Image:
    if mode == "stretch":
        return img.resize((box, box))
    # "contain": preserve aspect, letterbox onto a square
    w, h = img.size
    scale = min(box / w, box / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    canvas = Image.new("RGB", (box, box), (0, 0, 0))
    canvas.paste(img.resize((nw, nh)), ((box - nw) // 2, (box - nh) // 2))
    return canvas


def _truncate(draw, text: str, font, max_px: int) -> str:
    if draw.textlength(text, font=font) <= max_px:
        return text
    ell = "…"
    while text and draw.textlength(text + ell, font=font) > max_px:
        text = text[:-1]
    return text + ell


def build_contact_sheet(
    grid: Sequence[Sequence[Cell]],
    out_path: Union[str, Path],
    *,
    col_labels: Sequence[str] | None = None,
    row_labels: Sequence[str] | None = None,
    title: str | None = None,
    cell_px: int = 384,
    label_px: int = 24,
    row_label_w: int = 160,
    title_line_px: int = 26,
    font_px: int = 14,
    title_font_px: int = 16,
    fit: str = "contain",
    bg: tuple[int, int, int] = (20, 20, 20),
    label_bg: tuple[int, int, int] = (40, 40, 40),
    label_fg: tuple[int, int, int] = (235, 235, 235),
    missing_fill: tuple[int, int, int] = (60, 30, 30),
) -> Path:
    """Compose ``grid`` (rows x cols of image path / PIL.Image / None) into a labeled contact sheet.

    Layout: an optional full-width ``title`` band at the very top (one row of height ``title_line_px`` per
    explicit ``\\n``-separated line — put the prompt + fixed settings here), then an optional column-label
    header band (height ``label_px``), then an optional row-label left band (width ``row_label_w``). Ragged
    rows are padded to the widest row; ``None``/unreadable cells render as a ``missing_fill`` placeholder.
    Returns the written ``Path``. For figures committed under ``docs/``, prefer :func:`build_doc_figure`,
    which makes the title + axis labels mandatory.
    """
    if not grid:
        raise ValueError("grid must have at least one row")
    n_rows = len(grid)
    n_cols = max(len(r) for r in grid)
    title_lines = title.split("\n") if title else []
    title_h = len(title_lines) * title_line_px
    top_h = label_px if col_labels else 0
    left_w = row_label_w if row_labels else 0

    W = left_w + n_cols * cell_px
    H = title_h + top_h + n_rows * cell_px
    sheet = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(sheet)
    font = _load_font(font_px)

    if title_lines:
        draw.rectangle([0, 0, W, title_h], fill=label_bg)
        tfont = _load_font(title_font_px)
        for i, line in enumerate(title_lines):
            draw.text((6, i * title_line_px + 4), _truncate(draw, line, tfont, W - 12),
                      fill=label_fg, font=tfont)

    if col_labels:
        for c, lab in enumerate(col_labels[:n_cols]):
            x = left_w + c * cell_px
            draw.rectangle([x, title_h, x + cell_px, title_h + top_h], fill=label_bg)
            draw.text((x + 4, title_h + 4), _truncate(draw, str(lab), font, cell_px - 8),
                      fill=label_fg, font=font)

    if row_labels:
        for r, lab in enumerate(row_labels[:n_rows]):
            y = title_h + top_h + r * cell_px
            draw.rectangle([0, y, left_w, y + cell_px], fill=label_bg)
            draw.text((4, y + 4), _truncate(draw, str(lab), font, left_w - 8),
                      fill=label_fg, font=font)

    for r in range(n_rows):
        row = grid[r]
        for c in range(n_cols):
            x = left_w + c * cell_px
            y = title_h + top_h + r * cell_px
            cell = row[c] if c < len(row) else None
            img = _open_cell(cell)
            if img is None:
                draw.rectangle([x, y, x + cell_px, y + cell_px], fill=missing_fill)
            else:
                sheet.paste(_fit(img, cell_px, fit), (x, y))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def build_doc_figure(
    grid: Sequence[Sequence[Cell]],
    out_path: Union[str, Path],
    *,
    title: str,
    col_labels: Sequence[str],
    row_labels: Sequence[str],
    **kwargs,
) -> Path:
    """Contact sheet for a figure committed under ``docs/`` — a self-explaining legend is **mandatory**.

    Unlike :func:`build_contact_sheet`, ``title``, ``col_labels`` and ``row_labels`` are required (omitting
    any is a ``TypeError``), and they are validated: the title must be non-blank (put the prompt + the fixed
    settings there, one fact per ``\\n`` line), and there must be exactly one label per column and per row
    (so no reader has to guess what an axis value means). Everything else forwards to
    :func:`build_contact_sheet`. Use this for any figure that ships in the repo.
    """
    if not title or not title.strip():
        raise ValueError("doc figures need a non-blank title (prompt + settings)")
    n_rows = len(grid)
    n_cols = max(len(r) for r in grid) if grid else 0
    if len(col_labels) != n_cols:
        raise ValueError(f"need one col label per column: {len(col_labels)} labels vs {n_cols} columns")
    if len(row_labels) != n_rows:
        raise ValueError(f"need one row label per row: {len(row_labels)} labels vs {n_rows} rows")
    return build_contact_sheet(grid, out_path, title=title, col_labels=col_labels,
                               row_labels=row_labels, **kwargs)
