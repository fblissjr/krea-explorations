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
    cell_px: int = 384,
    label_px: int = 24,
    row_label_w: int = 160,
    font_px: int = 14,
    fit: str = "contain",
    bg: tuple[int, int, int] = (20, 20, 20),
    label_bg: tuple[int, int, int] = (40, 40, 40),
    label_fg: tuple[int, int, int] = (235, 235, 235),
    missing_fill: tuple[int, int, int] = (60, 30, 30),
) -> Path:
    """Compose ``grid`` (rows x cols of image path / PIL.Image / None) into a labeled contact sheet.

    Layout: optional column-label header band (height ``label_px``) and optional row-label left band
    (width ``row_label_w``). Ragged rows are padded to the widest row; ``None``/unreadable cells render
    as a ``missing_fill`` placeholder. Returns the written ``Path``.
    """
    if not grid:
        raise ValueError("grid must have at least one row")
    n_rows = len(grid)
    n_cols = max(len(r) for r in grid)
    top_h = label_px if col_labels else 0
    left_w = row_label_w if row_labels else 0

    W = left_w + n_cols * cell_px
    H = top_h + n_rows * cell_px
    sheet = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(sheet)
    font = _load_font(font_px)

    if col_labels:
        for c, lab in enumerate(col_labels[:n_cols]):
            x = left_w + c * cell_px
            draw.rectangle([x, 0, x + cell_px, top_h], fill=label_bg)
            draw.text((x + 4, 4), _truncate(draw, str(lab), font, cell_px - 8),
                      fill=label_fg, font=font)

    if row_labels:
        for r, lab in enumerate(row_labels[:n_rows]):
            y = top_h + r * cell_px
            draw.rectangle([0, y, left_w, y + cell_px], fill=label_bg)
            draw.text((4, y + 4), _truncate(draw, str(lab), font, left_w - 8),
                      fill=label_fg, font=font)

    for r in range(n_rows):
        row = grid[r]
        for c in range(n_cols):
            x = left_w + c * cell_px
            y = top_h + r * cell_px
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
