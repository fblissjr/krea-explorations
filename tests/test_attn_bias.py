"""Logic tests for the EXP5 Phase-D attention-bias node's pure parse helpers (block/position specs)."""

from krea2_explorations.krea2_attn_bias_node import parse_blocks, parse_positions


def test_parse_blocks_range_and_list_and_clamp():
    assert parse_blocks("5-10", 28) == [5, 6, 7, 8, 9, 10]
    assert parse_blocks("5,6,7", 28) == [5, 6, 7]
    assert parse_blocks("5-7,18-20", 28) == [5, 6, 7, 18, 19, 20]
    assert parse_blocks("25-40", 28) == [25, 26, 27]  # clamped to [0, n)
    assert parse_blocks(" 5 - 6 ", 28) == [5, 6]  # whitespace tolerant


def test_parse_positions():
    assert parse_positions("41,42,46") == [41, 42, 46]
    assert parse_positions("") == []
    assert parse_positions(" 1, 2 ,3 ") == [1, 2, 3]
