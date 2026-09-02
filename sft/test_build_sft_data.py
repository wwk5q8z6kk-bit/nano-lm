import pytest
from build_sft_data import symbol_ratio

def test_symbol_ratio_empty():
    assert symbol_ratio("") == 0.0

def test_symbol_ratio_no_symbols():
    assert symbol_ratio("hello world") == 0.0

def test_symbol_ratio_all_symbols():
    assert symbol_ratio("\\${}#`^_") == 1.0

def test_symbol_ratio_mixed():
    assert symbol_ratio("hello $world") == 1.0 / 12.0

def test_symbol_ratio_different_symbols():
    # 4 matching symbols: {, }, \, #
    assert symbol_ratio("x = {a, b} \\ y # test") == 4.0 / 21.0
