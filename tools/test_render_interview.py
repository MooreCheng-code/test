"""
tools/test_render_interview.py
Unit tests for render_interview.py covering:
1. Positive test: successful placeholder interpolation (CSV and calc).
2. Positive test: JSON placeholder interpolation.
3. Negative test 1: missing column in CSV.
4. Negative test 2: where query selects multiple rows.
5. Negative test 3: bare metric numbers in template without placeholder.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from tools.render_interview import render_template, check_bare_metric_numbers

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_positive_interpolation():
    template = """# Test Title
R90 預警家數為 {{csv:bankruptcy/outputs/e6/e6_operating_points.csv|group=='E_A' and operating_point=='R90'|alerts_count|%d}} 家，
AR 為 {{csv:bankruptcy/outputs/e6/e6_operating_points.csv|group=='E_A' and operating_point=='R90'|alert_ratio|%.2f}}。
預警數與命中數比值約 {{calc:alerts_count / positives_caught|csv:bankruptcy/outputs/e6/e6_operating_points.csv|group=='E_A' and operating_point=='R90'|%.2f}}。
"""
    rendered, ledger = render_template(template, REPO_ROOT)
    assert "R90 預警家數為 337 家" in rendered
    assert "AR 為 7.66" in rendered
    assert "預警數與命中數比值約 7.84" in rendered
    assert len(ledger) == 3


def test_positive_json_interpolation():
    template = """
M5 不平衡策略為 {{json:bankruptcy/outputs/models/best_params/M5.json|imbalance|%s}}。
"""
    rendered, ledger = render_template(template, REPO_ROOT)
    assert "M5 不平衡策略為 S3。" in rendered
    assert len(ledger) == 1


def test_negative_missing_column():
    template = """
數值為 {{csv:bankruptcy/outputs/e6/e6_operating_points.csv|group=='E_A' and operating_point=='R90'|nonexistent_column|%d}}
"""
    with pytest.raises(KeyError, match="nonexistent_column"):
        render_template(template, REPO_ROOT)


def test_negative_multiple_rows_selected():
    template = """
數值為 {{csv:bankruptcy/outputs/tables/e4a_threshold_transfer_summary.csv|group=='E_A' and target_recall==0.9|attainment_rate|%.2f}}
"""
    with pytest.raises(ValueError, match="expected exactly 1"):
        render_template(template, REPO_ROOT)


def test_negative_bare_metric_number():
    template = """
本專案在測試集上達成 12.8% 的精準度。
"""
    unallowed = check_bare_metric_numbers(template)
    assert len(unallowed) > 0
    assert any("12.8%" in u for u in unallowed)
    with pytest.raises(ValueError, match="Bare metric numbers detected"):
        render_template(template, REPO_ROOT)
