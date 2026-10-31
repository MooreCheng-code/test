import io
from pathlib import Path
import sys
import unittest
from contextlib import redirect_stdout

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.check_code_docs import check_recipes


class TestRecipesChecker(unittest.TestCase):
    """Unit and negative tests for check_code_docs.py --recipes mode."""

    def setUp(self):
        self.test_dir = REPO_ROOT / "scratch" / "test_recipes"
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_positive_valid_recipe(self):
        """Positive test: Valid recipe with all 11 sections, valid citations, and invalidation details."""
        doc = """# 配方手冊

## 配方 1：新增一個模型（示範）

- 目標：示範新增模型
- 前置知識：閱讀 pipelines.py
- 風險等級：【高】會使固化產物或 E6 結果不可比
- 需要修改的位置：
  - `bankruptcy/config.py:28` (`TARGET_RECALLS`)
  - `bankruptcy/src/metrics.py:202` (`evaluate_scores`)
- 步驟：
  1. 步驟一
- 必補的單元測試：
  - test_m6
- 防洩漏檢查：
  - 獨立評估
- 驗證方式：
  - pytest
- 常見的坑：
  - 忘記實作 predict_proba
- 如何復原：
  - git restore
- 這個修改會讓哪些固化產物失效或需要重建：
  - `outputs/models/best_params/*.json` 需要重跑重建
"""
        test_file = self.test_dir / "valid_recipe.md"
        test_file.write_text(doc, encoding="utf-8")

        f = io.StringIO()
        with redirect_stdout(f):
            errs, warns = check_recipes(test_file)
        out = f.getvalue()

        self.assertEqual(errs, 0, f"Expected 0 errors, got {errs}. Output:\n{out}")
        self.assertIn("Total recipes verified: 1", out)

    def test_negative_missing_section(self):
        """Negative test 1: Missing required section header (e.g. missing '防洩漏檢查')."""
        doc = """# 配方手冊

## 配方 1：缺欄位配方

- 目標：示範缺少欄位
- 前置知識：無
- 風險等級：【低】
- 需要修改的位置：
  - `bankruptcy/config.py:28` (`TARGET_RECALLS`)
- 步驟：
  1. 步驟一
- 必補的單元測試：
  - test_abc
- 驗證方式：
  - pytest
- 常見的坑：
  - 坑一
- 如何復原：
  - git restore
- 這個修改會讓哪些固化產物失效或需要重建：
  - 無影響
"""
        test_file = self.test_dir / "missing_section.md"
        test_file.write_text(doc, encoding="utf-8")

        f = io.StringIO()
        with redirect_stdout(f):
            errs, warns = check_recipes(test_file)
        out = f.getvalue()

        self.assertGreater(errs, 0, "Expected errors for missing section header!")
        self.assertIn("missing required section header: '- 防洩漏檢查：'", out)
        print("\n[Negative Test 1 Passed] Caught missing section header:\n", out.strip())

    def test_negative_symbol_or_line_error(self):
        """Negative test 2: Line number out of bounds or symbol mismatch in '需要修改的位置'."""
        doc = """# 配方手冊

## 配方 1：符號錯誤配方

- 目標：示範符號錯誤
- 前置知識：無
- 風險等級：【低】
- 需要修改的位置：
  - `bankruptcy/config.py:9999` (`TARGET_RECALLS`)
  - `bankruptcy/src/metrics.py:202` (`NON_EXISTENT_FUNCTION_NAME`)
- 步驟：
  1. 步驟一
- 必補的單元測試：
  - test_abc
- 防洩漏檢查：
  - 無洩漏
- 驗證方式：
  - pytest
- 常見的坑：
  - 坑一
- 如何復原：
  - git restore
- 這個修改會讓哪些固化產物失效或需要重建：
  - 無影響
"""
        test_file = self.test_dir / "symbol_error.md"
        test_file.write_text(doc, encoding="utf-8")

        f = io.StringIO()
        with redirect_stdout(f):
            errs, warns = check_recipes(test_file)
        out = f.getvalue()

        self.assertGreater(errs, 0, "Expected errors for symbol or line out of bounds!")
        self.assertTrue("out of bounds" in out or "Symbol 'NON_EXISTENT_FUNCTION_NAME' not found" in out)
        print("\n[Negative Test 2 Passed] Caught line out of bounds and symbol mismatch:\n", out.strip())

    def test_negative_high_risk_missing_invalidation(self):
        """Negative test 3: High-risk recipe missing invalidation/rebuild details (empty or '無')."""
        doc = """# 配方手冊

## 配方 1：高風險缺失效說明

- 目標：高風險修改
- 前置知識：無
- 風險等級：【高】會使固化產物或 E6 結果不可比
- 需要修改的位置：
  - `bankruptcy/config.py:28` (`TARGET_RECALLS`)
- 步驟：
  1. 步驟一
- 必補的單元測試：
  - test_abc
- 防洩漏檢查：
  - 無洩漏
- 驗證方式：
  - pytest
- 常見的坑：
  - 坑一
- 如何復原：
  - git restore
- 這個修改會讓哪些固化產物失效或需要重建：
  - 無
"""
        test_file = self.test_dir / "high_risk_empty_inv.md"
        test_file.write_text(doc, encoding="utf-8")

        f = io.StringIO()
        with redirect_stdout(f):
            errs, warns = check_recipes(test_file)
        out = f.getvalue()

        self.assertGreater(errs, 0, "Expected errors for high-risk missing invalidation explanation!")
        self.assertIn("invalidation section must specify actual artifacts", out)
        print("\n[Negative Test 3 Passed] Caught high-risk missing invalidation details:\n", out.strip())

    def test_negative_margins_wrong_location_and_nonexistent_artifact(self):
        """Negative test 4: Catches config.py:30 (MARGINS) wrong location and nonexistent operating_thresholds.json."""
        doc = """# 配方手冊

## 配方 3：修改目標召回率與邊際（以新增 R=0.85 為例）

- 目標：在專案中新增 R=0.85 目標召回率
- 前置知識：閱讀門檻模組
- 風險等級：【高】會使固化產物或 E6 結果不可比
- 需要修改的位置：
  - `bankruptcy/config.py:30` (`MARGINS`，例如改為 (0.0, 0.02, 0.04))
- 步驟：
  1. 修改 config
- 必補的單元測試：
  - test_threshold
- 防洩漏檢查：
  - 獨立折內
- 驗證方式：
  - pytest
- 常見的坑：
  - 坑
- 如何復原：
  - git checkout
- 這個修改會讓哪些固化產物失效或需要重建：
  - `outputs/models/final/operating_thresholds.json` 需要重建
"""
        test_file = self.test_dir / "margins_negative_recipe.md"
        test_file.write_text(doc, encoding="utf-8")

        f = io.StringIO()
        with redirect_stdout(f):
            errs, warns = check_recipes(test_file)
        out = f.getvalue()

        self.assertGreater(errs, 0, "Expected errors for wrong symbol location and nonexistent artifact!")
        self.assertIn("Symbol 'MARGINS' not found in bankruptcy/config.py:30", out)
        self.assertIn("Invalidated artifact file not found: 'outputs/models/final/operating_thresholds.json'", out)
        print("\n[Negative Test 4 Passed] Caught config.py:30 (MARGINS) and operating_thresholds.json:\n", out.strip())


if __name__ == "__main__":
    unittest.main()

