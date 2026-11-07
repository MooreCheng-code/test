import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.check_code_docs import check_glossary, GLOSSARY_MD


class TestGlossaryChecker(unittest.TestCase):
    def setUp(self):
        self.scratch_dir = REPO_ROOT / "scratch" / "test_glossary"
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

    def test_real_glossary_passes(self):
        """Positive test: Verifies the official 08_術語表.md has 0 errors and >= 40 terms."""
        errors, warnings = check_glossary(GLOSSARY_MD)
        self.assertEqual(errors, 0, f"Expected 0 errors in official glossary, got {errors}")
        self.assertEqual(warnings, 0, f"Expected 0 warnings in official glossary, got {warnings}")

    def test_negative_missing_field(self):
        """Negative test 1: Catches missing required field (e.g. 最小數字例子)."""
        bad_md = self.scratch_dir / "missing_field.md"
        content = """# 術語表
### 1. ROC-AUC (Area Under Curve)
- **白話定義**：衡量排序指標。
- **專案中的位置**：`bankruptcy/src/metrics.py:43` (`roc_auc_score`)
- **第一次講解的導讀或參數卡**：[評估指標](../../docs/code_learning/02_模組導讀/metrics.md)
"""
        bad_md.write_text(content, encoding="utf-8")
        errors, _ = check_glossary(bad_md)
        self.assertGreater(errors, 0, "Checker should fail on missing field '最小數字例子'")

    def test_negative_symbol_mismatch(self):
        """Negative test 2: Catches symbol not found in file or line out of bounds."""
        bad_md = self.scratch_dir / "symbol_mismatch.md"
        content = """# 術語表
### 1. ROC-AUC (Area Under Curve)
- **白話定義**：衡量排序指標。
- **最小數字例子**：20 家玩具公司測試。
- **專案中的位置**：`bankruptcy/src/metrics.py:43` (`NON_EXISTENT_SYMBOL_NAME`)
- **第一次講解的導讀或參數卡**：[評估指標](../../docs/code_learning/02_模組導讀/metrics.md)
"""
        bad_md.write_text(content, encoding="utf-8")
        errors, _ = check_glossary(bad_md)
        self.assertGreater(errors, 0, "Checker should fail on non-existent symbol")

    def test_negative_broken_link(self):
        """Negative test 3: Catches linked doc file that does not exist."""
        bad_md = self.scratch_dir / "broken_link.md"
        content = """# 術語表
### 1. ROC-AUC (Area Under Curve)
- **白話定義**：衡量排序指標。
- **最小數字例子**：20 家玩具公司測試。
- **專案中的位置**：`bankruptcy/src/metrics.py:43` (`roc_auc_score`)
- **第一次講解的導讀或參數卡**：[評估指標](../../docs/code_learning/non_existent_doc_123.md)
"""
        bad_md.write_text(content, encoding="utf-8")
        errors, _ = check_glossary(bad_md)
        self.assertGreater(errors, 0, "Checker should fail on broken document link")

    def test_negative_ar_definition_violation(self):
        """Negative test 4: Catches AR definition using '比例' instead of '預警家數 ÷ 破產家數'."""
        bad_md = self.scratch_dir / "ar_violation.md"
        content = """# 術語表
### 1. 預警倍數 (Alert Ratio)
- **白話定義**：模型發出預警的樣本比例。
- **最小數字例子**：20 家玩具公司測試。
- **專案中的位置**：`bankruptcy/src/metrics.py:219` (`alert_ratio`)
- **第一次講解的導讀或參數卡**：[評估指標](../../docs/code_learning/02_模組導讀/metrics.md)
"""
        bad_md.write_text(content, encoding="utf-8")
        errors, _ = check_glossary(bad_md)
        self.assertGreater(errors, 0, "Checker should fail on AR definition with '比例'")

    def test_negative_duplicate_term(self):
        """Negative test 5: Catches duplicate term definitions."""
        bad_md = self.scratch_dir / "duplicate_term.md"
        content = """# 術語表
### 1. ROC-AUC (Area Under Curve)
- **白話定義**：衡量排序指標。
- **最小數字例子**：20 家玩具公司測試。
- **專案中的位置**：`bankruptcy/src/metrics.py:43` (`roc_auc_score`)
- **第一次講解的導讀或參數卡**：[評估指標](../../docs/code_learning/02_模組導讀/metrics.md)

### 2. ROC-AUC (Area Under Curve)
- **白話定義**：衡量排序指標重複項。
- **最小數字例子**：20 家玩具公司測試。
- **專案中的位置**：`bankruptcy/src/metrics.py:43` (`roc_auc_score`)
- **第一次講解的導讀或參數卡**：[評估指標](../../docs/code_learning/02_模組導讀/metrics.md)
"""
        bad_md.write_text(content, encoding="utf-8")
        errors, _ = check_glossary(bad_md)
        self.assertGreater(errors, 0, "Checker should fail on duplicate term")


if __name__ == "__main__":
    unittest.main()
