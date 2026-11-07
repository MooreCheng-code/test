import io
from pathlib import Path
import sys
import unittest
from contextlib import redirect_stdout

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.check_code_docs import check_exercises


class TestExercisesChecker(unittest.TestCase):
    """Unit and negative tests for check_code_docs.py --exercises mode."""

    def setUp(self):
        self.test_dir = REPO_ROOT / "scratch" / "test_exercises"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.bugs_dir = self.test_dir / "bugs"
        self.bugs_dir.mkdir(parents=True, exist_ok=True)

        self.valid_header = (
            "# ============================================================\n"
            "# 教學用「含錯誤」練習腳本：請勿當作專案寫法使用\n"
            "# 內含一個刻意植入的錯誤，請找出來。解答見 docs/code_learning/06_解答.md\n"
            "# 只使用合成資料，不讀取專案資料。\n"
            "# ============================================================\n"
        )

    def tearDown(self):
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_positive_official_exercises(self):
        """Positive test: Real repo exercises and bug scripts pass with 0 errors."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            errors, warnings = check_exercises()
        out = buf.getvalue()
        self.assertEqual(errors, 0, f"Expected 0 errors on official exercises, got: {out}")
        self.assertIn("Exercises check completed: 0 error(s)", out)

    def test_negative_forbidden_answer_word(self):
        """Negative test: Exercises doc containing forbidden answer keyword ('答案') fails."""
        bad_md = self.test_dir / "bad_ex.md"
        bad_md.write_text("# 練習題\n\n本題標準答案為選項 A。\n", encoding="utf-8")

        buf = io.StringIO()
        with redirect_stdout(buf):
            errors, _ = check_exercises(exercises_path=bad_md, bugs_dir=self.bugs_dir)
        out = buf.getvalue()
        self.assertGreater(errors, 0)
        self.assertIn("Forbidden answer keyword '答案'", out)

    def test_negative_missing_link_target(self):
        """Negative test: Exercises doc referencing non-existent file link fails."""
        bad_md = self.test_dir / "bad_link.md"
        bad_md.write_text("# 練習題\n\n- [`non_existent.py`](non_existent.py)\n", encoding="utf-8")

        buf = io.StringIO()
        with redirect_stdout(buf):
            errors, _ = check_exercises(exercises_path=bad_md, bugs_dir=self.bugs_dir)
        out = buf.getvalue()
        self.assertGreater(errors, 0)
        self.assertIn("Cited link target does not exist", out)

    def test_negative_incomplete_lab_status(self):
        """Negative test: Predictive question citing a lab marked '未完成' fails."""
        bad_md = self.test_dir / "bad_lab.md"
        bad_md.write_text(
            "# 練習題\n\n## 2. 預測題\n\n- 對應實驗：lab_01\n\n## 3. 除錯題\n",
            encoding="utf-8"
        )
        fake_lab_map = self.test_dir / "fake_lab_map.csv"
        fake_lab_map.write_text(
            "lab_id,名稱,觀察參數組,狀態\nlab_01,單決策樹,max_depth,未完成\n",
            encoding="utf-8-sig"
        )

        buf = io.StringIO()
        with redirect_stdout(buf):
            errors, _ = check_exercises(
                exercises_path=bad_md,
                bugs_dir=self.bugs_dir,
                lab_map_file=fake_lab_map
            )
        out = buf.getvalue()
        self.assertGreater(errors, 0)
        self.assertIn("with incomplete status: '未完成'", out)

    def test_negative_bug_script_missing_header(self):
        """Negative test: Bug script missing required safety header fails."""
        bad_py = self.bugs_dir / "bug_no_header.py"
        bad_py.write_text("import numpy as np\nprint('hello')\n", encoding="utf-8")

        buf = io.StringIO()
        with redirect_stdout(buf):
            errors, _ = check_exercises(bugs_dir=self.bugs_dir)
        out = buf.getvalue()
        self.assertGreater(errors, 0)
        self.assertIn("Missing or invalid required safety header", out)

    def test_negative_bug_script_forbidden_data_access(self):
        """Negative test: Bug script referencing data.csv or get_train_data fails."""
        bad_py = self.bugs_dir / "bug_data_leak.py"
        bad_py.write_text(self.valid_header + "\nimport pandas as pd\ndf = pd.read_csv('data.csv')\n", encoding="utf-8")

        buf = io.StringIO()
        with redirect_stdout(buf):
            errors, _ = check_exercises(bugs_dir=self.bugs_dir)
        out = buf.getvalue()
        self.assertGreater(errors, 0)
        self.assertIn("Forbidden token 'data.csv'", out)

    def test_negative_bug_script_leak_comment(self):
        """Negative test: Bug script containing answer-revealing comment (# BUG) fails."""
        bad_py = self.bugs_dir / "bug_comment.py"
        bad_py.write_text(self.valid_header + "\n# BUG: this is the error line\nx = 1\n", encoding="utf-8")

        buf = io.StringIO()
        with redirect_stdout(buf):
            errors, _ = check_exercises(bugs_dir=self.bugs_dir)
        out = buf.getvalue()
        self.assertGreater(errors, 0)
        self.assertIn("Answer leak comment detected", out)


if __name__ == "__main__":
    unittest.main()
