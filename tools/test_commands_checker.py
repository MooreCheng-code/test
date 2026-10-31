"""tools/test_commands_checker.py - Unit tests and negative tests for check_code_docs.py --commands mode."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.check_code_docs import check_commands_in_docs, extract_script_arguments, validate_single_command

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestCommandsChecker(unittest.TestCase):
    """Test suite for CLI command validation and argument AST extraction."""

    def test_extract_script_arguments_real_scripts(self) -> None:
        """Verify AST extraction of arguments from real scripts."""
        tune_flags = extract_script_arguments(REPO_ROOT / "bankruptcy" / "src" / "tune.py")
        self.assertIn("--quick", tune_flags)
        self.assertIn("--outer-repeats", tune_flags)
        self.assertIn("--run-e2b", tune_flags)

        scoring_flags = extract_script_arguments(REPO_ROOT / "bankruptcy" / "score_new_data.py")
        self.assertIn("--input", scoring_flags)
        self.assertIn("--output", scoring_flags)
        self.assertIn("--operating-point", scoring_flags)
        self.assertIn("--sort-by-risk", scoring_flags)

    def test_validate_single_command_valid_cases(self) -> None:
        """Verify valid commands pass inspection."""
        valid_cmds = [
            "python bankruptcy/src/tune.py --quick --run-e2b",
            "python -m bankruptcy.src.data_audit --data bankruptcy/data/data.csv",
            "python bankruptcy/src/run_baselines.py --quick",
            "python bankruptcy/src/preflight.py",
            "python bankruptcy/score_new_data.py --input test.csv --output out.csv --operating-point R90",
            "streamlit run bankruptcy/app/app.py",
            "uvicorn bankruptcy.api.main:app",
        ]
        for cmd in valid_cmds:
            ok, msg = validate_single_command(cmd, REPO_ROOT)
            self.assertTrue(ok, f"Command failed: {cmd} ({msg})")

    def test_negative_missing_script(self) -> None:
        """Negative test: Command referencing non-existent script must fail."""
        ok, msg = validate_single_command("python bankruptcy/src/fake_non_existent.py --quick", REPO_ROOT)
        self.assertFalse(ok)
        self.assertIn("does not exist", msg)

    def test_negative_unknown_flag_on_argparse_script(self) -> None:
        """Negative test: Invented argument flag must fail validation."""
        ok, msg = validate_single_command("python bankruptcy/src/tune.py --invented-fake-flag", REPO_ROOT)
        self.assertFalse(ok)
        self.assertIn("Unknown argument flag '--invented-fake-flag'", msg)

    def test_negative_flag_on_no_argparse_script(self) -> None:
        """Negative test: Passing argument to script without argparse must fail."""
        ok, msg = validate_single_command("python bankruptcy/src/preflight.py --quick", REPO_ROOT)
        self.assertFalse(ok)
        self.assertIn("Unknown argument flag '--quick'", msg)

    def test_negative_markdown_file_scan(self) -> None:
        """Negative test: Markdown file containing illegal commands must trigger error exit."""
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("""# Test Guide
```powershell
python bankruptcy/src/tune.py --fake-param
```
""")
            temp_path = Path(f.name)

        try:
            errors, warnings = check_commands_in_docs(temp_path)
            self.assertEqual(errors, 1)
        finally:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
