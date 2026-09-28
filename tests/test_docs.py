from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYTHON_FENCE = re.compile(r"(?ms)^```python[ \t]*\n(.*?)^```[ \t]*$")


def python_examples() -> list[tuple[Path, int, str]]:
    paths = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    examples = []
    for path in paths:
        contents = path.read_text(encoding="utf-8")
        for match in PYTHON_FENCE.finditer(contents):
            line = contents.count("\n", 0, match.start()) + 1
            examples.append((path, line, match.group(1)))
    return examples


@pytest.mark.parametrize(
    ("path", "line", "code"),
    [
        pytest.param(path, line, code, id=f"{path.relative_to(ROOT)}:{line}")
        for path, line, code in python_examples()
    ],
)
def test_documented_python_is_ruff_formatted(path: Path, line: int, code: str) -> None:
    result = subprocess.run(
        ["ruff", "format", "--check", "--stdin-filename", "doc_example.py", "-"],
        input=code,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"{path.relative_to(ROOT)}:{line} has unformatted Python:\n{result.stdout}{result.stderr}"
    )
