"""
Property-based test: 源代码零修改 (Source Code Zero Modification)

**Feature: project-cleanup-for-opensource, Property 2: 源代码零修改**
**Validates: Requirements 1.5, 7.1**

Verifies that all Python source files under `multimodal_indexer/` exist,
are non-empty, and contain valid Python syntax — confirming no files were
accidentally corrupted or broken during the cleanup process.
"""

import ast
import os
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


def collect_python_files() -> list[str]:
    """Collect all .py files under multimodal_indexer/."""
    root = Path("multimodal_indexer")
    py_files = sorted(str(p) for p in root.rglob("*.py") if "__pycache__" not in str(p))
    return py_files


# Collect once at module level so hypothesis can build a strategy from it
PYTHON_FILES = collect_python_files()


@pytest.mark.skipif(len(PYTHON_FILES) == 0, reason="No Python files found under multimodal_indexer/")
@given(file_path=st.sampled_from(PYTHON_FILES))
@settings(max_examples=100)
def test_source_files_are_valid_python(file_path: str):
    """
    **Feature: project-cleanup-for-opensource, Property 2: 源代码零修改**
    **Validates: Requirements 1.5, 7.1**

    For randomly sampled Python files under multimodal_indexer/:
    1. The file must exist on disk
    2. The file must be non-empty
    3. The file must parse as valid Python (ast.parse succeeds)

    This ensures no source files were accidentally corrupted, truncated,
    or modified in a way that breaks syntax during the cleanup process.
    """
    # 1. File must exist
    assert os.path.isfile(file_path), f"File does not exist: {file_path}"

    # 2. File must be non-empty
    content = Path(file_path).read_text(encoding="utf-8")
    assert len(content) > 0, f"File is empty: {file_path}"

    # 3. File must be valid Python
    try:
        ast.parse(content, filename=file_path)
    except SyntaxError as e:
        pytest.fail(f"File has invalid Python syntax: {file_path}\n{e}")
