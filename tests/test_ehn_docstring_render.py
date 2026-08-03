"""A docstring that renders wrong fails silently — nothing raises, nothing warns.

`relax.py` documented the alpha stability bound as an aligned table, drawn with a
`\\` as the top arm of a brace. Inside a normal triple-quoted string a backslash
immediately before a newline is a LINE CONTINUATION: Python removes both, so the
dx=1.60 and dx=0.80 rows silently rendered as one line, in the one place a reader
goes to decide whether their step size is safe. Review caught it; the suite could
not, because nothing here read a docstring.

The general guard is worth more than the specific one: any docstring line ending
in an odd number of backslashes is a continuation the author almost certainly did
not intend, since the alternative (deliberately joining two lines of prose) has
no reason to appear in a docstring.
"""
import ast
import pathlib
import sys

import pytest

import jax_solitons.ehn as ehn_pkg

EHN_DIR = pathlib.Path(ehn_pkg.__file__).resolve().parent
SOURCES = sorted(EHN_DIR.glob("*.py"))


def _docstring_line_ranges(path):
    """(start, end) 1-based inclusive line spans of every docstring in a module.

    A docstring is the first statement of a module/class/function and is an
    ast.Expr wrapping a str constant, so its lineno/end_lineno bound exactly the
    physical lines of the literal — which is what the backslash check must look
    at and nothing else.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    spans = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            spans.append((first.lineno, first.end_lineno))
    return spans


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_accidental_line_continuation_in_docstrings(path):
    """Scans DOCSTRING lines only.

    An explicit continuation in ordinary code is legitimate and common, so
    checking every physical line would fail honest code — the earlier version of
    this test did exactly that, and only passed because no ehn module happens to
    use one. Inside a docstring the same backslash silently eats the next line.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    for start, end in _docstring_line_ranges(path):
        for lineno in range(start, min(end, len(lines)) + 1):
            stripped = lines[lineno - 1].rstrip()
            if not stripped.endswith("\\"):
                continue
            trailing = len(stripped) - len(stripped.rstrip("\\"))
            if trailing % 2 == 1:
                pytest.fail(
                    f"{path.name}:{lineno} ends in an odd number of backslashes "
                    f"INSIDE A DOCSTRING:\n    {stripped!r}\n"
                    "That joins the next line onto this one. Use a box-drawing "
                    "character for ASCII art, or double the backslash."
                )


def test_the_alpha_bound_table_still_has_one_row_per_dx():
    """The specific regression: six rows, each on its own line.

    Unlike the check above — which reads the SOURCE and so is unaffected — this
    one reads the rendered __doc__, and `python -OO` strips docstrings entirely.
    Skip explicitly in that case rather than letting `in None` raise TypeError:
    a stripped docstring is not a defect, it just means this guard cannot run,
    and a skip says that where a failure would misreport it.
    """
    if sys.flags.optimize >= 2:
        pytest.skip("-OO strips docstrings; this guard has nothing to read")
    from jax_solitons.ehn import relax
    doc = relax.__doc__
    assert doc is not None, "relax has no module docstring at all"
    assert "α_max = 2/H" in doc, "the stability table is gone"
    body = doc[doc.index("α_max = 2/H"):]
    for dx in ("1.60", "0.80", "0.40", "0.20", "0.10", "0.05"):
        rows = [ln for ln in body.splitlines() if ln.strip().startswith(dx)]
        assert len(rows) == 1, f"dx={dx} does not start its own line — table merged"
