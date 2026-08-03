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

import pytest

import jax_solitons.ehn as ehn_pkg

EHN_DIR = pathlib.Path(ehn_pkg.__file__).resolve().parent
SOURCES = sorted(EHN_DIR.glob("*.py"))


def _docstrings(path):
    """(qualname, text) for every docstring in a module, including the module's."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                name = getattr(node, "name", "<module>")
                out.append((f"{path.name}:{name}", doc))
    return out


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_accidental_line_continuation_in_docstrings(path):
    """ast.get_docstring returns the RENDERED text, so a swallowed newline is
    already gone by the time we see it — detect it in the source instead."""
    raw = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(raw.splitlines(), 1):
        stripped = line.rstrip()
        if not stripped.endswith("\\"):
            continue
        trailing = len(stripped) - len(stripped.rstrip("\\"))
        if trailing % 2 == 1 and not stripped.lstrip().startswith("#"):
            # An odd count escapes the newline. Legal in code (explicit
            # continuation); in a docstring table it eats a row.
            pytest.fail(
                f"{path.name}:{lineno} ends in an odd number of backslashes:\n"
                f"    {stripped!r}\n"
                "Inside a docstring this joins the next line onto this one. Use a "
                "box-drawing character for ASCII art, or double the backslash."
            )


def test_the_alpha_bound_table_still_has_one_row_per_dx():
    """The specific regression: six rows, each on its own line."""
    from jax_solitons.ehn import relax
    doc = relax.__doc__
    assert "α_max = 2/H" in doc, "the stability table is gone"
    body = doc[doc.index("α_max = 2/H"):]
    for dx in ("1.60", "0.80", "0.40", "0.20", "0.10", "0.05"):
        rows = [ln for ln in body.splitlines() if ln.strip().startswith(dx)]
        assert len(rows) == 1, f"dx={dx} does not start its own line — table merged"
