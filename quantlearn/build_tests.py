"""
Build and verify a *_tests.ipynb exercise notebook from a quantlearn questions module.

    python -m quantlearn.build_tests                      # rebuild + verify every notebook
    python -m quantlearn.build_tests 02_pandas/00_series_fundamentals

or from Python:

    from quantlearn.build_tests import build_tests, verify
    verify("02_pandas/00_series_fundamentals")      # every solution must pass its own check
    build_tests("02_pandas/00_series_fundamentals", title="pandas Series")

The notebook is written UNEXECUTED (the user runs it). verify() also executes a
solved copy of the notebook in the scratchpad to prove the setup cells run.
"""
import importlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(__file__).resolve().parent / "_solved"
sys.path.insert(0, str(ROOT))


def _load(name):
    import quantlearn
    importlib.invalidate_caches()
    mod_name = "quantlearn.questions.q_" + name.replace("/", "__")
    if mod_name in sys.modules:
        importlib.reload(sys.modules[mod_name])
    return quantlearn.load_questions(name), sys.modules[mod_name]


def _cells(name, title, qs, mod, solved=False):
    n = len(qs)
    qnames = ", ".join(f"q{i}" for i in range(1, n + 1))
    common = textwrap.dedent(getattr(mod, "COMMON_SETUP", "")).strip("\n")
    intro = getattr(mod, "INTRO", "")
    cells = [new_markdown_cell(textwrap.dedent(f"""
    # {title} — exercises

    {n} tasks that test what the cheat sheet `{Path(name).name}.ipynb` covers. Try each one
    **before** looking at the cheat sheet.

    How it works:
    1. Read the task. Write your code in the cell below it, assigning the result to the
       variable the task names (usually `answer`).
    2. Run the cell: the last line `qN.check()` tells you ✅ or ❌ with a short reason.
    3. Stuck? Uncomment `qN.hint()` for a nudge, or `qN.solution()` to see the reference code.
    {intro}
    """).strip())]
    setup_src = f'import sys; sys.path.append("..")\nfrom quantlearn import load_questions\n{qnames} = load_questions("{name}")\n'
    if common:
        setup_src += "\n" + common
    cells.append(new_code_cell(setup_src))
    for q in qs:
        var = q.answer_var
        var_txt = f"`{var}`" if isinstance(var, str) else ", ".join(f"`{v}`" for v in var)
        cells.append(new_markdown_cell(f"## Task {q.number}\n\n{q.prompt}\n\nAssign the result to {var_txt}."))
        body = q.setup + ("\n\n" if q.setup else "")
        if solved:
            body += q.solution_code + "\n\n"
        else:
            body += "# your solution here\n"
            if isinstance(var, str):
                body += f"{var} = ____\n\n"
            else:
                body += "".join(f"{v} = ____\n" for v in var) + "\n"
        body += f"q{q.number}.check()"
        cells.append(new_code_cell(body))
        cells.append(new_code_cell(f"# q{q.number}.hint()\n# q{q.number}.solution()"))
    cells.append(new_markdown_cell("---\nDone? Re-open the cheat sheet for anything you had to look up, then try the next `_tests` notebook."))
    return cells


def build_tests(name, title):
    qs, mod = _load(name)
    nb = new_notebook(cells=_cells(name, title, qs, mod))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    out = ROOT / (name + "_tests.ipynb")
    nbformat.write(nb, out)
    print("built", out.relative_to(ROOT), f"({len(qs)} tasks)")
    return out


def verify(name, title="x"):
    """Every solution passes its own check, a wrong answer fails without raising,
    and the solved notebook executes end to end."""
    qs, mod = _load(name)
    folder = ROOT / Path(name).parent
    cwd = os.getcwd()
    os.chdir(folder)
    try:
        import io, contextlib
        for q in qs:
            exp = q._expected()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ok = q.check(exp) if not isinstance(q.answer_var, (list, tuple)) else q.check(**exp)
            assert ok, f"{name} task {q.number}: reference solution fails its own check: {buf.getvalue()}"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                bad = q.check(None) if not isinstance(q.answer_var, (list, tuple)) else q.check(**{k: None for k in exp})
            assert not bad, f"{name} task {q.number}: None passed the check"
            assert q.hint_text and q.solution_code, f"{name} task {q.number}: missing hint/solution"
    finally:
        os.chdir(cwd)
    # solved notebook executes
    nb = new_notebook(cells=_cells(name, title, qs, mod, solved=True))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    SCRATCH.mkdir(exist_ok=True)
    tmp = folder / ("_solved_" + Path(name).name + ".ipynb")
    nbformat.write(nb, tmp)
    r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                        "--ExecutePreprocessor.timeout=600", "--output", str(SCRATCH / tmp.name), str(tmp)],
                       cwd=folder, capture_output=True, text=True)
    tmp.unlink()
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise SystemExit(f"solved notebook failed: {name}")
    solved = nbformat.read(SCRATCH / tmp.name, as_version=4)
    fails = []
    for c in solved.cells:
        if c.cell_type == "code":
            for o in c.outputs:
                txt = o.get("text", "") or ""
                if "❌" in txt or o.output_type == "error":
                    fails.append(txt[:200] or o.get("ename"))
    assert not fails, f"{name}: solved notebook has failing checks: {fails}"
    print("verified", name, f"({len(qs)} tasks)")


def _title(name):
    stem = Path(name).name
    return stem.split("_", 1)[1].replace("_", " ") if "_" in stem else stem


if __name__ == "__main__":
    import glob
    names = sys.argv[1:]
    if not names:
        mods = sorted(glob.glob(str(ROOT / "quantlearn" / "questions" / "q_*.py")))
        names = [Path(m).stem[2:].replace("__", "/") for m in mods]
    for n in names:
        verify(n)
        build_tests(n, _title(n))
