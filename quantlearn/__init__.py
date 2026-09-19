"""
quantlearn — tiny Kaggle-Learn-style checker for the *_tests.ipynb notebooks.

In a tests notebook:

    import sys; sys.path.append("..")
    from quantlearn import load_questions
    q1, q2, q3 = load_questions("02_pandas/00_series_fundamentals")

    answer = ____          # your code, assigned to the variable the task names
    q1.check()             # ✅ / ❌ with a short reason (reads `answer` from the notebook)
    q1.hint()              # a nudge
    q1.solution()          # the reference code

Aliases: q1.give_hint(), q1.give_solution().
"""
import importlib
import inspect
import textwrap
import traceback

import numpy as np
import pandas as pd

__all__ = ["load_questions", "Question"]


class Question:
    def __init__(self, number, prompt, setup, solution, hint, answer_var="answer",
                 common_setup="", tol=1e-6, check_fn=None, note=""):
        self.number = number
        self.prompt = textwrap.dedent(prompt).strip()
        self.setup = textwrap.dedent(setup).strip("\n")
        self.solution_code = textwrap.dedent(solution).strip("\n")
        self.hint_text = textwrap.dedent(hint).strip()
        self.answer_var = answer_var
        self.common_setup = textwrap.dedent(common_setup).strip("\n")
        self.tol = tol
        self.check_fn = check_fn
        self.note = textwrap.dedent(note).strip()

    # ------------------------------------------------------------------ public
    def hint(self):
        print(f"Hint for task {self.number}:\n  " + self.hint_text.replace("\n", "\n  "))

    def solution(self):
        print(f"Solution for task {self.number}:\n")
        print(textwrap.indent(self.solution_code, "    "))
        if self.note:
            print("\n" + self.note)

    give_hint = hint
    give_solution = solution

    def check(self, *args, **named):
        """Check your answer.

        q1.check()            reads the answer variable(s) from the notebook
        q1.check(my_value)    checks an explicit value
        q1.check(a=..., b=...) for tasks with several answer variables
        """
        try:
            expected = self._expected()
        except Exception:
            print("⚠️  The reference solution itself failed to run:")
            traceback.print_exc()
            return False
        if not args and not named:
            frame = inspect.currentframe().f_back
            scope = dict(frame.f_globals); scope.update(frame.f_locals)
            names = self.answer_var if isinstance(self.answer_var, (list, tuple)) else [self.answer_var]
            missing = [n for n in names if n not in scope]
            if missing:
                print(f"❌ Not yet. Assign your result to `{missing[0]}` first.")
                return False
            if isinstance(self.answer_var, (list, tuple)):
                named = {n: scope[n] for n in names}
            else:
                args = (scope[self.answer_var],)
        if named:
            ok, msg = self._compare_many(named, expected)
        else:
            ok, msg = self._compare(args[0], expected)
        print(("✅ Correct. " if ok else "❌ Not yet. ") + msg)
        return ok

    # ----------------------------------------------------------------- private
    def _expected(self):
        ns = {}
        exec(self.common_setup + "\n" + self.setup + "\n" + self.solution_code, ns)
        if isinstance(self.answer_var, (list, tuple)):
            return {v: ns[v] for v in self.answer_var}
        return ns[self.answer_var]

    def _compare_many(self, named, expected):
        for k, exp in expected.items():
            if k not in named:
                return False, f"missing `{k}=`"
            ok, msg = self._compare(named[k], exp)
            if not ok:
                return False, f"`{k}`: {msg}"
        return True, ""

    def _compare(self, got, exp):
        if self.check_fn is not None:
            try:
                res = self.check_fn(got, exp)
            except Exception as e:
                return False, f"{type(e).__name__}: {e}"
            if isinstance(res, tuple):
                return res
            return bool(res), "" if res else "does not match the reference"
        return compare(got, exp, self.tol)


def compare(got, exp, tol=1e-6):
    """Generic comparison returning (ok, message)."""
    if got is None:
        return False, "answer is None — assign your result to the answer variable"
    try:
        if isinstance(exp, pd.DataFrame):
            if not isinstance(got, pd.DataFrame):
                return False, f"expected a DataFrame, got {type(got).__name__}"
            if got.shape != exp.shape:
                return False, f"shape {got.shape} differs from expected {exp.shape}"
            if list(got.columns) != list(exp.columns):
                if sorted(map(str, got.columns)) == sorted(map(str, exp.columns)):
                    return False, f"columns are right but in a different order: {list(got.columns)}"
                return False, f"columns {list(got.columns)} differ from expected {list(exp.columns)}"
            if not got.index.equals(exp.index):
                if len(got) == len(exp) and sorted(map(str, got.index)) == sorted(map(str, exp.index)):
                    return False, "index labels are right but rows are in a different order"
                return False, "index differs from expected"
            pd.testing.assert_frame_equal(got, exp, check_dtype=False, check_exact=False,
                                          rtol=tol, atol=tol, check_names=False, check_freq=False)
            return True, ""
        if isinstance(exp, pd.Series):
            if isinstance(got, pd.DataFrame) and got.shape[1] == 1:
                return False, "expected a Series but got a one-column DataFrame (use df['col'] not df[['col']])"
            if not isinstance(got, pd.Series):
                return False, f"expected a Series, got {type(got).__name__}"
            if len(got) != len(exp):
                return False, f"length {len(got)} differs from expected {len(exp)}"
            if not got.index.equals(exp.index):
                if sorted(map(str, got.index)) == sorted(map(str, exp.index)):
                    return False, "values may be right but the index order differs"
                return False, "index differs from expected"
            pd.testing.assert_series_equal(got, exp, check_dtype=False, check_exact=False,
                                           rtol=tol, atol=tol, check_names=False, check_freq=False,
                                           check_index_type=False)
            return True, ""
        if isinstance(exp, pd.Index):
            if not isinstance(got, pd.Index):
                got = pd.Index(got)
            if not got.equals(exp):
                return False, "index differs from expected"
            return True, ""
        if isinstance(exp, np.ndarray):
            got_a = np.asarray(got)
            if got_a.shape != exp.shape:
                return False, f"shape {got_a.shape} differs from expected {exp.shape}"
            if exp.dtype.kind in "fiu" and got_a.dtype.kind in "fiu":
                if not np.allclose(got_a, exp, rtol=tol, atol=tol, equal_nan=True):
                    return False, "values differ from expected"
            elif not np.array_equal(got_a, exp):
                return False, "values differ from expected"
            return True, ""
        if isinstance(exp, (bool, np.bool_)):
            if bool(got) != bool(exp):
                return False, f"expected {exp}, got {got}"
            return True, ""
        if isinstance(exp, (int, float, np.integer, np.floating)):
            try:
                g = float(got)
            except Exception:
                return False, f"expected a number, got {type(got).__name__}"
            if np.isnan(exp) and np.isnan(g):
                return True, ""
            if not np.isclose(g, float(exp), rtol=max(tol, 1e-6), atol=tol):
                return False, f"expected {exp:.6g}, got {g:.6g}"
            return True, ""
        if isinstance(exp, str):
            if str(got).strip() != exp.strip():
                return False, f"expected {exp!r}, got {got!r}"
            return True, ""
        if isinstance(exp, (list, tuple)):
            got_l = list(got) if not isinstance(got, (list, tuple)) else list(got)
            if len(got_l) != len(exp):
                return False, f"length {len(got_l)} differs from expected {len(exp)}"
            for g, e in zip(got_l, exp):
                ok, msg = compare(g, e, tol)
                if not ok:
                    return False, msg
            return True, ""
        if isinstance(exp, dict):
            if not isinstance(got, dict):
                return False, f"expected a dict, got {type(got).__name__}"
            if set(got) != set(exp):
                return False, f"keys {sorted(map(str, got))} differ from expected {sorted(map(str, exp))}"
            for k in exp:
                ok, msg = compare(got[k], exp[k], tol)
                if not ok:
                    return False, f"key {k!r}: {msg}"
            return True, ""
        if isinstance(exp, set):
            if set(got) != exp:
                return False, "set differs from expected"
            return True, ""
        # fallback
        if got == exp:
            return True, ""
        return False, "does not match the reference"
    except AssertionError as e:
        first = str(e).strip().splitlines()[0] if str(e).strip() else "values differ"
        return False, first
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def load_questions(name):
    """`name` like '02_pandas/00_series_fundamentals' -> list of Question objects."""
    mod_name = "quantlearn.questions.q_" + name.replace("/", "__")
    mod = importlib.import_module(mod_name)
    common = getattr(mod, "COMMON_SETUP", "")
    out = []
    for i, spec in enumerate(mod.QUESTIONS, start=1):
        spec = dict(spec)
        out.append(Question(i, common_setup=common, **spec))
    return out
