"""Grade CodeEval-Pro programs inside a Docker sandbox (image siamese-cep:1; third_party/codeeval_pro/Dockerfile).

Like siam/grader.py (same hardened container, ctxfile/sandbox.py: no network, no host mounts, read-only root, limits;
leftover processes killed after every job; results cached by (item, tests, code)), but the tests are code: our driver
(siam/cep_driver.py, in the image) runs the program and then the item's tests in the program's namespace. `which`:
public = the item's first test only, all = every test. The result has the keys of siam/grader.py's.
"""
from __future__ import annotations

import hashlib
import re
import json

from ctxfile import sandbox as sbx

MARK = "__grade__"
SUSPICIOUS = re.compile(r"wasSuccessful|_getframe|multiprocessing|/work/job|conn\.send|gc\.get_objects|"
                        r"unittest\.TestResult|sys\.modules\[.__main__.\]", re.I)   # tests run in the program's process


def feedback(error: dict | None, tests: list[dict]) -> str:
    """The text for a failed run (in the words of LCB's self-repair feedback where they fit)."""
    if error is None:
        return ""
    text = (error.get("text") or "").rstrip()
    if error["kind"] == "load":
        if "SyntaxError" in text or "IndentationError" in text:
            return f"The above code is incorrect and got the following compilation error.\n{text}"
        return f"The above code is incorrect and got a runtime error while it was loaded.\n{text}"
    if error["kind"] == "framework":
        return f"The above code could not be tested.\n{text}"
    test = tests[error["index"]]["source"].strip("\n") if error.get("index") is not None and error["index"] < len(tests) else ""
    shown = f"\n```python\n{test}\n```" if test else ""
    if error["kind"] == "timeout":
        return f"The above code is incorrect and got time limit exceeded on this test:{shown}"
    if error["kind"] == "fail":
        return f"The above code is incorrect and failed this test:{shown}\n{text}"
    return f"The above code is incorrect and got a runtime error on this test:{shown}\n{text}"


class CepGrader:
    def __init__(self, settings):
        self.s = settings
        sbx.check_available(str(settings.image))
        self.box = sbx.Sandbox(settings)
        self.box.start()
        self.cache: dict[str, dict] = {}
        self.image_id = sbx.image_id(str(settings.image))

    def grade(self, item: dict, code: str, which: str = "all") -> dict:
        """{passed, n_tests, n_passed, n_passed_before_fail, results, metadata, feedback, seconds, cached}."""
        t = item["tests"]
        tests = t["cases"][:1] if which == "public" else t["cases"]
        key = hashlib.sha256(f"{item['id']}\0{which}\0{code}".encode()).hexdigest()
        if key in self.cache:
            return dict(self.cache[key], cached=True)
        n = len(tests)
        if not code.strip():
            out, seconds = {"results": [False] * n, "error": {"index": None, "kind": "load", "text": "no code was given"}}, 0.0
        else:
            job = json.dumps({"code": code, "kind": t["kind"], "setup": t.get("setup", ""), "tests": tests,
                              "timeout": int(self.s.get("timeout_per_test", 30))})
            self._put("/work/job.json", job)
            run = self.box.run("python3 /grader/cep_driver.py /work/job.json")
            seconds = run["seconds"]
            line = next((l for l in run["stdout"].splitlines() if l.startswith(MARK)), None)
            if line is None:   # the driver itself failed (e.g. the container's memory limit): count as a failed run
                out = {"results": [False] * n, "error": {"index": None, "kind": "framework",
                                                         "text": (run["stderr"] or "the grader gave no result")[-300:]}}
            else:
                out = json.loads(line[len(MARK):])
            self.box.reset_scratch()
        results = [r is True for r in out["results"]][:n]
        results += [False] * (n - len(results))
        passed = len(results) == n and all(results)
        done = 0
        for r in results:
            if not r:
                break
            done += 1
        error = out.get("error")
        meta = {} if passed else {"error_code": (error or {}).get("kind", "fail"), "error": ((error or {}).get("text") or "")[-500:],
                                  "test": (error or {}).get("name")}
        meta["suspicious"] = bool(SUSPICIOUS.search(code or ""))   # reaching into the grader: flagged, not blocked
        result = {"passed": passed, "n_tests": n, "n_passed": sum(results), "n_passed_before_fail": done, "results": results,
                  "metadata": meta, "seconds": seconds, "cached": False,
                  "feedback": "" if passed else feedback(error, tests)}
        self.cache[key] = result
        return result

    def _put(self, path: str, text: str) -> None:
        out = sbx.docker(["exec", "-i", self.box.name, "sh", "-c", f"cat > {path}"], timeout=300, data=text.encode("utf-8"))
        if out.returncode != 0:
            self.box.restart()
            out = sbx.docker(["exec", "-i", self.box.name, "sh", "-c", f"cat > {path}"], timeout=300, data=text.encode("utf-8"))
            if out.returncode != 0:
                raise RuntimeError(f"could not write the grading job: {out.stderr.decode(errors='replace')[:300]}")

    def close(self) -> None:
        self.box.stop()
