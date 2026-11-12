"""Runs inside the CodeEval-Pro grader container (image siamese-cep:1): grade one program on a list of tests.

    python3 /grader/cep_driver.py /work/job.json
    job = {"code": "...", "kind": "script" | "unittest", "setup": "...", "tests": [{"name": ..., "source": ...}],
           "timeout": 30}

The program is run as one module; the tests run in that same module's namespace, as CodeEval-Pro (program + test code
in one file) and BigCodeBench (program + its unittest module, exec'd together) do.
- script (CodeEval-Pro test_code): each test is a piece of the test code, run in order.
- unittest (BigCodeBench): `setup` is the test module (class TestCases); each test is one of its methods.
Every test is run (a failure does not stop the rest). All of it happens in a child process with its output thrown
away (a program's prints cannot reach this process's stdout); per-test time limit by SIGALRM, and a global one for
the child. Prints one line: __grade__ followed by {"results": [true|false, ...], "error": {...} | null} as JSON,
where `error` describes the first failed test (index, name, kind: load | fail | error | timeout, text).
"""
import json
import linecache
import multiprocessing
import os
import signal
import sys
import traceback

MARK = "__grade__"
LIMIT = 1500


class _Timeout(BaseException):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _source(text: str, name: str) -> str:
    """Compile with a name whose lines tracebacks can show."""
    linecache.cache[name] = (len(text), None, text.splitlines(True), name)
    return compile(text, name, "exec")


def _text(error: BaseException) -> str:
    """The exception and the last frames of its traceback that are in the program or the tests, short."""
    frames = [f for f in traceback.extract_tb(error.__traceback__) if f.filename.startswith("<")][-3:]
    text = "".join(["Traceback (most recent call last):\n"] * bool(frames) + traceback.format_list(frames) +
                   traceback.format_exception_only(type(error), error))
    return text if len(text) <= LIMIT else "..." + text[-LIMIT:]


def _child(job, conn):
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    sys.stdin = open(os.devnull)
    os.environ.setdefault("MPLBACKEND", "Agg")
    signal.signal(signal.SIGALRM, _alarm)
    limit = int(job.get("timeout", 30))
    ns = {"__name__": "solution", "__builtins__": __builtins__}
    try:
        signal.alarm(limit)
        exec(_source(job["code"], "<solution>"), ns)
        if job["kind"] == "unittest":
            exec(_source(job.get("setup") or "", "<tests>"), ns)
        signal.alarm(0)
    except BaseException as error:  # noqa: BLE001  (a program may raise anything, even SystemExit)
        signal.alarm(0)
        kind = "timeout" if isinstance(error, _Timeout) else "load"
        conn.send(("load", kind, "time limit exceeded while loading the program" if kind == "timeout" else _text(error)))
        return
    import unittest
    for i, test in enumerate(job["tests"]):
        try:
            signal.alarm(limit)
            if job["kind"] == "script":
                exec(_source(test["source"], f"<test {i + 1}>"), ns)
                outcome = (True, None, None)
            else:
                result = unittest.TestResult()
                ns["TestCases"](test["name"]).run(result)
                bad = result.errors + result.failures
                outcome = (True, None, None) if result.wasSuccessful() else \
                    (False, "fail" if result.failures else "error", (bad[0][1] if bad else "the test did not pass")[-LIMIT:])
            signal.alarm(0)
        except _Timeout:
            outcome = (False, "timeout", "time limit exceeded")
        except BaseException as error:  # noqa: BLE001
            signal.alarm(0)
            outcome = (False, "fail" if isinstance(error, AssertionError) else "error", _text(error))
        conn.send(("test", i, outcome))
    conn.send(("done",))


def grade(job: dict) -> dict:
    n = len(job["tests"])
    results, error = [], None
    parent, child = multiprocessing.Pipe(duplex=False)
    p = multiprocessing.Process(target=_child, args=(job, child))
    p.start()
    child.close()
    limit = int(job.get("timeout", 30)) * (n + 1) + 10
    import time
    deadline = time.time() + limit
    while True:
        left = deadline - time.time()
        if left <= 0 or not parent.poll(left):
            why = "time limit exceeded" if left <= 0 else "the program ended the test run"
            break
        try:
            msg = parent.recv()
        except EOFError:
            why = "the program ended the test run"
            break
        if msg[0] == "load":
            error = {"index": 0, "name": None, "kind": msg[1], "text": msg[2]}
            results = [False] * n
            break
        if msg[0] == "done":
            why = None if len(results) == n else "the test run ended early"
            break
        if msg[0] != "test" or msg[1] != len(results) or len(results) >= n:   # out of order, repeated or extra: the
            why = "the test run sent an unexpected result"                      # program tampered with the run
            error = error or {"index": len(results), "name": None, "kind": "tampered", "text": why}
            break
        ok, kind, text = msg[2]
        results.append(ok is True)
        if not ok and error is None:
            error = {"index": msg[1], "name": job["tests"][msg[1]]["name"], "kind": kind, "text": text}
    if p.is_alive():
        p.kill()
    p.join()
    if len(results) < n:                         # the child stopped early (time limit, crash, os._exit)
        if error is None:
            i = len(results)
            error = {"index": i, "name": job["tests"][i]["name"], "kind": "timeout" if "time" in why else "error", "text": why}
        results += [False] * (n - len(results))
    return {"results": results, "error": error}


def main(path):
    job = json.load(open(path))
    os.remove(path)                                # the program must not read the tests (review, 28 Oct)
    try:
        out = grade(job)
    except Exception as error:  # the test framework itself failed
        out = {"results": [False] * len(job.get("tests") or [None]),
               "error": {"index": 0, "name": None, "kind": "framework", "text": repr(error)[-LIMIT:]}}
    print(MARK + json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1])
