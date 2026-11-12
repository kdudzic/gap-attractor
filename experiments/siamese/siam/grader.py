"""Grade programs with LiveCodeBench's own test runner inside a Docker sandbox (image siamese-grader:1).

Agents never get a shell: the harness alone writes the job into the container and runs LCB's procedure there
(lcb/driver.py: run_test in a child process, LCB's global timeout). The container is the hardened one of
experiments/context_file (ctxfile/sandbox.py: no network, no host mounts, read-only root, limits); after every job
all leftover processes are killed. A result is cached by (problem, tests, code).
"""
from __future__ import annotations

import hashlib
import json

from ctxfile import sandbox as sbx
from lcb.prompts import check_prompt
from siam.problems import input_output

MARK = "__grade__"


class Grader:
    def __init__(self, settings):
        self.s = settings
        sbx.check_available(str(settings.image))
        self.box = sbx.Sandbox(settings)
        self.box.start()
        self.cache: dict[str, dict] = {}
        self.image_id = sbx.image_id(str(settings.image))

    def grade(self, problem: dict, code: str, which: str = "all") -> dict:
        """{passed, n_tests, n_passed_before_fail, results, metadata, feedback, seconds, cached}."""
        io_json = input_output(problem, which)
        key = hashlib.sha256(f"{problem['id']}\0{which}\0{code}".encode()).hexdigest()
        if key in self.cache:
            return dict(self.cache[key], cached=True)
        n = len(json.loads(io_json)["inputs"])
        if not code.strip():
            out = {"results": [-1] * n, "metadata": {"error_code": -1, "error": "no code was given", "error_message": "no code"}}
            seconds = 0.0
        else:
            job = json.dumps({"input_output": io_json, "code": code, "timeout": int(self.s.get("timeout_per_test", 6))})
            self._put("/work/job.json", job)
            run = self.box.run("python3 /grader/driver.py /work/job.json")
            seconds = run["seconds"]
            line = next((l for l in run["stdout"].splitlines() if l.startswith(MARK)), None)
            if line is None:   # the driver itself failed (e.g. the container's memory limit): count as a failed run
                out = {"results": [-1] * n, "metadata": {"error_code": -5, "error_message": "TestRunnerError",
                                                          "error": (run["stderr"] or "the grader gave no result")[-300:]}}
            else:
                out = json.loads(line[len(MARK):])
            self.box.reset_scratch()
        results = out["results"]
        passed = len(results) == n and all(r is True for r in results)
        done = 0
        for r in results:
            if r is not True:
                break
            done += 1
        result = {"passed": passed, "n_tests": n, "n_passed_before_fail": done, "results": results,
                  "metadata": out["metadata"], "seconds": seconds, "cached": False,
                  "feedback": "" if passed else check_prompt(out["metadata"])}
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
