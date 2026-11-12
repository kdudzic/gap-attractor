"""Runs inside the grader container: grade one program on one problem's tests, the way LiveCodeBench does.

    python3 /grader/driver.py /work/job.json      job = {"input_output": "<LCB json>", "code": "...", "timeout": 6}

Same procedure as lcb_runner/evaluation/compute_code_generation_metrics.py check_correctness (LCB, MIT): run_test in
a child process with LCB's global timeout, (timeout + 1) x number of tests + 5 s; if the child gives nothing, every
test counts as failed (-1). Prints one line: __grade__ followed by {"results": [...], "metadata": {...}} as JSON.
"""
import json
import multiprocessing
import sys

sys.set_int_max_str_digits(50000)
sys.path.insert(0, "/grader")

from testing_util import run_test  # noqa: E402  (vendored, unchanged)


def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    res, metadata = run_test(sample, test=generation, debug=debug, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)


def check_correctness(sample, generation, timeout, debug=False):
    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(sample, generation, debug, result, metadata_list, timeout))
    p.start()
    p.join(timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5)
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        return [-1 for _ in range(len(in_outs["inputs"]))], {"error_code": -1, "error_message": "global timeout",
                                                             "error": "global timeout"}
    return list(result[0]), dict(metadata_list[0])


def plain(value):
    """numpy scalars and the like -> JSON values (as compute_code_generation_metrics does)."""
    if hasattr(value, "item"):
        value = value.item()
    return value if isinstance(value, (bool, int, float, str)) or value is None else str(value)


def main(path):
    job = json.load(open(path))
    try:
        results, metadata = check_correctness({"input_output": job["input_output"]}, job["code"], int(job.get("timeout", 6)))
    except Exception as error:  # as LCB: the test framework failed
        results, metadata = [-2], {"error": repr(error), "error_code": -5, "error_message": "TestRunnerError"}
    out = {"results": [plain(r) for r in results], "metadata": {k: plain(v) for k, v in metadata.items()}}
    print("__grade__" + json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1])
