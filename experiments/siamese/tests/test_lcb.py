import base64
import json
import pickle
import shutil
import subprocess
import zlib

import pytest
from omegaconf import OmegaConf

from lcb.prompts import check_prompt, extract_code, question_prompt
from siam.problems import decode_private, streams


def test_private_tests_decode_plain_data_only():
    tests = [{"input": "1", "output": "2", "testtype": "stdin"}]
    packed = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode()
    assert decode_private(packed) == tests

    class Evil:
        def __reduce__(self):
            return (print, ("pwned",))
    bad = base64.b64encode(zlib.compress(pickle.dumps(Evil()))).decode()
    with pytest.raises(pickle.UnpicklingError):
        decode_private(bad)


def test_streams_same_and_different():
    pool = [{"id": f"p{i}"} for i in range(10)]
    diff = streams(pool, 2, 3, "different", 7)
    same = streams(pool, 2, 3, "same", 7)
    assert not {p["id"] for p in diff[0]} & {p["id"] for p in diff[1]}
    assert same[0] == same[1] == diff[0]              # same seed: the first block is the same problems
    with pytest.raises(ValueError):
        streams(pool, 2, 6, "different", 7)


def test_lcb_texts():
    p = {"question_content": "Q", "starter_code": "class Solution:\n    def f(self): pass"}
    assert "### Format: You will use the following starter code" in question_prompt(p)
    assert "Read the inputs from stdin" in question_prompt({"question_content": "Q", "starter_code": ""})
    assert check_prompt({"error_code": -2, "inputs": "1", "output": "3", "expected": "2"}).startswith(
        "The above code is incorrect and got a wrong answer.")
    assert "None" not in check_prompt({"error_code": -4, "error_message": "Error during testing: invalid syntax"})
    assert extract_code("a\n```python\nx=1\n```\nb\n```python\nx=2\n```") == "x=2"


def docker_image():
    if not shutil.which("docker"):
        return False
    out = subprocess.run(["docker", "image", "inspect", "siamese-grader:1"], capture_output=True)
    return out.returncode == 0


@pytest.mark.skipif(not docker_image(), reason="needs Docker and siamese-grader:1")
def test_real_grader_stdin_and_functional():
    from siam.grader import Grader
    g = Grader(OmegaConf.create({"image": "siamese-grader:1", "memory": "1g", "tmpfs_size": "256m", "command_timeout_s": 300,
                                    "max_file_bytes": 134217728}))
    try:
        stdin = {"id": "s", "public": [{"input": "2 3\n", "output": "5\n", "testtype": "stdin"}],
                 "private": [{"input": "10 20\n", "output": "30\n", "testtype": "stdin"}], "fn_name": None}
        ok = "a, b = map(int, input().split())\nprint(a + b)"
        assert g.grade(stdin, ok, "all")["passed"]
        wrong = g.grade(stdin, "print(5)", "all")
        assert not wrong["passed"] and wrong["n_passed_before_fail"] == 1 and "wrong answer" in wrong["feedback"]
        assert g.grade(stdin, "print(5)", "public")["passed"]          # public only: the hidden test is not run
        func = {"id": "f", "public": [{"input": "[1, 2]\n3", "output": "[1, 2, 3]", "testtype": "functional"}],
                "private": [], "fn_name": "add"}
        code = "class Solution:\n    def add(self, xs, x):\n        return xs + [x]\n"
        assert g.grade(func, code, "all")["passed"]
        big = {"id": "b", "public": [], "private": [{"input": "x" * (10 * 2**20) + "\n", "output": "1\n", "testtype": "stdin"}],
               "fn_name": None}
        assert g.grade(big, "print(1)", "all")["passed"]                # a 10 MB test set (LCB has up to ~32 MB)
        loop = g.grade(stdin, "while True: pass", "public")
        assert not loop["passed"] and "time limit" in loop["feedback"]
    finally:
        g.close()
