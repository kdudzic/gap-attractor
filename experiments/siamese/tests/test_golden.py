"""The requests every scripted world sends equal the golden ones (written before the Task refactor)."""
import json

import pytest

from golden import CASES, FIXTURE, record


@pytest.fixture(scope="module")
def now():
    return record()


@pytest.mark.parametrize("case", sorted(CASES))
def test_requests_and_logs_are_unchanged(now, case):
    gold = json.loads(FIXTURE.read_text(encoding="utf-8"))[case]
    got = now[case]
    for k, (g, n) in enumerate(zip(gold["requests"], got["requests"])):
        assert len(g) == len(n), f"agent {k + 1}: {len(n)} calls, golden {len(g)}"
        for i, (gm, nm) in enumerate(zip(g, n)):
            assert gm == nm, f"agent {k + 1}, call {i + 1} differs"
    assert gold["tools"] == got["tools"]
    assert gold["logs"] == got["logs"]
    assert gold["files"] == got["files"]
