# CodeEval-Pro data: not included, fetch it yourself

The task `task.name=codeeval_pro` reads three JSON files from this folder. They are not redistributed here,
because the CodeEval-Pro repository states no licence for its data. Put them in this folder under these names:

| File | Source |
|---|---|
| `bigcodebench_lite_pro.json` | github.com/CodeEval-Pro/CodeEval-Pro, commit 36f292eafc597b38535fbc8b4aafea8e5c654e3c, `dataset/` (57 records; also on Hugging Face as CodeEval-Pro/bigcodebench-lite-pro, revision f70ee47) |
| `humaneval_pro.json` | same repository and commit, `dataset/` (164 records; the base problems are OpenAI's HumanEval, MIT) |
| `bigcodebench_v0.1.4_tests.json` | huggingface.co/datasets/bigcode/bigcodebench (Apache-2.0), revision b74c0d0bf70d2c0bc459be537895cca163007f1a, split v0.1.4: for the 57 tasks that are the bases of BigCodeBench-Lite-Pro, the fields task_id, complete_prompt, canonical_solution, test, entry_point, libs, as a JSON list |

References: Yu, Zhao, Cohan, Zhang, "HumanEval Pro and MBPP Pro: Evaluating Large Language Models on Self-invoking
Code Generation", Findings of ACL 2025, arXiv 2412.21199. Zhuo et al., "BigCodeBench", ICLR 2025.

Notes:
- The `id` field of `bigcodebench_lite_pro.json` is wrong for 46 records (shifted from the 12th record on);
  `siam/tasks/codeeval_pro.py` matches each base problem to its BigCodeBench task by text.
- `Dockerfile` in this folder builds the grading sandbox; its library versions follow BigCodeBench's
  `Requirements/requirements-eval.txt` (github.com/bigcode-project/bigcodebench, commit 09dd993).
- The tests that need this data (`tests/test_codeeval_pro.py`) are not in this repository.

What is ours (MIT like the rest of this folder): the grader driver (`siam/cep_driver.py`), the grader
(`siam/cep_grader.py`), the task and the streams (`siam/tasks/codeeval_pro.py`, `siam/tasks/cep_streams.py`).
