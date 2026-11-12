"""LiveCodeBench's own texts, copied verbatim (LiveCodeBench, MIT, commit 28fef95; see lcb/LICENSE and NOTICE.md).

- the generic system message and question template of the code-generation scenario
  (lcb_runner/prompts/code_generation.py: PromptConstants, get_generic_question_template_answer);
- the self-repair feedback text (lcb_runner/prompts/self_repair.py: get_check_prompt), which we use for the public-test
  feedback of run_tests;
- the code extraction rule (lcb_runner/utils/extraction_utils.py: extract_code, generic styles): the last fenced block.
Only the function signatures are adapted (a problem dict instead of their dataclass; metadata as a dict).
"""
from __future__ import annotations

SYSTEM_MESSAGE_GENERIC = ("You are an expert Python programmer. You will be given a question (problem specification) and "
                          "will generate a correct Python program that matches the specification and passes all tests.")

FORMATTING_MESSAGE_WITH_STARTER_CODE = ("You will use the following starter code to write the solution to the problem and "
                                        "enclose your code within delimiters.")

FORMATTING_WITHOUT_STARTER_CODE = ("Read the inputs from stdin solve the problem and write the answer to stdout (do not "
                                   "directly test on the sample inputs). Enclose your code within delimiters as follows. "
                                   "Ensure that when the python program runs, it reads the inputs, runs the algorithm and "
                                   "writes output to STDOUT.")


def question_prompt(problem: dict) -> str:
    """get_generic_question_template_answer."""
    prompt = f"### Question:\n{problem['question_content']}\n\n"
    if problem.get("starter_code"):
        prompt += f"### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{problem['starter_code']}\n```\n\n"
    else:
        prompt += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def check_prompt(metadata: dict) -> str:
    """get_check_prompt: the feedback for a failed run ("" when the metadata has no error)."""
    if "error_code" not in metadata:
        return ""
    code = metadata["error_code"]
    if code == -4 and "inputs" not in metadata:   # ours: LCB's run_test reports a program that does not compile as -4
        code = -1                                  # with no test and only error_message (LCB's text would say
        metadata = dict(metadata, error=metadata.get("error") or metadata.get("error_message"))   # "Input: None"): we
                                                   # use LCB's own compilation-error text for it
    if code == -1:
        return f"The above code is incorrect and got the following compilation error.\n{metadata.get('error')}"
    if code == -2:
        return (f"The above code is incorrect and got a wrong answer.\nInput: {metadata.get('inputs')}\n"
                f"Generated Output: {metadata.get('output')}\nExpected: {metadata.get('expected')}")
    if code == -3:
        return (f"The above code is incorrect and got time limit exceeded.\n{metadata.get('error')}\n"
                f"Input: {metadata.get('inputs')}\nExpected: {metadata.get('expected')}")
    if code == -4:
        return (f"The above code is incorrect and got a runtime error.\nInput: {metadata.get('inputs')}\n"
                f"Expected: {metadata.get('expected')}\n{metadata.get('error')}")
    # -5 (the test framework failed) is not covered by LCB's text: ours, in the same style
    return f"The above code could not be tested.\n{metadata.get('error') or metadata.get('error_message', '')}"


def extract_code(model_output: str) -> str:
    """extract_code for chat models: the text between the last two ``` lines, or "" when there are fewer than two."""
    lines = (model_output or "").split("\n")
    marks = [i for i, line in enumerate(lines) if "```" in line]
    if len(marks) < 2:
        return ""
    return "\n".join(lines[marks[-2] + 1:marks[-1]])
