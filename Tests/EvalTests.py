from MiniLLM.Evaluation.Common import run_python, truncate_completion
from MiniLLM.Evaluation.HumanEval import build_program
from MiniLLM.Evaluation.Report import loss_curve_svg, parse_training_log, results_table

HUMANEVAL_PROBLEM = {
    "prompt": 'def add(a, b):\n    """Return a + b."""\n',
    "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
    "entry_point": "add",
}


def test_truncate_completion_stops_at_top_level_code():
    completion = "    return a + b\n\n\ndef helper():\n    pass\nprint(add(1, 2))\nmore junk"
    assert truncate_completion(completion) == "    return a + b\n\n\ndef helper():\n    pass"


def test_humaneval_program_base_and_chat():
    assert run_python(build_program(HUMANEVAL_PROBLEM, "    return a + b\nprint('x')", chat=False))
    assert not run_python(build_program(HUMANEVAL_PROBLEM, "    return a - b\n", chat=False))
    chat_answer = "Sure:\n```python\ndef add(a, b):\n    return a + b\n```"
    assert run_python(build_program(HUMANEVAL_PROBLEM, chat_answer, chat=True))


def test_report_parses_logs_and_renders_table():
    log = "\n".join([
        "params=42.5M world=1 grad_accum=8 tokens/step=65,536 steps=20 train_windows=1",
        "step 10/20 loss 6.5000 lr 1e-3 gnorm 1.0 9.0k tok/s eta 0.1h",
        "eval val/english_loss=6.4000 val/english_ppl=601.8450",
        "step 20/20 loss 5.0000 lr 1e-4 gnorm 1.0 9.0k tok/s eta 0.0h",
    ])
    train_points, eval_points = parse_training_log(log, tokens_per_step=100)
    assert train_points == [(10, 1000, 6.5), (20, 2000, 5.0)]
    assert eval_points[0][2]["english_loss"] == 6.4
    assert loss_curve_svg(train_points, eval_points).startswith("<svg")

    table = results_table([
        {"name": "gpt2", "limit": None, "scores": {"hellaswag_acc_norm": 0.311, "blimp_acc": 0.82}},
        {"name": "minillm", "limit": 100, "scores": {"english_ppl": 21.5}},
    ])
    assert "| gpt2 | 31.1 | 82.0 | - |" in table
    assert "21.50" in table and "first 100 examples" in table
