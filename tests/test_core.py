"""Offline unit tests: answer parsing, grading, domain routing and the voting/reflection logic.

No model server is needed. Calls to the model are replaced with fakes.
"""
import src.strategies as strategies
from src.agent import Agent
from src.evaluator import extract_number, grade_answer
from src.strategies import extract_answer_candidate, normalize_text


# ---------------------------------------------------------------- answer parsing
def test_boxed_answer_is_preferred():
    assert extract_answer_candidate("Work... so the result is \\boxed{869}. Answer: 12") == "869"


def test_answer_prefix_is_used_when_there_is_no_box():
    assert extract_answer_candidate("Some reasoning.\nAnswer: Paris") == "paris"


def test_plain_text_is_normalized():
    assert normalize_text("  Hello,   World!! ") == "hello world"


def test_extract_number_takes_the_last_number():
    assert extract_number("first 12, then 7, finally \\boxed{391}") == "391"
    assert extract_number("no digits here") is None


# ---------------------------------------------------------------- grading
def test_math_answers_match_numerically():
    assert grade_answer("391", "The answer is \\boxed{391.0}", "math")
    assert not grade_answer("391", "\\boxed{381}", "math")


def test_non_math_answers_match_as_text():
    assert grade_answer("Paris", "  paris ", "general")
    assert not grade_answer("Paris", "London", "general")


# ---------------------------------------------------------------- routing
def test_domain_is_inferred_from_the_prompt():
    agent = Agent()
    assert agent._infer_domain("Write a Python function that reverses a list") == "coding"
    assert agent._infer_domain("Calculate 17 * 23") == "math"
    assert agent._infer_domain("Who wrote Pride and Prejudice?") == "general"


# ---------------------------------------------------------------- self consistency
def test_self_consistency_returns_the_majority_answer(monkeypatch):
    replies = iter(["so \\boxed{391}", "so \\boxed{381}", "so \\boxed{391}", "so \\boxed{391}", "so \\boxed{17}"])
    monkeypatch.setattr(strategies, "chain_of_thought", lambda *a, **k: next(replies))
    assert strategies.self_consistency("17 * 23?", n=5, domain="math") == "so \\boxed{391}"


# ---------------------------------------------------------------- reflection
def test_reflection_stops_as_soon_as_the_critique_passes(monkeypatch):
    calls = []

    def fake_model(prompt, system="", model="", **kwargs):
        calls.append(prompt)
        return {"ok": True, "text": "CORRECT. The algebra checks out."}

    monkeypatch.setattr(strategies, "call_model_chat_completions", fake_model)
    out = strategies.reflection_strategy("17 * 23?", initial_ans="\\boxed{391}", max_steps=3)
    assert out == "\\boxed{391}"
    assert len(calls) == 1  # one critique, no fix needed


def test_reflection_uses_the_fixed_answer_when_the_critique_fails(monkeypatch):
    replies = iter([
        {"ok": True, "text": "This is wrong: 17 * 23 is not 381."},
        {"ok": True, "text": "Fixed: \\boxed{391}"},
        {"ok": True, "text": "CORRECT"},
    ])
    monkeypatch.setattr(strategies, "call_model_chat_completions", lambda *a, **k: next(replies))
    out = strategies.reflection_strategy("17 * 23?", initial_ans="\\boxed{381}", max_steps=3)
    assert out == "Fixed: \\boxed{391}"
