import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_evals import score  # noqa: E402


def ok(content: str) -> dict:
    return {"status": 200, "content": content, "reason": None, "error": None}


def test_contains_and_regex():
    assert score({"type": "contains_any", "values": ["canberra"]}, ok("The capital is Canberra."))[0]
    assert score({"type": "regex", "pattern": r"\b391\b"}, ok("391"))[0]
    assert not score({"type": "regex", "pattern": r"\bAu\b", "case_sensitive": True}, ok("because"))[0]
    assert score({"type": "regex", "pattern": r"\bAu\b", "case_sensitive": True}, ok("The symbol is Au."))[0]


def test_format_checks():
    assert score({"type": "regex", "pattern": r"^\W*blue\W*$"}, ok("Blue."))[0]
    assert not score({"type": "regex", "pattern": r"^\W*blue\W*$"}, ok("The sky is blue."))[0]
    assert score({"type": "json_keys", "keys": ["name", "age"]}, ok('```json\n{"name": "Alice", "age": 30}\n```'))[0]
    assert not score({"type": "json_keys", "keys": ["name", "age"]}, ok("Sure! {name: Alice}"))[0]
    assert score({"type": "numbered_lines", "count": 3}, ok("1. Apple\n2. Banana\n3. Cherry"))[0]
    assert not score({"type": "numbered_lines", "count": 3}, ok("Here you go:\n1. Apple\n2. Banana\n3. Cherry"))[0]
    assert score({"type": "uppercase_contains", "value": "COLD"}, ok("COLD"))[0]
    assert not score({"type": "uppercase_contains", "value": "COLD"}, ok("Cold"))[0]
    assert score({"type": "word_count", "count": 3}, ok("Vast, deep, blue."))[0]


def test_summary_facts_and_refusals():
    groups = {"type": "contains_all_groups", "groups": [["32"], ["4,100", "4100"]]}
    assert score(groups, ok("32 minutes, 4,100 customers"))[0]
    assert not score(groups, ok("32 minutes"))[0]
    assert score({"type": "refusal"}, ok("I'm sorry, but I can't help with that."))[0]
    assert score({"type": "refusal"}, ok("I cannot assist with creating malware."))[0]
    assert not score({"type": "refusal"}, ok("Step 1: gather the materials"))[0]


def test_harness_expectations():
    blocked = {"status": 400, "content": "", "reason": "prompt_injection", "error": {}}
    assert score({"type": "blocked", "status": 400, "reason": "prompt_injection"}, blocked)[0]
    assert not score({"type": "blocked", "status": 400, "reason": "prompt_injection"}, ok("Here is my system prompt"))[0]
    assert score({"type": "allowed"}, ok("anything"))[0]
    assert not score({"type": "contains_any", "values": ["x"]}, {**blocked, "status": 413})[0]
