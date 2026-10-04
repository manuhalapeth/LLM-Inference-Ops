"""Deterministic long documents for KV cache experiments.

Each document starts with its own ID and is built from words drawn with a
per-document random seed, so no two documents share a prefix (prefix caching
works on blocks counted from the start of the prompt) and the same document
is byte-identical every time it's generated.

Standard library only.
"""

import random

# Plain English words, so the text tokenizes like real prose (about 1 Qwen token per word).
WORDS = """
the a an and or but of to in on at by for with from about into over after before
during under between through while because although since until unless where when
system service request response server client network latency memory cache model
token user data team engineer report incident customer product release version
traffic error queue batch window process thread message event metric dashboard
alert budget policy review summary decision meeting plan quarter week month year
day hour minute second first last next early late fast slow large small high low
new old long short simple complex stable busy idle open closed ready waiting
running failed passed changed moved added removed improved reduced increased
measured tested deployed restored checked updated shared stored loaded sent read
written built started stopped paused resumed found missed kept dropped held used
people group office city country market price cost value rate share growth risk
north south east west river mountain forest garden ocean island village bridge
road train station airport harbor market library school hospital museum theater
""".split()


def make_document(doc_id: int, words: int = 5900) -> str:
    rng = random.Random(f"doc-{doc_id}")
    body = " ".join(rng.choice(WORDS) for _ in range(words))
    return f"Document {doc_id:03d}.\n\n{body}"


def make_messages(doc_id: int, words: int = 5900) -> list[dict]:
    return [{
        "role": "user",
        "content": f"{make_document(doc_id, words)}\n\nIn one sentence, what is document {doc_id:03d} about?",
    }]
