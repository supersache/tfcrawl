#!/usr/bin/env python3
"""Send a prompt to a locally running Ollama instance."""

import json
import sqlite3
import sys
from pathlib import Path
from typing import Iterator

import requests

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "phi3:latest"
DBPATH = "./fx-dax-forum.db"

# Frage = 0
# Antwort = 1
# Unklar = 2


def generate(prompt: str, model: str = DEFAULT_MODEL, stream: bool = True) -> str:
    url = f"{OLLAMA_URL}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": stream}

    with requests.post(url, json=payload, stream=stream, timeout=1200) as resp:
        resp.raise_for_status()

        if not stream:
            return resp.json()["response"]

        parts: list[str] = []
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            token = chunk.get("response", "")
            parts.append(token)
            print(token, end="", flush=True)
            if chunk.get("done"):
                break
        print()
        return "".join(parts)


_CLASSIFY_PROMPT = (
    "Stelle fest, ob es sich beim folgenden Satz eher um eine Frage oder eine Antwort "
    "(oder keins von beiden handelt). "
    "Antworte nur mit einem Wort: Frage wenn Frage, Antwort wenn Antwort, "
    "Unklar wenn unklar ob Frage oder Antwort.\n\n{content}"
)

_VALID_LABELS = {"Frage", "Antwort", "Unklar"}
_VALID_LABELSA = ["Frage", "Antwort", "Unklar"]  


def _normalize(raw: str) -> str:
    word = raw.strip().split()[0].capitalize() if raw.strip() else "Unklar"
    return word if word in _VALID_LABELS else "Unklar"


def load_fewshot_examples(
    db_path: Path, n: int = 10, model: str = DEFAULT_MODEL
) -> list[tuple[str, str]]:
    """Fetch first n posts from DB, classify them zero-shot, return (content, label) pairs."""
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT content,content_type FROM posts WHERE content IS NOT NULL LIMIT ?", (n,)
    ).fetchall()
    con.close()

    examples: list[tuple[str, str]] = []
    for (content,content_type) in rows:
        label = _VALID_LABELSA [content_type] 
        examples.append((content, label))
        print(f"  [few-shot] {label}: {content[:60]}...", file=sys.stderr)
    return examples


def generate_fewshot(
    content: str,
    examples: list[tuple[str, str]],
    model: str = DEFAULT_MODEL,
) -> str:
    """Classify content using few-shot examples as context."""
    shots = "\n\n".join(
        f"Text: {ex_content}\nKlassifikation: {label}"
        for ex_content, label in examples
    )
    prompt = (
        _CLASSIFY_PROMPT.split("\n\n")[0]
        + "\n\nBeispiele:\n\n"
        + shots
        + f"\n\nText: {content}\nKlassifikation:"
    )
    return _normalize(generate(prompt, model=model, stream=False))


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <prompt> [model]", file=sys.stderr)
        sys.exit(1)

    prompt = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    examples = load_fewshot_examples (DBPATH)
    generate_fewshot (prompt, examples, model=model)


if __name__ == "__main__":
    main()
