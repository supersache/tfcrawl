#!/usr/bin/env python3
"""Classify each post as Frage/Antwort/Unklar using a local Ollama model."""

import sqlite3
import sys
from pathlib import Path

from ollama_client import generate
from parse_posts import DB_PATH

PROMPT_TEMPLATE = (
    "Stelle fest, ob es sich beim folgenden Satz eher um eine Frage oder eine Antwort "
    "(oder keins von beiden handelt). "
    "Antworte nur mit einem Wort: Frage wenn Frage, Antwort wenn Antwort, "
    "Unklar wenn unklar ob Frage oder Antwort.\n\n{content}"
)

VALID = {"Frage", "Antwort", "Unklar"}


def classify(content: str) -> str:
    prompt = PROMPT_TEMPLATE.format(content=content)
    raw = generate(prompt, stream=False).strip()
    # Normalize: take first word, capitalize
    word = raw.split()[0].capitalize() if raw else "Unklar"
    return word if word in VALID else "Unklar"


def main() -> None:
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DB_PATH
    con = sqlite3.connect(db_path)
    rows = con.execute("SELECT post_id, content FROM posts WHERE content IS NOT NULL").fetchall()
    con.close()

    for post_id, content in rows:
        result = classify(content)
        print(f"{post_id} -> {result}")


if __name__ == "__main__":
    main()
