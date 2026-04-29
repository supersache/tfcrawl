#!/usr/bin/env python3
"""Parse WPForo HTML and extract posts as JSON."""

import json
import re
import sqlite3
import sys
from pathlib import Path

from bs4 import BeautifulSoup

DB_PATH = Path("fx-dax-forum.db")

_DDL = """
CREATE TABLE IF NOT EXISTS posts (
    post_id        INTEGER PRIMARY KEY,
    user_id        INTEGER,
    parent_post_id INTEGER REFERENCES posts(post_id),
    date           TEXT,
    content        TEXT
);
"""

_DE_MONTHS = {
    "januar": 1, "februar": 2, "märz": 3, "april": 4,
    "mai": 5, "juni": 6, "juli": 7, "august": 8,
    "september": 9, "oktober": 10, "november": 11, "dezember": 12,
}

def _parse_date(raw: str) -> str | None:
    """Convert '1. September 2025 04:21' -> '2025-09-01 04:21'."""
    m = re.match(
        r"(\d{1,2})\.\s+(\w+)\s+(\d{4})\s+(\d{1,2}):(\d{2})",
        raw.strip(),
    )
    if not m:
        return None
    day, month_name, year, hour, minute = m.groups()
    month = _DE_MONTHS.get(month_name.lower())
    if not month:
        return None
    return f"{int(year):04d}-{month:02d}-{int(day):02d} {int(hour):02d}:{minute}"


def parse_posts(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    posts = []

    for post_div in soup.find_all("div", attrs={"data-postid": True}):
        post_id = int(post_div["data-postid"])
        user_id = int(post_div["data-userid"]) if post_div.get("data-userid") else None

        # Look for enclosing wpf-post-replies div to find parent
        parent_id = None
        for ancestor in post_div.parents:
            if ancestor.name == "div" and ancestor.get("id", "").startswith("wpf-post-replies-"):
                match = re.search(r"wpf-post-replies-(\d+)", ancestor["id"])
                if match:
                    parent_id = int(match.group(1))
                break

        date_div = post_div.find("div", class_="wpf-post-date")
        date = _parse_date(date_div.get_text()) if date_div else None

        content_div = post_div.find("div", class_="wpforo-post-content")
        content = content_div.get_text(separator="\n", strip=True) if content_div else None

        posts.append({
            "post_id": post_id,
            "user_id": user_id,
            "parent_post_id": parent_id,
            "date": date,
            "content": content,
        })

    return posts


def save_posts(posts: list[dict], db_path: Path = DB_PATH) -> int:
    con = sqlite3.connect(db_path)
    with con:
        con.executescript(_DDL)
        cur = con.executemany(
            """
            INSERT INTO posts (post_id, user_id, parent_post_id, date, content)
            VALUES (:post_id, :user_id, :parent_post_id, :date, :content)
            ON CONFLICT(post_id) DO UPDATE SET
                user_id        = excluded.user_id,
                parent_post_id = excluded.parent_post_id,
                date           = excluded.date,
                content        = excluded.content
            """,
            posts,
        )
    con.close()
    return cur.rowcount


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not path or not path.exists():
        print(f"Usage: {sys.argv[0]} <html-file>", file=sys.stderr)
        sys.exit(1)

    html = path.read_text(encoding="utf-8", errors="replace")
    posts = parse_posts(html)
    n = save_posts(posts)
    print(f"{len(posts)} posts parsed, {n} rows affected → {DB_PATH}", file=sys.stderr)
    print(json.dumps(posts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
