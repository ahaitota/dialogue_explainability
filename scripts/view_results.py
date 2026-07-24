"""Render results JSONL into readable Markdown.

    # a whole folder → one .md per file under <folder>/_readable/
    uv run python scripts/view_results.py results/step_a

    # a single file, filtered, printed to the terminal
    uv run python scripts/view_results.py results/logic_masking/train_ticket_price-Qwen3.5-4B-dialogue-scale2-add.jsonl --ids 3 7 --print

Open the generated .md in VS Code's Markdown preview (the giant system prompt is
collapsed). No model needed — pure formatting.
"""
import argparse
from pathlib import Path

from dialexp.readable import render_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Render results JSONL into readable Markdown")
    parser.add_argument("path", help="A results .jsonl file or a folder of them")
    parser.add_argument("--ids", nargs="+", type=int, help="Only these example ids")
    parser.add_argument("--limit", type=int, help="Max rows per file")
    parser.add_argument("--print", dest="to_stdout", action="store_true", help="Print instead of writing .md")
    parser.add_argument("--out", help="Output directory (default: <path>/_readable)")
    args = parser.parse_args()

    path = Path(args.path)
    ids = set(args.ids) if args.ids else None
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    if not files:
        print(f"No .jsonl files found at {path}")
        return

    if args.to_stdout:
        for f in files:
            print(render_file(f, ids=ids, limit=args.limit))
        return

    out_dir = Path(args.out) if args.out else (path if path.is_dir() else path.parent) / "_readable"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        md = render_file(f, ids=ids, limit=args.limit)
        out_file = out_dir / (f.stem + ".md")
        out_file.write_text(md)
        print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
