from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract selected fields from generation outputs")
    parser.add_argument(
        "--input",
        type=str,
        default="/home/cqjtu/NLP-Group/LZH/T3GRAG/GRAG/generation_outputs.jsonl",
        help="Path to generation outputs jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/cqjtu/NLP-Group/LZH/T3GRAG/GRAG/generation_outputs_selected.jsonl",
        help="Path to output jsonl with selected fields",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out_handle:
        for item in read_jsonl(input_path):
            if "EM" in item and "F1" in item:
                continue
            payload = {
                "doc_id": item.get("doc_id"),
                "question": item.get("question"),
                "answer": item.get("answer"),
                "generate_answer": item.get("generate_answer"),
                "em": item.get("em"),
                "f1": item.get("f1"),
            }
            out_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
