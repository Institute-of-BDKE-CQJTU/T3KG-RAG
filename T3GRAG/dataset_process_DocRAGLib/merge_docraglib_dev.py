import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_merged_dev_set(documents: List[Dict[str, Any]], dev_qa: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    doc_map: Dict[str, Dict[str, Any]] = {}
    for doc in documents:
        uid = doc.get("uid")
        if uid:
            doc_map[uid] = doc

    merged: List[Dict[str, Any]] = []
    missing_docs = 0

    for qa_item in dev_qa:
        doc_uid = qa_item.get("document_uid")
        doc = doc_map.get(doc_uid)

        if doc is None:
            missing_docs += 1
            continue

        merged_item = {
            "uid": doc_uid,
            "paragraphs": doc.get("paragraphs", []),
            "tables": doc.get("tables", []),
            "qa": {
                "document_uid": doc_uid,
                "question_uid": qa_item.get("question_uid"),
                "question": qa_item.get("question"),
                "answer": qa_item.get("answer"),
            },
        }
        merged.append(merged_item)

    if missing_docs > 0:
        print(f"[Warning] {missing_docs} QA items skipped because corresponding document_uid was not found.")

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge DocRAGLib documents and dev QA into a unified validation set.")
    base_dir = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--documents",
        type=Path,
        default=base_dir / "dataset" / "DocRAGLib" / "Documents.json",
        help="Path to Documents.json",
    )
    parser.add_argument(
        "--dev_qa",
        type=Path,
        default=base_dir / "dataset" / "DocRAGLib" / "dev_qa.json",
        help="Path to dev_qa.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base_dir / "dataset" / "DocRAGLib_outputs" / "docraglib_dev_merged.json",
        help="Output merged json path",
    )
    args = parser.parse_args()

    documents = load_json(args.documents)
    dev_qa = load_json(args.dev_qa)

    merged = build_merged_dev_set(documents, dev_qa)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Documents: {len(documents)}")
    print(f"Dev QA: {len(dev_qa)}")
    print(f"Merged samples: {len(merged)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
