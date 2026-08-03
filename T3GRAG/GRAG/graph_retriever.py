from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np
from neo4j import Driver, GraphDatabase
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GRAG.build_graph_vector_index import format_cell_text

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


@dataclass
class MappingRecord:
    node_id: str
    doc_id: str
    label: str


@dataclass
class EvaluationItem:
    question_id: str
    text: str
    doc_id: str
    question_type: str
    program: str
    answer: str
    true_evidence_keys: List[str]


class HybridGraphRetriever:
    def __init__(
        self,
        model_path: str,
        cell_index_path: str,
        paragraph_index_path: str,
        cell_mapping_path: str,
        paragraph_mapping_path: str,
        driver: Driver,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> None:
        self.model = SentenceTransformer(model_path)
        self.cell_index = faiss.read_index(cell_index_path)
        self.paragraph_index = faiss.read_index(paragraph_index_path)
        self.cell_mapping = self._load_mapping(cell_mapping_path)
        self.paragraph_mapping = self._load_mapping(paragraph_mapping_path)
        self.cell_doc_index = self._build_doc_index_map(self.cell_mapping)
        self.paragraph_doc_index = self._build_doc_index_map(self.paragraph_mapping)
        self.driver = driver

        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        self.node_text_cache: Dict[str, str] = {}
        self.doc_term_freqs: Dict[str, Counter[str]] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.doc_avg_len: Dict[str, float] = {}
        self.doc_idf: Dict[str, Dict[str, float]] = {}

        self._build_sparse_cache()

    @staticmethod
    def _build_doc_index_map(mapping: List[MappingRecord]) -> Dict[str, List[int]]:
        doc_map: Dict[str, List[int]] = {}
        for idx, record in enumerate(mapping):
            if not record.doc_id:
                continue
            doc_map.setdefault(record.doc_id, []).append(idx)
        return doc_map

    @staticmethod
    def _load_mapping(path: str) -> List[MappingRecord]:
        records: List[MappingRecord] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                records.append(
                    MappingRecord(
                        node_id=str(payload.get("node_id", "")),
                        doc_id=str(payload.get("doc_id", "")),
                        label=str(payload.get("label", "")),
                    )
                )
        return records

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        if not text:
            return []
        return re.findall(r"\w+", text.lower())

    @staticmethod
    def _minmax_normalize(scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        min_score = min(scores.values())
        max_score = max(scores.values())
        if math.isclose(max_score, min_score):
            return {key: 1.0 for key in scores}
        return {key: (value - min_score) / (max_score - min_score) for key, value in scores.items()}

    def _build_sparse_cache(self) -> None:
        with self.driver.session() as session:
            para_result = session.run(
                """
                MATCH (p:Paragraph)
                RETURN p.id AS node_id,
                       p.doc_id AS doc_id,
                       coalesce(p.text, p.content, p.value, "") AS content
                """
            )
            for record in para_result:
                node_id = record.get("node_id")
                content = (record.get("content") or "").strip()
                if node_id is None:
                    continue
                self.node_text_cache[str(node_id)] = content
                tokens = self._tokenize(content)
                self.doc_lengths[str(node_id)] = len(tokens)
                self.doc_term_freqs[str(node_id)] = Counter(tokens)

            cell_result = session.run(
                """
                MATCH (c:Cell)
                CALL {
                    WITH c
                    OPTIONAL MATCH (c)-[]-(td:TableDescription)
                    WHERE td IS NOT NULL
                      AND toString(coalesce(td.doc_id, "")) = toString(coalesce(c.doc_id, ""))
                      AND toString(coalesce(td.table_id, "")) = toString(coalesce(c.table_id, ""))
                      AND toString(coalesce(td.row, "")) = toString(coalesce(c.row, ""))
                      AND toString(coalesce(td.col, "")) = toString(coalesce(c.col, ""))
                    RETURN collect(td) AS exact_tds
                }
                RETURN c.id AS node_id,
                       c.doc_id AS doc_id,
                       c.table_id AS table_id,
                       c.row_path_str AS row_path_str,
                       c.col_path_str AS col_path_str,
                       c.row_group_path AS row_group_path,
                       c.header_path AS header_path,
                       c.value AS value,
                       c.numeric_value AS numeric_value,
                       c.unit AS unit,
                       coalesce(head(exact_tds).description, "") AS table_description
                """
            )
            for record in cell_result:
                node_id = record.get("node_id")
                if node_id is None:
                    continue
                content = format_cell_text(record)

                self.node_text_cache[str(node_id)] = content
                tokens = self._tokenize(content)
                self.doc_lengths[str(node_id)] = len(tokens)
                self.doc_term_freqs[str(node_id)] = Counter(tokens)

        all_doc_ids = list(self.doc_term_freqs.keys())
        total_docs = len(all_doc_ids)
        self.doc_avg_len = {
            "global": (sum(self.doc_lengths.values()) / total_docs) if total_docs else 0.0
        }

        doc_freq: Counter[str] = Counter()
        for node_id in all_doc_ids:
            for token in set(self.doc_term_freqs[node_id].keys()):
                doc_freq[token] += 1

        self.doc_idf = {"global": {}}
        for token, df in doc_freq.items():
            self.doc_idf["global"][token] = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)

    def _bm25_scores(self, node_ids: Sequence[str], query_text: str) -> Dict[str, float]:
        query_terms = self._tokenize(query_text)
        if not query_terms:
            return {node_id: 0.0 for node_id in node_ids}

        avgdl = self.doc_avg_len.get("global", 0.0) or 1.0
        idf_map = self.doc_idf.get("global", {})
        scores: Dict[str, float] = {}

        for node_id in node_ids:
            tf = self.doc_term_freqs.get(node_id, Counter())
            dl = self.doc_lengths.get(node_id, 0)
            score = 0.0
            for term in query_terms:
                freq = tf.get(term, 0)
                if freq <= 0:
                    continue
                idf = idf_map.get(term, 0.0)
                denom = freq + self.bm25_k1 * (1 - self.bm25_b + self.bm25_b * dl / avgdl)
                score += idf * (freq * (self.bm25_k1 + 1)) / denom
            scores[node_id] = score
        return scores

    def _encode(self, text: str) -> np.ndarray:
        vector = self.model.encode([text], normalize_embeddings=True)
        return np.asarray(vector, dtype="float32")

    def _faiss_search(
        self,
        index: faiss.Index,
        mapping: List[MappingRecord],
        query_vec: np.ndarray,
        doc_index_map: Dict[str, List[int]],
        target_doc_id: str,
        top_k: Optional[int] = None,
    ) -> List[Tuple[float, str]]:
        doc_indices = doc_index_map.get(target_doc_id, [])
        if not doc_indices:
            return []
        vectors = np.vstack([index.reconstruct(idx) for idx in doc_indices]).astype("float32")
        scores = np.dot(vectors, query_vec[0])
        if top_k is None:
            top_indices = np.argsort(scores)[::-1]
        else:
            top_k = min(top_k, len(doc_indices))
            top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(float(scores[i]), mapping[doc_indices[i]].node_id) for i in top_indices]

    def _fetch_paragraph_contexts(
        self, session, paragraph_node_ids: Sequence[str]
    ) -> Tuple[List[Dict[str, str]], List[str]]:
        if not paragraph_node_ids:
            return [], []
        result = session.run(
            """
            MATCH (p:Paragraph)
            WHERE p.id IN $ids
            RETURN p.id AS pid,
                   p.doc_id AS doc_id,
                   p.paragraph_index AS paragraph_index,
                   coalesce(p.text, p.content, p.value, "") AS content
            """,
            ids=list(paragraph_node_ids),
        )
        retrieved_records: List[Dict[str, str]] = []
        contexts: List[str] = []
        for record in result:
            pid = record["pid"]
            doc_id = record.get("doc_id")
            paragraph_index = record.get("paragraph_index")
            content = record["content"] or ""
            if pid is not None:
                retrieved_records.append(
                    {
                        "node_id": str(pid),
                        "doc_id": str(doc_id) if doc_id is not None else "",
                        "paragraph_index": str(paragraph_index) if paragraph_index is not None else "",
                    }
                )
            if content:
                contexts.append(content)
        return retrieved_records, contexts

    def _fetch_table_contexts(self, session, cell_node_ids: Sequence[str]) -> Tuple[List[Dict[str, str]], List[str]]:
        if not cell_node_ids:
            return [], []
        result = session.run(
            """
            MATCH (c:Cell)
            WHERE c.id IN $ids
            CALL {
                WITH c
                OPTIONAL MATCH (c)-[]-(td:TableDescription)
                WHERE td IS NOT NULL
                  AND toString(coalesce(td.doc_id, "")) = toString(coalesce(c.doc_id, ""))
                  AND toString(coalesce(td.table_id, "")) = toString(coalesce(c.table_id, ""))
                  AND toString(coalesce(td.row, "")) = toString(coalesce(c.row, ""))
                  AND toString(coalesce(td.col, "")) = toString(coalesce(c.col, ""))
                RETURN collect(td) AS exact_tds
            }
            RETURN c.id AS id,
                   c.doc_id AS doc_id,
                   c.table_id AS table_id,
                   c.row AS row,
                   coalesce(c.col, "") AS col,
                   c.row_path_str AS row_path_str,
                   c.col_path_str AS col_path_str,
                   c.row_group_path AS row_group_path,
                   c.header_path AS header_path,
                   c.value AS value,
                   c.numeric_value AS numeric_value,
                   c.unit AS unit,
                   coalesce(head(exact_tds).description, "") AS table_description
            """,
            ids=list(cell_node_ids),
        )
        retrieved_records: List[Dict[str, str]] = []
        contexts: List[str] = []
        for record in result:
            node_id = record["id"]
            doc_id = record.get("doc_id")
            table_id = record.get("table_id")
            row = record.get("row")
            col = record.get("col")

            if node_id is not None:
                retrieved_records.append(
                    {
                        "node_id": str(node_id),
                        "doc_id": str(doc_id) if doc_id is not None else "",
                        "table_id": str(table_id) if table_id is not None else "",
                        "row": str(row) if row is not None else "",
                        "col": str(col) if col is not None else "",
                    }
                )

            contexts.append(format_cell_text(record))
        return retrieved_records, contexts

    def _build_text_key(self, doc_id: str, paragraph_index: str) -> str:
        return f"{doc_id}::paragraph::{paragraph_index}"

    def _build_table_key(self, doc_id: str, table_id: str, row: str, col: str) -> str:
        return f"{doc_id}::table::{table_id}::row::{row}::col::{col}"

    def retrieve(self, question: str, target_doc_id: str, top_k: int = 50) -> Dict[str, List[str]]:
        _ = top_k  # 保留参数兼容旧调用，当前采用配额式召回
        kp = 20
        kc = 40

        query_vec = self._encode(question)
        paragraph_scored = self._faiss_search(
            self.paragraph_index,
            self.paragraph_mapping,
            query_vec,
            self.paragraph_doc_index,
            target_doc_id,
        )
        cell_scored = self._faiss_search(
            self.cell_index,
            self.cell_mapping,
            query_vec,
            self.cell_doc_index,
            target_doc_id,
        )

        para_dense_map = {node_id: score for score, node_id in paragraph_scored}
        cell_dense_map = {node_id: score for score, node_id in cell_scored}

        para_bm25_map = self._bm25_scores(list(para_dense_map.keys()), question)
        cell_bm25_map = self._bm25_scores(list(cell_dense_map.keys()), question)

        para_dense_norm = self._minmax_normalize(para_dense_map)
        cell_dense_norm = self._minmax_normalize(cell_dense_map)
        para_bm25_norm = self._minmax_normalize(para_bm25_map)
        cell_bm25_norm = self._minmax_normalize(cell_bm25_map)

        para_fused: List[Tuple[float, str]] = []
        for node_id in para_dense_map:
            fused = self.dense_weight * para_dense_norm.get(node_id, 0.0) + self.sparse_weight * para_bm25_norm.get(node_id, 0.0)
            para_fused.append((fused, node_id))

        cell_fused: List[Tuple[float, str]] = []
        for node_id in cell_dense_map:
            fused = self.dense_weight * cell_dense_norm.get(node_id, 0.0) + self.sparse_weight * cell_bm25_norm.get(node_id, 0.0)
            cell_fused.append((fused, node_id))

        para_fused.sort(key=lambda item: item[0], reverse=True)
        cell_fused.sort(key=lambda item: item[0], reverse=True)

        paragraph_node_ids = [node_id for _, node_id in para_fused[:kp]]
        cell_node_ids = [node_id for _, node_id in cell_fused[:kc]]

        retrieved_node_ids: List[str] = []
        retrieved_keys: List[str] = []
        context_texts: List[str] = []

        with self.driver.session() as session:
            paragraph_records, paragraph_contexts = self._fetch_paragraph_contexts(session, paragraph_node_ids)
            table_records, table_contexts = self._fetch_table_contexts(session, cell_node_ids)

        for record in paragraph_records:
            retrieved_node_ids.append(record["node_id"])
            doc_id = record.get("doc_id", "")
            paragraph_index = record.get("paragraph_index", "")
            if doc_id and paragraph_index:
                retrieved_keys.append(self._build_text_key(doc_id, paragraph_index))

        for record in table_records:
            retrieved_node_ids.append(record["node_id"])
            doc_id = record.get("doc_id", "")
            table_id = record.get("table_id", "")
            row = record.get("row", "")
            col = record.get("col", "")
            if doc_id and table_id and row and col:
                retrieved_keys.append(self._build_table_key(doc_id, table_id, row, col))

        context_texts.extend(paragraph_contexts)
        context_texts.extend(table_contexts)

        retrieved_node_ids = list(dict.fromkeys(retrieved_node_ids))
        retrieved_keys = list(dict.fromkeys(retrieved_keys))
        return {
            "context_texts": context_texts,
            "retrieved_node_ids": retrieved_node_ids,
            "retrieved_keys": retrieved_keys,
        }


class RetrieverEvaluator:
    def __init__(self, retriever: HybridGraphRetriever, driver: Driver) -> None:
        self.retriever = retriever
        self.driver = driver

    def fetch_evaluation_dataset(self) -> List[EvaluationItem]:
        items: List[EvaluationItem] = []
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (qa:QAInstance)-[:HAS_QUESTION]->(q:Question)
                OPTIONAL MATCH (qa)-[:HAS_QUESTION_TYPE|:QUESTION_TYPE|:TYPE|:HAS_TYPE]-(qt:QuestionType)
                OPTIONAL MATCH (qa)-[:HAS_PROGRAM|:PROGRAM|:HAS_FORMULA]-(p:Program)
                OPTIONAL MATCH (qa)-[:HAS_ANSWER|:ANSWER]-(a:Answer)
                OPTIONAL MATCH (qa)-[:HAS_TABLE_EVIDENCE|:TABLE_EVIDENCE]-(te:TableEvidence)
                OPTIONAL MATCH (qa)-[:HAS_TEXT_EVIDENCE|:TEXT_EVIDENCE]-(txe:TextEvidence)
                RETURN q.id AS qid,
                       q.text AS text,
                       q.doc_id AS doc_id,
                       coalesce(qt.type, qt.name, qt.label, "unknown") AS question_type,
                       coalesce(p.formula, p.text, p.value, "") AS program,
                       coalesce(a.answer, a.text, a.value, "") AS answer,
                       collect(DISTINCT {
                           doc_id: txe.doc_id,
                           paragraph_index: txe.paragraph_index
                       }) AS text_evidence,
                       collect(DISTINCT {
                           doc_id: te.doc_id,
                           table_id: te.table_id,
                           row: te.row,
                           col: te.col
                       }) AS table_evidence
                """
            )
            for record in result:
                evidence_keys: List[str] = []
                for text_item in record["text_evidence"]:
                    if text_item and text_item.get("doc_id") is not None and text_item.get("paragraph_index") is not None:
                        evidence_keys.append(
                            f"{text_item['doc_id']}::paragraph::{text_item['paragraph_index']}"
                        )
                for table_item in record["table_evidence"]:
                    if (
                        table_item
                        and table_item.get("doc_id") is not None
                        and table_item.get("table_id") is not None
                        and table_item.get("row") is not None
                        and table_item.get("col") is not None
                    ):
                        evidence_keys.append(
                            f"{table_item['doc_id']}::table::{table_item['table_id']}::row::{table_item['row']}::col::{table_item['col']}"
                        )
                items.append(
                    EvaluationItem(
                        question_id=str(record["qid"]),
                        text=record["text"] or "",
                        doc_id=record["doc_id"] or "",
                        question_type=record["question_type"] or "unknown",
                        program=record["program"] or "",
                        answer=record["answer"] or "",
                        true_evidence_keys=list(dict.fromkeys(evidence_keys)),
                    )
                )
        return items

    def evaluate(self, output_jsonl: Optional[str] = None) -> None:
        dataset = self.fetch_evaluation_dataset()
        total_questions = len(dataset)
        evaluated_questions = 0
        total_recall = 0.0
        total_precision = 0.0

        per_type: Dict[str, Dict[str, float]] = {}
        output_path = Path(output_jsonl) if output_jsonl else None
        output_handle = output_path.open("w", encoding="utf-8") if output_path else None

        skipped_questions = 0
        progress = tqdm(dataset, desc="Evaluating", unit="question", total=total_questions)

        try:
            for item in progress:
                if not item.true_evidence_keys:
                    skipped_questions += 1
                    progress.set_postfix(evaluated=evaluated_questions, skipped=skipped_questions)
                    continue
                evaluated_questions += 1
                output = self.retriever.retrieve(item.text, item.doc_id)
                retrieved_ids = output["retrieved_node_ids"]
                retrieved_keys = output["retrieved_keys"]
                retrieved_set = set(retrieved_keys)
                true_set = set(item.true_evidence_keys)

                intersection = len(retrieved_set.intersection(true_set))
                recall = intersection / len(true_set) if true_set else 0.0
                precision = intersection / len(retrieved_set) if retrieved_set else 0.0

                total_recall += recall
                total_precision += precision

                metrics = per_type.setdefault(item.question_type, {"count": 0, "recall": 0.0, "precision": 0.0})
                metrics["count"] += 1
                metrics["recall"] += recall
                metrics["precision"] += precision

                if output_handle:
                    output_payload = {
                        "question_id": item.question_id,
                        "doc_id": item.doc_id,
                        "question": item.text,
                        "question_type": item.question_type,
                        "program": item.program,
                        "answer": item.answer,
                        "true_evidence_keys": item.true_evidence_keys,
                        "retrieved_node_ids": retrieved_ids,
                        "retrieved_keys": retrieved_keys,
                        "context_texts": output["context_texts"],
                        "recall": recall,
                        "precision": precision,
                    }
                    output_handle.write(json.dumps(output_payload, ensure_ascii=False) + "\n")

                progress.set_postfix(evaluated=evaluated_questions, skipped=skipped_questions)
        finally:
            progress.close()
            if output_handle:
                output_handle.close()

        avg_recall = total_recall / evaluated_questions if evaluated_questions else 0.0
        avg_precision = total_precision / evaluated_questions if evaluated_questions else 0.0

        LOGGER.info("==== Hybrid Graph Retriever Evaluation ====")
        LOGGER.info("Total questions: %d", total_questions)
        LOGGER.info("Evaluated questions (with evidence): %d", evaluated_questions)
        LOGGER.info("Average Recall: %.4f", avg_recall)
        LOGGER.info("Average Precision: %.4f", avg_precision)
        LOGGER.info("---- Per Question Type ----")
        for qtype, metrics in sorted(per_type.items()):
            count = metrics["count"]
            type_recall = metrics["recall"] / count if count else 0.0
            type_precision = metrics["precision"] / count if count else 0.0
            LOGGER.info("%s | count=%d | recall=%.4f | precision=%.4f", qtype, count, type_recall, type_precision)


DEFAULT_MODEL_PATH = "/home/cqjtu/Data/sentence-transformers/all-mpnet-base-v2"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output_retriever"
DEFAULT_VECTOR_ROOT = Path(__file__).resolve().parent / "output_vector"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graph RAG hybrid retriever")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--MultiHiertt", "-MultiHiertt", action="store_true", help="运行 MultiHiertt")
    mode.add_argument("--DocRAGLib", "-DocRAGLib", action="store_true", help="运行 DocRAGLib")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vector_root", type=Path, default=DEFAULT_VECTOR_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--neo4j_uri", type=str, default="bolt://localhost:7687")
    parser.add_argument("--neo4j_user", type=str, default="neo4j")
    parser.add_argument("--neo4j_password", type=str, default="password123")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = "DocRAGLib" if args.DocRAGLib else "MultiHiertt"

    vector_dir = args.vector_root / mode
    cell_index_path = vector_dir / "Cell" / "faiss" / "cell.index"
    paragraph_index_path = vector_dir / "Paragraph" / "faiss" / "paragraph.index"
    cell_mapping_path = vector_dir / "Cell" / "cell_ids.jsonl"
    paragraph_mapping_path = vector_dir / "Paragraph" / "paragraph_ids.jsonl"
    output_dir = args.output_root / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl_path = output_dir / "retrieval_outputs.jsonl"

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    try:
        retriever = HybridGraphRetriever(
            model_path=args.model_path,
            cell_index_path=str(cell_index_path),
            paragraph_index_path=str(paragraph_index_path),
            cell_mapping_path=str(cell_mapping_path),
            paragraph_mapping_path=str(paragraph_mapping_path),
            driver=driver,
        )

        if args.DocRAGLib:
            with driver.session() as session:
                result = session.run(
                    """
                    MATCH (qa:QAInstance)-[:HAS_QUESTION]->(q:Question)
                    OPTIONAL MATCH (qa)-[:HAS_ANSWER|:ANSWER]-(a:Answer)
                    RETURN q.id AS qid,
                           q.text AS text,
                           q.doc_id AS doc_id,
                           coalesce(a.answer, a.text, a.value, "") AS answer
                    ORDER BY q.id
                    """
                )
                questions = [
                    {
                        "question_id": str(record["qid"]),
                        "question": record["text"] or "",
                        "doc_id": record["doc_id"] or "",
                        "answer": record["answer"] or "",
                    }
                    for record in result
                ]

            with output_jsonl_path.open("w", encoding="utf-8") as handle:
                for item in tqdm(questions, desc="Retrieving", unit="question"):
                    output = retriever.retrieve(item["question"], item["doc_id"])
                    payload = {
                        "question_id": item["question_id"],
                        "doc_id": item["doc_id"],
                        "question": item["question"],
                        "answer": item["answer"],
                        "retrieved_node_ids": output["retrieved_node_ids"],
                        "retrieved_keys": output["retrieved_keys"],
                        "context_texts": output["context_texts"],
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            LOGGER.info("DocRAGLib 检索完成，结果已保存到 %s", output_jsonl_path)
            return

        evaluator = RetrieverEvaluator(retriever, driver)
        evaluator.evaluate(output_jsonl=str(output_jsonl_path))
    finally:
        driver.close()


if __name__ == "__main__":
    main()
    driver.close()