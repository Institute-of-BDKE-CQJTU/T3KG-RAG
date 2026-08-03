"""
Graph RAG 向量构建脚本
- 属性拼接
- 批量向量化
- 导出向量到本地文件 / FAISS（用于快速检索）
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import faiss  # type: ignore[import]
import numpy as np  # type: ignore[import]
from sentence_transformers import SentenceTransformer  # type: ignore[import]
from tqdm import tqdm
import torch  # type: ignore[import]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Neo4j.connect_neo4j import Neo4jConnection


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "/home/cqjtu/Data/sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 10000
EMBED_BATCH_SIZE = 256
EMBED_DIM = 768
OUTPUT_ROOT = Path(__file__).resolve().parent / "output_vector"


def format_cell_text(record: Dict[str, Any]) -> str:
    # Cell 向量文本优先且主要使用 TableDescription.description
    table_description = (record.get("table_description") or "").strip()
    if table_description:
        return table_description

    # 若存在脏数据或未建立描述关系，保留原始回退逻辑，避免出现空文本
    table_id = record.get("table_id")
    row_path_str = record.get("row_path_str")
    col_path_str = record.get("col_path_str")
    row_group_path = record.get("row_group_path")
    header_path = record.get("header_path")
    value = record.get("value")
    numeric_value = record.get("numeric_value")
    unit = record.get("unit")

    value_text = value if value is not None else numeric_value

    parts = []
    if table_id is not None:
        parts.append(f"Table {table_id}")

    row_text = row_path_str or row_group_path
    col_text = col_path_str or header_path

    if row_text:
        parts.append(f"Row: {row_text}")
    if col_text:
        parts.append(f"Col: {col_text}")
    if value_text is not None:
        if unit:
            parts.append(f"Value: {value_text} {unit}")
        else:
            parts.append(f"Value: {value_text}")

    return ", ".join(parts)


def fetch_batch(session, label: str, skip: int, limit: int) -> List[Dict[str, Any]]:
    if label == "Cell":
        query = """
        MATCH (n:Cell)
        OPTIONAL MATCH (n)-[]-(td:TableDescription)
        WITH n,
             [x IN collect(td)
              WHERE x IS NOT NULL
                AND toString(coalesce(x.doc_id, "")) = toString(coalesce(n.doc_id, ""))
                AND toString(coalesce(x.table_id, "")) = toString(coalesce(n.table_id, ""))
                AND toString(coalesce(x.row, "")) = toString(coalesce(n.row, ""))
                AND toString(coalesce(x.col, "")) = toString(coalesce(n.col, ""))
             ] AS exact_tds
        RETURN n.id AS id,
               n.doc_id AS doc_id,
               n.value AS value,
               n.numeric_value AS numeric_value,
               n.row_path_str AS row_path_str,
               n.col_path_str AS col_path_str,
               n.header_path AS header_path,
               n.row_group_path AS row_group_path,
               n.table_id AS table_id,
               n.row AS row,
               n.col AS col,
               n.unit AS unit,
               n.desc_key AS desc_key,
               coalesce(head(exact_tds).description, "") AS table_description,
               coalesce(head(exact_tds).desc_key, "") AS table_desc_key,
               size(exact_tds) > 0 AS has_exact_desc_match
        ORDER BY n.id
        SKIP $skip LIMIT $limit
        """
    else:
        query = """
        MATCH (n:Paragraph)
        RETURN n.id AS id,
               n.doc_id AS doc_id,
               n.content AS content
        ORDER BY n.id
        SKIP $skip LIMIT $limit
        """

    result = session.run(query, {"skip": skip, "limit": limit})
    return [record.data() for record in result]


def append_vector_ids(path: Path, records: List[Dict[str, Any]]) -> None:
    if not records:
        return

    file_exists = path.exists()
    with path.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    if not file_exists:
        logger.info("已创建 ID 文件: %s", path)


def save_vectors_npy(path: Path, vectors: np.ndarray) -> None:
    np.save(path, vectors)


def ensure_dirs(output_root: Path = OUTPUT_ROOT) -> None:
    output_root.mkdir(parents=True, exist_ok=True)


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(EMBED_DIM)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    return index


def save_faiss_index(index: faiss.Index, path: Path) -> None:
    faiss.write_index(index, str(path))


def merge_faiss_index(index_path: Path, vectors: np.ndarray) -> None:
    if index_path.exists():
        index = faiss.read_index(str(index_path))
        faiss.normalize_L2(vectors)
        index.add(vectors)
    else:
        index = build_faiss_index(vectors)
    save_faiss_index(index, index_path)


def log_gpu_memory(device: str, stage: str) -> None:
    if device != "cuda":
        return

    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    logger.info("GPU 内存(%s) - allocated: %.1f MB, reserved: %.1f MB", stage, allocated, reserved)


def process_label(driver, label: str, model: SentenceTransformer, device: str, output_dir: Path) -> None:
    export_dir = output_dir / label
    faiss_dir = export_dir / "faiss"
    ensure_dirs()
    export_dir.mkdir(parents=True, exist_ok=True)
    faiss_dir.mkdir(parents=True, exist_ok=True)

    count_query = f"MATCH (n:{label}) RETURN count(n) AS count"
    with driver.session() as session:
        total = session.run(count_query).single()["count"]

    if total == 0:
        logger.info("%s 节点数量为 0，跳过。", label)
        return

    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info("开始处理 %s，共 %d 条，%d 个批次。", label, total, total_batches)

    global_index_offset = 0
    with driver.session() as session:
        for batch_idx in tqdm(range(total_batches), desc=f"{label} batches"):
            skip = batch_idx * BATCH_SIZE
            start_time = time.time()

            logger.info("%s 批次 %d/%d 开始，skip=%d, limit=%d", label, batch_idx + 1, total_batches, skip, BATCH_SIZE)
            log_gpu_memory(device, "batch_start")

            rows = fetch_batch(session, label, skip, BATCH_SIZE)
            if not rows:
                logger.warning("%s 批次 %d 拉取为空。", label, batch_idx + 1)
                continue

            if label == "Cell":
                exact_match_count = sum(1 for row in rows if row.get("has_exact_desc_match"))
                missing_desc_count = sum(1 for row in rows if not (row.get("table_description") or "").strip())
                logger.info(
                    "Cell 批次 %d/%d 描述匹配统计: exact=%d/%d, missing_description=%d",
                    batch_idx + 1,
                    total_batches,
                    exact_match_count,
                    len(rows),
                    missing_desc_count,
                )
                embedding_texts = [format_cell_text(row) for row in rows]
            else:
                embedding_texts = [row.get("content") or "" for row in rows]

            if not any(text.strip() for text in embedding_texts):
                logger.warning("%s 批次 %d 文本为空，跳过。", label, batch_idx + 1)
                continue

            valid_indices = [idx for idx, text in enumerate(embedding_texts) if text.strip()]
            valid_texts = [embedding_texts[idx] for idx in valid_indices]

            log_gpu_memory(device, "before_encode")
            vectors = model.encode(
                valid_texts,
                batch_size=EMBED_BATCH_SIZE if device == "cuda" else min(64, EMBED_BATCH_SIZE),
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            log_gpu_memory(device, "after_encode")

            records: List[Dict[str, Any]] = []
            for item_idx, idx in enumerate(valid_indices, start=1):
                row = rows[idx]
                records.append(
                    {
                        "index": global_index_offset + item_idx - 1,
                        "node_id": row.get("id"),
                        "doc_id": row.get("doc_id") or "",
                        "label": label,
                    }
                )

                if item_idx % 100 == 0:
                    logger.info("%s 批次 %d/%d 已生成 %d 条向量", label, batch_idx + 1, total_batches, item_idx)

            if not records:
                logger.warning("%s 批次 %d 无有效文本，未导出。", label, batch_idx + 1)
                continue

            id_path = export_dir / f"{label.lower()}_ids.jsonl"
            id_export_start = time.time()
            append_vector_ids(id_path, records)
            id_export_elapsed = time.time() - id_export_start
            logger.info("%s 批次 %d/%d 导出 %d 条 ID 到 %s，耗时 %.2fs", label, batch_idx + 1, total_batches, len(records), id_path, id_export_elapsed)

            vectors_path = export_dir / f"{label.lower()}_vectors_batch_{batch_idx + 1}.npy"
            save_vectors_npy(vectors_path, vectors)
            logger.info("%s 批次 %d/%d 保存向量到 %s", label, batch_idx + 1, total_batches, vectors_path)

            global_index_offset += len(records)

            index_path = faiss_dir / f"{label.lower()}.index"
            faiss_start = time.time()
            merge_faiss_index(index_path, vectors)
            faiss_elapsed = time.time() - faiss_start
            logger.info("%s 批次 %d/%d 更新 FAISS 索引 %s，耗时 %.2fs", label, batch_idx + 1, total_batches, index_path, faiss_elapsed)

            log_gpu_memory(device, "after_export")
            elapsed = time.time() - start_time
            logger.info("%s 批次 %d/%d 完成，导出 %d 条，耗时 %.2fs", label, batch_idx + 1, total_batches, len(records), elapsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graph RAG 向量构建")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--MultiHiertt", "-MultiHiertt", action="store_true", help="处理 MultiHiertt 数据集")
    mode.add_argument("--DocRAGLib", "-DocRAGLib", action="store_true", help="处理 DocRAGLib 数据集")
    parser.add_argument("--output_root", type=Path, default=OUTPUT_ROOT, help="输出根目录")
    parser.add_argument("--device", type=str, default=None, help="向量构建设备，默认自动选择 cuda 或 cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root: Path = args.output_root
    ensure_dirs(output_root)

    mode = "DocRAGLib" if args.DocRAGLib else "MultiHiertt"
    dataset_output_dir = output_root / mode
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("使用设备: %s", device)

    neo4j = Neo4jConnection(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password123",
    )

    try:
        if not neo4j.test_connection():
            return

        model = SentenceTransformer(MODEL_PATH, device=device)
        model.max_seq_length = 384

        process_label(neo4j.driver, "Cell", model, device, dataset_output_dir)
        process_label(neo4j.driver, "Paragraph", model, device, dataset_output_dir)

        logger.info("导出完成：JSONL ID 文件 + .npy 向量文件 + FAISS 索引已生成。")
        logger.info("输出目录：%s", dataset_output_dir)

    except Exception as exc:
        logger.exception("执行失败: %s", exc)
    finally:
        neo4j.close()


if __name__ == "__main__":
    main()
