#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行版本：将 MultiHiertt 数据集转为 Neo4j Admin Import CSV（按 label/type 拆分）。
保持原 T3G 图谱设计逻辑，使用多进程加速样本转换。

输出目录结构：
T3G_csv/outputs/{train|dev|test}/
  - nodes_{Label}.csv
  - edges_{TYPE}.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import networkx as nx
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from T3G.table_to_graph_converter import TableToGraphConverter


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def normalize_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return " | ".join(normalize_csv_value(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def collect_schema(graph: nx.MultiDiGraph) -> Tuple[Dict[str, Set[str]], Dict[str, List[Tuple[str, Dict[str, Any]]]], Dict[str, List[Tuple[str, str]]]]:
    label_props: Dict[str, Set[str]] = defaultdict(set)
    nodes_by_label: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    edges_by_type: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for node_id, node_data in graph.nodes(data=True):
        label = node_data.get("type", "Unknown")
        props = {k: v for k, v in node_data.items() if k != "type"}
        nodes_by_label[label].append((str(node_id), props))
        label_props[label].update(props.keys())

    for u, v, edge_data in graph.edges(data=True):
        edge_type = edge_data.get("edge_type", "RELATED_TO").upper()
        edges_by_type[edge_type].append((str(u), str(v)))

    return label_props, nodes_by_label, edges_by_type


def merge_schema(
    master_label_props: Dict[str, Set[str]],
    master_nodes_by_label: Dict[str, List[Tuple[str, Dict[str, Any]]]],
    master_edges_by_type: Dict[str, List[Tuple[str, str]]],
    part: Tuple[Dict[str, Set[str]], Dict[str, List[Tuple[str, Dict[str, Any]]]], Dict[str, List[Tuple[str, str]]]],
) -> None:
    label_props, nodes_by_label, edges_by_type = part

    for label, props in label_props.items():
        master_label_props[label].update(props)
    for label, nodes in nodes_by_label.items():
        master_nodes_by_label[label].extend(nodes)
    for edge_type, edges in edges_by_type.items():
        master_edges_by_type[edge_type].extend(edges)


def write_nodes_csv(output_dir: Path, label: str, nodes: List[Tuple[str, Dict[str, Any]]], props: List[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"nodes_{label}.csv"

    headers = ["id:ID"] + props + [":LABEL"]

    with file_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(
            f,
            quoting=csv.QUOTE_ALL,
            escapechar="\\",
            lineterminator="\n",
        )
        writer.writerow(headers)

        for node_id, node_props in tqdm(nodes, desc=f"Writing nodes_{label}.csv", unit="node"):
            row = [node_id]
            for prop in props:
                row.append(normalize_csv_value(node_props.get(prop)))
            row.append(label)
            writer.writerow(row)


def write_edges_csv(output_dir: Path, edge_type: str, edges: List[Tuple[str, str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"edges_{edge_type}.csv"

    with file_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(
            f,
            quoting=csv.QUOTE_ALL,
            escapechar="\\",
            lineterminator="\n",
        )
        writer.writerow([":START_ID", ":END_ID", ":TYPE"])

        for start_id, end_id in tqdm(edges, desc=f"Writing edges_{edge_type}.csv", unit="edge"):
            writer.writerow([start_id, end_id, edge_type])


def _convert_one_sample(sample: Dict[str, Any]) -> Tuple[Dict[str, Set[str]], Dict[str, List[Tuple[str, Dict[str, Any]]]], Dict[str, List[Tuple[str, str]]]]:
    converter = TableToGraphConverter()
    converter.graph = nx.MultiDiGraph()
    converter.node_counter = 0
    converter.convert_sample(sample)
    return collect_schema(converter.graph)


def convert_dataset(dataset_path: Path, output_dir: Path, num_workers: int, chunk_size: int) -> None:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info("开始构建图：%s", dataset_path)
    logger.info("并行进程数: %d", num_workers)

    master_label_props: Dict[str, Set[str]] = defaultdict(set)
    master_nodes_by_label: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    master_edges_by_type: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    with mp.get_context("spawn").Pool(processes=num_workers, maxtasksperchild=50) as pool:
        results = pool.imap_unordered(_convert_one_sample, data, chunksize=chunk_size)
        for part in tqdm(results, total=len(data), desc="Building graph", unit="sample"):
            merge_schema(master_label_props, master_nodes_by_label, master_edges_by_type, part)

    logger.info("图构建完成：label=%d, edge_type=%d", len(master_nodes_by_label), len(master_edges_by_type))

    for label, nodes in master_nodes_by_label.items():
        props = sorted(master_label_props[label])
        write_nodes_csv(output_dir, label, nodes, props)

    for edge_type, edges in master_edges_by_type.items():
        write_edges_csv(output_dir, edge_type, edges)

    logger.info("CSV 输出完成：%s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MultiHiertt -> Neo4j CSV 并行导出")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/cqjtu/NLP-Group/LZH/T3GRAG/dataset/MultiHiertt",
        help="MultiHiertt 数据集根目录",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/cqjtu/NLP-Group/LZH/T3GRAG/T3G_csv/outputs",
        help="CSV 输出根目录（含 train/dev/test 子目录）",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,dev,test",
        help="要处理的数据集划分，逗号分隔",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="并行进程数，默认 CPU 核心数-1",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=8,
        help="Pool chunksize，默认 8",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    for split in splits:
        dataset_path = dataset_root / f"{split}.json"
        if not dataset_path.exists():
            raise FileNotFoundError(f"数据集不存在: {dataset_path}")

        output_dir = output_root / split
        logger.info("开始处理 %s", split)
        convert_dataset(dataset_path, output_dir, args.num_workers, args.chunk_size)


if __name__ == "__main__":
    main()
