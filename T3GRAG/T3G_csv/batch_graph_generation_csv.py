#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 MultiHiertt 数据集转为 Neo4j Admin Import CSV（按 label/type 拆分）。
保持原 T3G 图谱设计：节点/边/属性完全来自 TableToGraphConverter。

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
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
from tqdm import tqdm

import sys

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


def collect_schema(graph) -> Tuple[Dict[str, Set[str]], Dict[str, List[Tuple[str, Dict[str, Any]]]], Dict[str, List[Tuple[str, str]]]]:
    """
    返回：
    - label_props: label -> set(property)
    - nodes_by_label: label -> [(node_id, props)]
    - edges_by_type: type -> [(start_id, end_id)]
    """
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


def convert_dataset(dataset_path: Path, output_dir: Path) -> None:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info("开始构建图：%s", dataset_path)
    converter = TableToGraphConverter()
    converter.graph = nx.MultiDiGraph()
    converter.node_counter = 0

    for sample in tqdm(data, desc="Building graph", unit="sample"):
        converter.convert_sample(sample)

    graph = converter.graph
    logger.info("图构建完成：%d 节点, %d 边", graph.number_of_nodes(), graph.number_of_edges())

    label_props, nodes_by_label, edges_by_type = collect_schema(graph)

    # 写节点 CSV（按 label）
    for label, nodes in nodes_by_label.items():
        props = sorted(label_props[label])
        write_nodes_csv(output_dir, label, nodes, props)

    # 写边 CSV（按 type）
    for edge_type, edges in edges_by_type.items():
        write_edges_csv(output_dir, edge_type, edges)

    logger.info("CSV 输出完成：%s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MultiHiertt -> Neo4j CSV 导出")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "MultiHiertt"),
        help="MultiHiertt 数据集根目录",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(Path(__file__).resolve().parent / "outputs"),
        help="CSV 输出根目录（含 train/dev/test 子目录）",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,dev,test",
        help="要处理的数据集划分，逗号分隔",
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
        convert_dataset(dataset_path, output_dir)


if __name__ == "__main__":
    main()
