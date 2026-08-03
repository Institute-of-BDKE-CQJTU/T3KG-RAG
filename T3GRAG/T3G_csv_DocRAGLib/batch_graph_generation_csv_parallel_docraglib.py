#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DocRAGLib 并行批量图谱导出（Neo4j Admin Import CSV）。

输入：docraglib_dev_merged.json（列表，每项一个 uid+qa 样本）
输出：
- nodes_{Label}.csv
- edges_{TYPE}.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
from tqdm import tqdm

from docraglib_table_to_graph_converter import DocRAGLibTableToGraphConverter


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


def collect_schema(
    graph: nx.MultiDiGraph,
) -> Tuple[
    Dict[str, Set[str]],
    Dict[str, List[Tuple[str, Dict[str, Any]]]],
    Dict[str, List[Tuple[str, str]]],
]:
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
    part: Tuple[
        Dict[str, Set[str]],
        Dict[str, List[Tuple[str, Dict[str, Any]]]],
        Dict[str, List[Tuple[str, str]]],
    ],
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
    out = output_dir / f"nodes_{label}.csv"
    headers = ["id:ID"] + props + [":LABEL"]

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")
        writer.writerow(headers)
        for node_id, node_props in tqdm(nodes, desc=f"Writing nodes_{label}.csv", unit="node"):
            row = [node_id]
            row.extend(normalize_csv_value(node_props.get(p)) for p in props)
            row.append(label)
            writer.writerow(row)


def write_edges_csv(output_dir: Path, edge_type: str, edges: List[Tuple[str, str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"edges_{edge_type}.csv"

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")
        writer.writerow([":START_ID", ":END_ID", ":TYPE"])
        for s, t in tqdm(edges, desc=f"Writing edges_{edge_type}.csv", unit="edge"):
            writer.writerow([s, t, edge_type])


def _convert_one_sample(sample: Dict[str, Any]):
    converter = DocRAGLibTableToGraphConverter()
    converter.graph = nx.MultiDiGraph()
    converter.node_counter = 0
    converter.convert_sample(sample)
    return collect_schema(converter.graph)


def convert_dataset(input_json: Path, output_dir: Path, num_workers: int, chunk_size: int, limit: int | None) -> Dict[str, Any]:
    with input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if limit is not None:
        data = data[:limit]

    logger.info("开始并行转换: %s", input_json)
    logger.info("样本数: %d | workers: %d", len(data), num_workers)

    master_label_props: Dict[str, Set[str]] = defaultdict(set)
    master_nodes_by_label: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    master_edges_by_type: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    with mp.get_context("spawn").Pool(processes=num_workers, maxtasksperchild=50) as pool:
        results = pool.imap_unordered(_convert_one_sample, data, chunksize=chunk_size)
        for part in tqdm(results, total=len(data), desc="Building graph", unit="sample"):
            merge_schema(master_label_props, master_nodes_by_label, master_edges_by_type, part)

    logger.info("图构建完成: labels=%d edge_types=%d", len(master_nodes_by_label), len(master_edges_by_type))

    for label, nodes in master_nodes_by_label.items():
        props = sorted(master_label_props[label])
        write_nodes_csv(output_dir, label, nodes, props)
    for et, edges in master_edges_by_type.items():
        write_edges_csv(output_dir, et, edges)

    summary = {
        "input": str(input_json),
        "output_dir": str(output_dir),
        "samples": len(data),
        "labels": sorted(list(master_nodes_by_label.keys())),
        "edge_types": sorted(list(master_edges_by_type.keys())),
        "node_count": int(sum(len(v) for v in master_nodes_by_label.values())),
        "edge_count": int(sum(len(v) for v in master_edges_by_type.values())),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("CSV 输出完成: %s", output_dir)
    logger.info("nodes=%d edges=%d", summary["node_count"], summary["edge_count"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="DocRAGLib -> Neo4j CSV 并行导出")
    parser.add_argument(
        "--input_json",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset" / "DocRAGLib_outputs" / "docraglib_dev_merged.json",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "dev_csv_with_desc",
    )
    parser.add_argument("--num_workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    configure_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = convert_dataset(
        input_json=args.input_json,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
