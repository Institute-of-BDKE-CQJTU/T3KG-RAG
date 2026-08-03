#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DocRAGLib 文本-表格-问答 -> 知识图谱转换器。

使用以下字段：
- uid
- paragraphs
- tables
- qa.document_uid / qa.question_uid / qa.question / qa.answer
- table_description（可选，键格式: table_id-row-col）

不使用（也不建模）: table_evidence / text_evidence / program / question_type
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from T3G.grid_builder import analyze_grid, html_to_grid


class DocRAGLibTableToGraphConverter:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self.node_counter = 0
        self.current_prefix = "sample"

    def _safe_prefix(self, text: str) -> str:
        if not text:
            return "sample"
        safe = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_")
        if len(safe) > 64:
            safe = safe[:64]
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
        return f"{safe}_{digest}" if safe else f"sample_{digest}"

    def _next_node_id(self) -> str:
        self.node_counter += 1
        return f"{self.current_prefix}__n{self.node_counter}"

    @staticmethod
    def _normalize_cell_value(value: str) -> Tuple[Optional[float], str]:
        raw = value or ""
        clean = raw.strip()
        if not clean:
            return None, raw

        num_txt = clean.replace(",", "")
        m = re.search(r"[-+]?\d*\.?\d+", num_txt)
        if not m:
            return None, raw
        try:
            return float(m.group()), raw
        except ValueError:
            return None, raw

    @staticmethod
    def _build_header_levels_from_col_paths(col_paths: Dict[int, List[str]], num_cols: int) -> List[List[str]]:
        if not col_paths:
            return []
        max_depth = max((len(v) for v in col_paths.values()), default=0)
        if max_depth == 0:
            return []

        levels = [["" for _ in range(num_cols)] for _ in range(max_depth)]
        for c in range(num_cols):
            path = col_paths.get(c, [])
            offset = max_depth - len(path)
            for i, txt in enumerate(path):
                levels[offset + i][c] = txt

        for r in range(max_depth):
            last = ""
            for c in range(num_cols):
                if levels[r][c]:
                    last = levels[r][c]
                elif last:
                    levels[r][c] = last

        return levels

    @staticmethod
    def _extract_table_unit(table_html: str) -> Optional[str]:
        text = BeautifulSoup(table_html, "html.parser").get_text(" ", strip=True).lower()
        if "(in millions)" in text:
            return "million"
        if "(in billions)" in text:
            return "billion"
        if "(in thousands)" in text:
            return "thousand"
        if "%" in text or "percent" in text:
            return "percent"
        return None

    def _parse_table(self, table_html: str, table_idx: int) -> Dict[str, Any]:
        grid, cell_meta = html_to_grid(table_html)
        if not grid:
            return {
                "table_idx": table_idx,
                "num_cols": 0,
                "header_rows": 0,
                "col_paths": {},
                "data_rows": [],
                "unit": self._extract_table_unit(table_html),
            }

        info = analyze_grid(grid, cell_meta)
        header_rows = info.get("header_rows", 0)
        col_paths = info.get("col_paths", {})
        num_cols = max(len(r) for r in grid)

        data_rows: List[List[str]] = []
        for r in range(header_rows, len(grid)):
            row_vals: List[str] = []
            for c in range(num_cols):
                cid = grid[r][c] if c < len(grid[r]) else -1
                txt = cell_meta[cid]["text"] if cid != -1 else ""
                row_vals.append(txt)
            data_rows.append(row_vals)

        return {
            "table_idx": table_idx,
            "num_cols": num_cols,
            "header_rows": header_rows,
            "col_paths": col_paths,
            "data_rows": data_rows,
            "unit": self._extract_table_unit(table_html),
        }

    def _build_table_subgraph(self, table_info: Dict[str, Any], doc_id: str) -> str:
        table_idx = table_info["table_idx"]
        num_cols = table_info["num_cols"]
        data_rows: List[List[str]] = table_info["data_rows"]
        col_paths: Dict[int, List[str]] = table_info["col_paths"]
        header_rows = table_info["header_rows"]
        unit = table_info.get("unit")

        table_node = self._next_node_id()
        self.graph.add_node(
            table_node,
            type="Table",
            table_index=table_idx,
            table_id=table_idx,
            doc_id=doc_id,
            num_columns=num_cols,
            unit=unit,
        )

        # HeaderCell（多层列头）
        header_levels = self._build_header_levels_from_col_paths(col_paths, num_cols)
        level_nodes: Dict[int, List[Tuple[int, int, str, str]]] = {}

        for lvl, row in enumerate(header_levels):
            level_nodes[lvl] = []
            c = 0
            while c < num_cols:
                txt = row[c].strip() if c < len(row) else ""
                if not txt:
                    c += 1
                    continue
                start = c
                end = c
                while end + 1 < num_cols and row[end + 1].strip() == txt:
                    end += 1

                h_id = self._next_node_id()
                self.graph.add_node(
                    h_id,
                    type="HeaderCell",
                    doc_id=doc_id,
                    table_index=table_idx,
                    level=lvl,
                    text=txt,
                    span_start=start,
                    span_end=end,
                )
                level_nodes[lvl].append((start, end, txt, h_id))
                if lvl == 0:
                    self.graph.add_edge(table_node, h_id, edge_type="CONTAINS_HEADER")
                c = end + 1

        for lvl in sorted(level_nodes.keys()):
            if lvl == 0:
                continue
            for c_start, c_end, _, child_id in level_nodes[lvl]:
                for p_lvl in range(lvl - 1, -1, -1):
                    parent_found = False
                    for p_start, p_end, _, parent_id in level_nodes.get(p_lvl, []):
                        if p_start <= c_start and c_end <= p_end:
                            self.graph.add_edge(parent_id, child_id, edge_type="CHILD_HEADER")
                            parent_found = True
                            break
                    if parent_found:
                        break
                else:
                    self.graph.add_edge(table_node, child_id, edge_type="CONTAINS_HEADER")

        # RowHeader（行头）: 使用每行第一列的文本（若非空）
        row_header_nodes: Dict[int, str] = {}
        prev_row_header: Optional[str] = None
        for r_idx, row in enumerate(data_rows):
            row_label = row[0].strip() if row and len(row) > 0 else ""
            if not row_label:
                continue
            rh_id = self._next_node_id()
            row_header_nodes[r_idx] = rh_id
            self.graph.add_node(
                rh_id,
                type="RowHeader",
                doc_id=doc_id,
                table_index=table_idx,
                table_id=table_idx,
                level=0,
                text=row_label,
                span_start=r_idx,
                span_end=r_idx,
            )
            self.graph.add_edge(table_node, rh_id, edge_type="CONTAINS_ROW_HEADER")
            if prev_row_header:
                self.graph.add_edge(prev_row_header, rh_id, edge_type="NEXT_SIBLING_ROW_HEADER")
            prev_row_header = rh_id

        # Cell：保留行/列语义路径
        for r_idx, row in enumerate(data_rows):
            if len(row) < num_cols:
                row = row + ["" for _ in range(num_cols - len(row))]

            row_path = []
            if r_idx in row_header_nodes:
                row_path.append(self.graph.nodes[row_header_nodes[r_idx]]["text"])
            elif row and row[0].strip():
                row_path.append(row[0].strip())

            for c_idx in range(num_cols):
                val = row[c_idx]
                numeric_value, original_val = self._normalize_cell_value(val)
                col_path = col_paths.get(c_idx, [])

                cell_id = self._next_node_id()
                attrs: Dict[str, Any] = {
                    "type": "Cell",
                    "doc_id": doc_id,
                    "table_index": table_idx,
                    "table_id": table_idx,
                    "row": header_rows + r_idx,
                    "col": c_idx,
                    "value": original_val,
                    "col_header_path": col_path,
                    "row_header_path": row_path,
                    "col_path_str": " > ".join(col_path),
                    "row_path_str": " > ".join(row_path),
                    "full_path_str": (" > ".join(row_path) + " | " if row_path else "") + " > ".join(col_path),
                }
                if numeric_value is not None:
                    attrs["numeric_value"] = numeric_value
                self.graph.add_node(cell_id, **attrs)

                self.graph.add_edge(table_node, cell_id, edge_type="CONTAINS_CELL")

                # 列语义边：连接最底层覆盖当前列的 HeaderCell
                linked_header = False
                for lvl in sorted(level_nodes.keys(), reverse=True):
                    for start, end, _, h_id in level_nodes[lvl]:
                        if start <= c_idx <= end:
                            self.graph.add_edge(h_id, cell_id, edge_type="DEFINES_COLUMN")
                            linked_header = True
                            break
                    if linked_header:
                        break

                # 行语义边
                if r_idx in row_header_nodes:
                    self.graph.add_edge(row_header_nodes[r_idx], cell_id, edge_type="DEFINES_ROW")

        return table_node

    def _build_paragraphs(self, paragraphs: List[str], doc_id: str, doc_node: str) -> None:
        prev_para = None
        for idx, text in enumerate(paragraphs or []):
            p_id = self._next_node_id()
            self.graph.add_node(
                p_id,
                type="Paragraph",
                doc_id=doc_id,
                paragraph_index=idx,
                content=text,
            )
            self.graph.add_edge(doc_node, p_id, edge_type="CONTAINS_PARAGRAPH")
            if prev_para:
                self.graph.add_edge(prev_para, p_id, edge_type="NEXT_PARAGRAPH")
            prev_para = p_id

    def _build_table_description_graph(
        self,
        table_descriptions: Dict[str, str],
        doc_id: str,
    ) -> List[str]:
        """构建表格描述子图：Cell -> TableDescription（HAS_DESCRIPTION）"""
        desc_nodes: List[str] = []
        if not isinstance(table_descriptions, dict) or not table_descriptions:
            return desc_nodes

        # 先建立 (table_id, row, col) -> cell_id 索引，加速匹配
        cell_index: Dict[Tuple[int, int, int], str] = {}
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get("type") != "Cell":
                continue
            try:
                t = int(node_data.get("table_id"))
                r = int(node_data.get("row"))
                c = int(node_data.get("col"))
                cell_index[(t, r, c)] = node_id
            except Exception:
                continue

        for desc_key, desc_text in table_descriptions.items():
            try:
                parts = str(desc_key).split("-")
                if len(parts) < 3:
                    continue
                table_id = int(parts[0])
                row = int(parts[1])
                col = int(parts[2])
            except Exception:
                continue

            desc_id = self._next_node_id()
            self.graph.add_node(
                desc_id,
                type="TableDescription",
                doc_id=doc_id,
                table_id=table_id,
                row=row,
                col=col,
                desc_key=str(desc_key),
                description=str(desc_text),
            )
            desc_nodes.append(desc_id)

            cell_id = cell_index.get((table_id, row, col))
            if cell_id:
                self.graph.add_edge(cell_id, desc_id, edge_type="HAS_DESCRIPTION")

        return desc_nodes

    def _build_qa(self, qa: Dict[str, Any], doc_id: str, doc_node: str, table_nodes: List[str]) -> str:
        qa_instance = self._next_node_id()
        self.graph.add_node(
            qa_instance,
            type="QAInstance",
            doc_id=doc_id,
            question_uid=qa.get("question_uid"),
            document_uid=qa.get("document_uid"),
        )
        self.graph.add_edge(doc_node, qa_instance, edge_type="HAS_QA")

        q_node = self._next_node_id()
        self.graph.add_node(
            q_node,
            type="Question",
            doc_id=doc_id,
            question_uid=qa.get("question_uid"),
            text=qa.get("question", ""),
        )
        self.graph.add_edge(qa_instance, q_node, edge_type="HAS_QUESTION")

        a_node = self._next_node_id()
        ans = qa.get("answer")
        a_attrs: Dict[str, Any] = {
            "type": "Answer",
            "doc_id": doc_id,
            "question_uid": qa.get("question_uid"),
            "raw": ans,
            "text": str(ans) if ans is not None else "",
        }
        if isinstance(ans, (int, float)):
            a_attrs["numeric_value"] = float(ans)
        self.graph.add_node(a_node, **a_attrs)
        self.graph.add_edge(qa_instance, a_node, edge_type="HAS_ANSWER")

        # QA 到文档 / 表格语义连接（无 evidence 时的弱连接）
        self.graph.add_edge(qa_instance, doc_node, edge_type="ABOUT_DOCUMENT")
        for t_node in table_nodes:
            self.graph.add_edge(qa_instance, t_node, edge_type="REFERENCES_TABLE")

        return qa_instance

    def convert_sample(self, sample: Dict[str, Any]) -> nx.MultiDiGraph:
        doc_id = sample.get("uid", "unknown_doc")
        qa = sample.get("qa") or {}
        question_uid = qa.get("question_uid") or "no_question_uid"
        prefix_source = f"{doc_id}::{question_uid}"
        self.current_prefix = self._safe_prefix(prefix_source)
        self.node_counter = 0

        doc_node = self._next_node_id()
        self.graph.add_node(doc_node, type="Doc", doc_id=doc_id, uid=doc_id)

        self._build_paragraphs(sample.get("paragraphs", []), doc_id, doc_node)

        table_nodes: List[str] = []
        for idx, table_html in enumerate(sample.get("tables", [])):
            table_info = self._parse_table(table_html, idx)
            table_node = self._build_table_subgraph(table_info, doc_id)
            table_nodes.append(table_node)
            self.graph.add_edge(doc_node, table_node, edge_type="HAS_TABLE")

        table_descriptions = sample.get("table_description") or {}
        self._build_table_description_graph(table_descriptions, doc_id)

        qa = sample.get("qa") or {}
        if qa:
            self._build_qa(qa, doc_id, doc_node, table_nodes)

        return self.graph

    def convert_samples(self, samples: List[Dict[str, Any]]) -> nx.MultiDiGraph:
        self.graph = nx.MultiDiGraph()
        self.node_counter = 0
        for sample in samples:
            self.convert_sample(sample)
        return self.graph

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\\", "\\\\")
        text = text.replace('"', '\\"')
        text = text.replace("\n", "\\n")
        text = text.replace("\r", "\\r")
        return text

    def export_to_edgelist(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for u, v, data in self.graph.edges(data=True):
                f.write(f"{u}\t{v}\t{data.get('edge_type', 'RELATED_TO')}\n")

    def export_to_neo4j_cypher(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write("// DocLibRAG Graph Cypher\n\n")
            for n_id, n_data in self.graph.nodes(data=True):
                label = n_data.get("type", "Unknown")
                props = [f"id: {repr(str(n_id))}"]
                for k, v in n_data.items():
                    if k == "type" or v is None:
                        continue
                    if isinstance(v, bool):
                        props.append(f"{k}: {'true' if v else 'false'}")
                    elif isinstance(v, (int, float)):
                        props.append(f"{k}: {v}")
                    elif isinstance(v, (list, tuple, set)):
                        val = " | ".join(str(x) for x in v)
                        props.append(f'{k}: "{self._clean_text(val)}"')
                    elif isinstance(v, dict):
                        props.append(f'{k}: "{self._clean_text(json.dumps(v, ensure_ascii=False))}"')
                    else:
                        props.append(f'{k}: "{self._clean_text(str(v))}"')

                f.write(f"CREATE (:{label} {{{', '.join(props)}}});\n")

            f.write("\n")
            for u, v, e_data in self.graph.edges(data=True):
                e_type = e_data.get("edge_type", "RELATED_TO").upper()
                f.write(
                    f"MATCH (a {{id: {repr(str(u))}}}), (b {{id: {repr(str(v))}}}) CREATE (a)-[:{e_type}]->(b);\n"
                )


def run_single_file(input_json: Path, output_dir: Path, limit: Optional[int] = None) -> None:
    with input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if limit is not None:
        data = data[:limit]

    converter = DocRAGLibTableToGraphConverter()
    graph = converter.convert_samples(data)

    output_dir.mkdir(parents=True, exist_ok=True)
    edgelist_path = output_dir / "docraglib_dev_graph_edgelist.txt"
    cypher_path = output_dir / "docraglib_dev_graph.cypher"
    summary_path = output_dir / "docraglib_dev_graph_summary.json"

    converter.export_to_edgelist(edgelist_path)
    converter.export_to_neo4j_cypher(cypher_path)

    summary = {
        "input": str(input_json),
        "samples": len(data),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "edgelist": str(edgelist_path),
        "cypher": str(cypher_path),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="DocRAGLib merged json -> graph")
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
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_single_file(args.input_json, args.output_dir, args.limit)


if __name__ == "__main__":
    main()
