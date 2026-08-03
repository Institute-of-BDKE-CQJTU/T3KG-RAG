#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiHiertt 表格转图转换器
基于改进的"文档-文本-表格-事实-问答"五层异构图结构
"""

import json
import re
import pandas as pd
from bs4 import BeautifulSoup
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TableToGraphConverter:
    """表格转图转换器"""
    
    def __init__(self, qa_decompose: bool = True, keep_legacy_qa_node: bool = True):
        self.graph = nx.MultiDiGraph()
        self.node_counter = 0
        self.current_doc_prefix = "global"
        # QA建模策略:
        # - qa_decompose=True: 拆分 Question/Answer/Program/QuestionType/Evidence 节点（推荐，便于检索/推理/监督训练）
        # - keep_legacy_qa_node=True: 仍然保留原 QuestionAnswer 节点与旧边（兼容既有脚本/导出）
        self.qa_decompose = qa_decompose
        self.keep_legacy_qa_node = keep_legacy_qa_node
        
    def _sanitize_doc_id_for_prefix(self, doc_id: str) -> str:
        """将 doc_id 归一化为可作为前缀的安全字符串。"""
        if not doc_id:
            return "unknown"
        # 仅保留字母数字，其他替换为下划线，避免特殊字符污染 id
        return re.sub(r"[^0-9a-zA-Z]+", "_", doc_id)

    def get_next_node_id(self) -> str:
        """获取下一个节点ID（带 doc 级前缀，确保全局唯一）"""
        self.node_counter += 1
        return f"{self.current_doc_prefix}__node_{self.node_counter}"
    
    def parse_unit_from_text(self, text: str) -> Tuple[Optional[str], Optional[float]]:
        """从文本中解析单位和尺度"""
        text = text.lower()
        
        # 常见单位模式
        unit_patterns = {
            r'\(in millions\)': ('USD', 1e6),
            r'\(in billions\)': ('USD', 1e9),
            r'\(in thousands\)': ('USD', 1e3),
            r'\(millions\)': ('USD', 1e6),
            r'\(billions\)': ('USD', 1e9),
            r'%': ('%', 1.0),
            r'\$': ('USD', 1.0)
        }
        
        for pattern, (unit, scale) in unit_patterns.items():
            if re.search(pattern, text):
                return unit, scale
        
        return None, 1.0
    
    def _analyze_question_complexity(self, question: str) -> str:
        """分析问题复杂度"""
        if not question:
            return "unknown"
        
        question_lower = question.lower()
        
        # 复杂问题关键词
        complex_keywords = ['compare', 'difference', 'ratio', 'percentage', 'average', 'total', 'sum', 
                          'maximum', 'minimum', 'highest', 'lowest', 'increase', 'decrease', 
                          'change', 'growth', 'multiple', 'times', 'more than', 'less than']
        
        # 简单问题关键词  
        simple_keywords = ['what', 'which', 'who', 'when', 'where', 'how many', 'how much']
        
        complex_count = sum(1 for keyword in complex_keywords if keyword in question_lower)
        simple_count = sum(1 for keyword in simple_keywords if keyword in question_lower)
        
        if complex_count > simple_count:
            return "complex"
        elif simple_count > 0:
            return "simple"
        else:
            return "medium"
    
    def _has_numerical_reasoning(self, program: str) -> bool:
        """检查是否包含数值推理"""
        if not program:
            return False
        
        program_str = str(program).lower()
        numerical_ops = ['add', 'subtract', 'multiply', 'divide', 'sum', 'average', 'max', 'min', 
                        'greater', 'less', 'equal', 'count', 'percentage', 'ratio']
        
        return any(op in program_str for op in numerical_ops)
    
    def _has_multi_step_reasoning(self, program: str) -> bool:
        """检查是否包含多步推理"""
        if not program:
            return False
        
        program_str = str(program)
        # 检查是否有多个操作步骤
        step_indicators = [';', 'then', 'next', 'after', 'step']
        multi_ops = program_str.count('(') > 2 or any(indicator in program_str.lower() for indicator in step_indicators)
        
        return multi_ops
    
    def normalize_cell_value(self, cell_text: str, unit: str = None, scale: float = 1.0) -> Tuple[Optional[float], str]:
        """标准化单元格值"""
        if not cell_text or cell_text.strip() == '':
            return None, cell_text
        
        # 清理文本
        clean_text = re.sub(r'[\,\s]', '', cell_text)
        
        # 尝试提取数值
        number_match = re.search(r'[-+]?(\d+\.?\d*|\.\d+)', clean_text)
        if number_match:
            try:
                value = float(number_match.group())
                if unit and scale:
                    value *= scale
                return value, cell_text
            except ValueError:
                pass
        
        return None, cell_text
    
    def _get_cell_text(self, cell) -> str:
        """提取单元格纯文本"""
        return cell.get_text(strip=True) if cell else ''
    
    def _expand_table_to_grid(self, table) -> Tuple[List[List[str]], List[List[str]]]:
        """改进版：解析 rowspan/colspan 并生成对齐网格。
        通过维护 ``rowspan_tracker``（记录上一行各列剩余可用行数）来精确放置单元格，
        可避免旧实现因重复填 '' 导致列偏移的问题。
        返回 (header_grid, data_grid)。"""
        # 初始化
        header_rows_raw: List[Any] = []
        thead = table.find('thead')
        if thead:
            header_rows_raw = thead.find_all('tr')
        else:
            all_trs = table.find_all('tr')
            for tr in all_trs:
                if tr.find('th') is not None:
                    header_rows_raw.append(tr)
                else:
                    break  # 遇到首列非空或明显数据行，停止搜集表头
            # 若仍未检测到表头，保底取首行
            if not header_rows_raw and all_trs:
                header_rows_raw = [all_trs[0]]
        # 数据行 = 其余行 / tbody
        data_rows_raw: List[Any] = []
        if table.find('tbody'):
            data_rows_raw = table.find('tbody').find_all('tr')
        else:
            all_trs = table.find_all('tr')
            data_rows_raw = [tr for tr in all_trs if tr not in header_rows_raw]

        def build_grid(rows_raw: List[Any]) -> List[List[str]]:
            grid: List[List[str]] = []
            rowspan_tracker: List[int] = []  # 每列剩余占位行数
            for tr in rows_raw:
                cur_row: List[str] = []
                cells = tr.find_all(['th', 'td'])
                cell_idx = 0  # 指向 cells
                col_pos = 0   # 指向输出列索引
                # 逐列填充到当前行
                while cell_idx < len(cells) or (col_pos < len(rowspan_tracker) and rowspan_tracker[col_pos] > 0):
                    # 处理来自上一行的 rowspan 占位
                    if col_pos < len(rowspan_tracker) and rowspan_tracker[col_pos] > 0:
                        cur_row.append('')
                        rowspan_tracker[col_pos] -= 1
                        col_pos += 1
                        continue
                    # 没有更多 cell 可放
                    if cell_idx >= len(cells):
                        # 剩余列只受 rowspan 影响，若无则结束
                        break
                    cell = cells[cell_idx]
                    cell_idx += 1
                    text = self._get_cell_text(cell)
                    rs = int(cell.get('rowspan', '1') or '1')
                    cs = int(cell.get('colspan', '1') or '1')
                    # 确保 tracker 足够长
                    while len(rowspan_tracker) < col_pos + cs:
                        rowspan_tracker.append(0)
                    for span in range(cs):
                        cur_row.append(text)
                        if rs > 1:
                            rowspan_tracker[col_pos] = rs - 1
                        col_pos += 1
                grid.append(cur_row)
            return grid

        header_grid = build_grid(header_rows_raw)
        data_grid = build_grid(data_rows_raw)
        return header_grid, data_grid
        """将HTML表格展开为网格，分别返回header网格和数据网格。
        处理 thead/tbody, th/td, 以及 rowspan/colspan。
        """
        # 收集header行（来自 thead 或 以 th 为主的顶部行）
        header_rows_raw = []
        thead = table.find('thead')
        if thead:
            header_rows_raw = thead.find_all('tr')
        else:
            # 如果没有thead，找到从顶部开始连续包含th的tr作为表头
            all_trs = table.find_all('tr')
            for tr in all_trs:
                if tr.find('th') is not None:
                    header_rows_raw.append(tr)
                else:
                    # 一旦遇到不含th的行，停止视为header连续区
                    break
            
            # 如果没有找到th标签，启发式识别表头
        # 如果没有找到 th 标签：启发式识别表头（即使全是 td）
        if not header_rows_raw:
            all_trs = table.find_all('tr')
            header_count = 0
            for i, tr in enumerate(all_trs):
                cells = tr.find_all(['td', 'th'])
                if not cells:
                    continue
                row_text = ' '.join([c.get_text(strip=True) for c in cells]).lower()
                non_empty = [c.get_text(strip=True) for c in cells if c.get_text(strip=True)]
                year_cells = [c for c in non_empty if c.isdigit() and 1900 <= int(c) <= 2099]
                has_colspan = any(c.get('colspan') for c in cells)
                # 经验规则：单位行/年份行/跨列描述行/首行，判为表头
                is_header = (
                    i == 0 or
                    has_colspan or
                    '(in ' in row_text or
                    'year ended' in row_text or
                    'december' in row_text or
                    (non_empty and len(year_cells) >= max(1, int(len(non_empty) * 0.5)))
                )
                if is_header:
                    header_rows_raw.append(tr)
                    header_count += 1
                    if header_count >= 6:
                        break
                else:
                    break
            if not header_rows_raw and all_trs:
                header_rows_raw = [all_trs[0]]

        # 数据行 = tbody 或剩余行
        data_rows_raw = []
        tbody = table.find('tbody')
        if tbody:
            data_rows_raw = tbody.find_all('tr')
        else:
            all_trs = table.find_all('tr')
            data_rows_raw = [tr for tr in all_trs if tr not in header_rows_raw]

        def build_grid(rows_raw: List[Any]) -> List[List[str]]:
            grid: List[List[str]] = []
            rowspan_tracker: List[int] = []
            for tr in rows_raw:
                cur_row: List[str] = []
                cells = tr.find_all(['th', 'td'])
                cell_idx = 0
                col_pos = 0
                while cell_idx < len(cells) or (col_pos < len(rowspan_tracker) and rowspan_tracker[col_pos] > 0):
                    if col_pos < len(rowspan_tracker) and rowspan_tracker[col_pos] > 0:
                        cur_row.append('')
                        rowspan_tracker[col_pos] -= 1
                        col_pos += 1
                        continue
                    if cell_idx >= len(cells):
                        break
                    cell = cells[cell_idx]
                    cell_idx += 1
                    text = self._get_cell_text(cell)
                    rs = int(cell.get('rowspan', '1') or '1')
                    cs = int(cell.get('colspan', '1') or '1')
                    while len(rowspan_tracker) < col_pos + cs:
                        rowspan_tracker.append(0)
                    for _ in range(cs):
                        cur_row.append(text)
                        if rs > 1:
                            rowspan_tracker[col_pos] = rs - 1
                        col_pos += 1
                grid.append(cur_row)
            return grid

        header_grid = build_grid(header_rows_raw) if header_rows_raw else []
        data_grid = build_grid(data_rows_raw) if data_rows_raw else []
        return header_grid, data_grid
    
    def _collapse_headers(self, header_grid: List[List[str]], num_columns: int) -> List[str]:
        """将多行表头按列垂直拼接生成最终列名，长度为 num_columns。"""
        if not header_grid:
            return [f"col_{i}" for i in range(num_columns)]
        # 归一化每行长度
        normalized = [row + [''] * (num_columns - len(row)) if len(row) < num_columns else row[:num_columns] for row in header_grid]
        headers: List[str] = []
        for c in range(num_columns):
            parts = [normalized[r][c] for r in range(len(normalized)) if normalized[r][c]]
            headers.append(' > '.join(parts) if parts else f"col_{c}")
        return headers
    
    def _detect_header_role_and_time(self, text: str) -> Tuple[str, Optional[int]]:
        """检测表头的语义角色和时间值
        
        Args:
            text: 表头文本
            
        Returns:
            (role, time_value) 其中 role ∈ {entity, measure, time, unit, other}
        """
        text_lower = text.lower().strip()
        
        # 检测时间（年份）
        import re
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', text)
        time_value = int(year_match.group(1)) if year_match else None
        
        # 检测角色
        if time_value or any(kw in text_lower for kw in ['year', 'month', 'quarter', 'date', 'period']):
            role = 'time'
        elif any(kw in text_lower for kw in ['$', 'usd', 'revenue', 'income', 'profit', 'cost', 'expense', 
                                              'sales', 'earnings', 'assets', 'liabilities', 'equity',
                                              'cash', 'debt', 'value', 'amount', 'total']):
            role = 'measure'
        elif any(kw in text_lower for kw in ['million', 'billion', 'thousand', '%', 'percent', 
                                              'per share', 'in thousands', 'in millions']):
            role = 'unit'
        elif len(text_lower) > 0 and not text_lower[0].isdigit():
            role = 'entity'
        else:
            role = 'other'
        
        return role, time_value
    
    def build_header_cells(self, header_grid: List[List[str]], table_id: int, doc_id: str, 
                           table_node_id: str) -> Dict[str, Any]:
        """构建HeaderCell节点和层级关系
        
        Args:
            header_grid: 表头网格
            table_id: 表格ID
            doc_id: 文档ID
            table_node_id: 表格节点ID
            
        Returns:
            包含header_cells信息的字典
        """
        if not header_grid:
            return {"header_cells": [], "col_to_header_path": {}, "col_to_unit": {}}
        
        header_cells = []
        # 记录每个层级的HeaderCell：level -> [(col_start, col_end, text, node_id)]
        level_headers: Dict[int, List[Tuple[int, int, str, str]]] = {}
        # 记录每列的表头路径：col_idx -> [header_texts from top to bottom]
        col_to_header_path: Dict[int, List[str]] = {}
        # 记录每列的单位信息：col_idx -> (unit, scale)
        col_to_unit: Dict[int, Tuple[Optional[str], float]] = {}
        
        num_levels = len(header_grid)
        num_columns = len(header_grid[0]) if header_grid else 0
        
        # 逐层处理表头
        for level in range(num_levels):
            row = header_grid[level]
            level_headers[level] = []
            
            # 识别连续相同文本的单元格（合并后的表头）
            col_idx = 0
            while col_idx < num_columns:
                text = row[col_idx]
                if not text or text.strip() == '':
                    col_idx += 1
                    continue
                
                # 找到相同文本的范围
                span_start = col_idx
                span_end = col_idx
                while span_end < num_columns - 1 and row[span_end + 1] == text:
                    span_end += 1
                
                # 计算跨度大小
                span_size = span_end - span_start + 1
                
                # 检测语义角色和时间值
                role, time_value = self._detect_header_role_and_time(text)
                
                # 解析单位信息
                unit, scale = self.parse_unit_from_text(text)
                
                # 创建HeaderCell节点（暂不设置is_leaf，稍后统一设置）
                header_cell_id = self.get_next_node_id()
                node_attrs = {
                    "type": "HeaderCell",
                    "table_id": table_id,
                    "level": level,
                    "position": len(level_headers[level]),
                    "text": text,
                    "span_start": span_start,
                    "span_end": span_end,
                    "span_size": span_size,
                    "role": role,
                    "doc_id": doc_id
                }
                
                if time_value is not None:
                    node_attrs["time_value"] = time_value
                if unit is not None:
                    node_attrs["unit"] = unit
                    node_attrs["scale"] = scale
                    # 记录该列范围的单位
                    for c in range(span_start, span_end + 1):
                        if c not in col_to_unit:
                            col_to_unit[c] = (unit, scale)
                
                self.graph.add_node(header_cell_id, **node_attrs)
                
                level_headers[level].append((span_start, span_end, text, header_cell_id))
                header_cells.append(header_cell_id)
                
                # 第一层表头直接连接到Table
                if level == 0:
                    self.graph.add_edge(table_node_id, header_cell_id, edge_type="CONTAINS_HEADER")
                
                col_idx = span_end + 1
        
        # 建立父子层级关系
        for level in range(1, num_levels):
            for child_start, child_end, child_text, child_id in level_headers[level]:
                # 向上查找所有可能的父层级，找到最近的完全包含当前节点的父节点
                parent_found = False
                for parent_level in range(level - 1, -1, -1):  # 从近到远查找父层级
                    if parent_level not in level_headers:
                        continue
                    
                    for parent_start, parent_end, parent_text, parent_id in level_headers[parent_level]:
                        # 检查父节点是否完全包含子节点
                        if parent_start <= child_start and child_end <= parent_end:
                            # 建立父子关系
                            self.graph.add_edge(parent_id, child_id, edge_type="CHILD_HEADER")
                            # 在子节点中记录父节点ID
                            self.graph.nodes[child_id]['parent_id'] = parent_id
                            parent_found = True
                            break
                    
                    if parent_found:
                        break
                
                # 如果找不到父节点，说明这是一个独立的顶层节点（如跨多列的单位说明行）
                # 这种情况下不设置 parent_id 是正确的
                # 但我们需要建立它到 Table 的连接
                if not parent_found:
                    # 将这个独立节点连接到 Table
                    self.graph.add_edge(table_node_id, child_id, edge_type="CONTAINS_HEADER")
        
        # 标记叶节点（最底层）并建立同层兄弟关系
        for level in range(num_levels):
            if level not in level_headers:
                continue
            
            # 按position排序的同层节点
            level_nodes = sorted(level_headers[level], key=lambda x: x[0])  # 按span_start排序
            
            for i, (span_start, span_end, text, node_id) in enumerate(level_nodes):
                # 检查是否为叶节点（没有子节点）
                has_children = False
                if level < num_levels - 1:
                    # 检查下一层是否有被当前节点包含的子节点
                    for child_start, child_end, child_text, child_id in level_headers.get(level + 1, []):
                        if span_start <= child_start and child_end <= span_end:
                            has_children = True
                            break
                
                self.graph.nodes[node_id]['is_leaf'] = not has_children
                
                # 建立同层兄弟关系（NEXT_SIBLING_HEADER）
                if i < len(level_nodes) - 1:
                    next_node_id = level_nodes[i + 1][3]
                    self.graph.add_edge(node_id, next_node_id, edge_type="NEXT_SIBLING_HEADER")
        
        # 构建每列的表头路径（关键改进：使用 span 覆盖关系向下传播父级表头）
        # 旧逻辑直接读取 header_grid[level][col]，遇到合并单元格导致的空白会丢失父级信息。
        # 预构建每层列->文本映射，并进行横向前向填充 (forward fill)
        level_col_text: Dict[int, List[str]] = {}
        for level in range(num_levels):
            arr = [''] * num_columns
            for span_start, span_end, text, _nid in level_headers.get(level, []):
                for c in range(span_start, span_end + 1):
                    arr[c] = text
            # 前向填充：如果当前列为空，继承左侧最近的非空
            last_seen = ''
            for c in range(num_columns):
                if arr[c]:
                    last_seen = arr[c]
                else:
                    arr[c] = last_seen
            level_col_text[level] = arr

        for col in range(num_columns):
            path: List[str] = []
            for level in range(num_levels):
                text = level_col_text[level][col]
                if text and (not path or path[-1] != text):
                    path.append(text)
            col_to_header_path[col] = path

        return {
            "header_cells": header_cells,
            "col_to_header_path": col_to_header_path,
            "col_to_unit": col_to_unit,
            "level_headers": level_headers
        }
    
    def _identify_row_groups(self, rows: List[List[str]], num_columns: int) -> List[Tuple[int, str, bool]]:
        """识别行分组结构
        
        Args:
            rows: 数据行列表
            num_columns: 列数
            
        Returns:
            [(row_index, group_label, is_summary), ...]
            row_index: 行索引
            group_label: 分组标签文本
            is_summary: 是否为汇总行
        """
        row_groups = []
        
        for row_idx, row in enumerate(rows):
            if len(row) < num_columns:
                continue
                
            first_cell = row[0].strip()
            rest_cells = [cell.strip() for cell in row[1:]]
            
            # 分组标识行：第一列有内容，其他列为空
            if first_cell and all(not cell for cell in rest_cells):
                # 检查是否为汇总/比较分组
                is_summary = any(keyword in first_cell.lower() 
                               for keyword in ['percentage', 'increase', 'decrease', 'total', 'summary'])
                row_groups.append((row_idx, first_cell, is_summary))
        
        return row_groups
    
    def build_row_headers(self, rows: List[List[str]], table_id: int, doc_id: str,
                          table_node_id: str, num_columns: int) -> Dict[str, Any]:
        """构建RowHeader节点和层级关系
        
        Args:
            rows: 数据行列表
            table_id: 表格ID
            doc_id: 文档ID
            table_node_id: 表格节点ID
            num_columns: 列数
            
        Returns:
            包含row_headers信息的字典
        """
        # 识别行分组
        row_groups = self._identify_row_groups(rows, num_columns)
        
        if not row_groups:
            # 回退策略：若未识别到分组，则使用第0列的非数值文本作为逐行RowHeader
            row_headers = []
            level_row_headers: Dict[int, List[Tuple[int, int, str, str, bool]]] = {0: []}
            row_to_group_path: Dict[int, List[str]] = {}
            for row_idx, row in enumerate(rows):
                if len(row) < 1:
                    continue
                label = (row[0] or "").strip()
                if not label:
                    continue
                # 跳过明显数值/符号串，只保留文本行头
                num_like = re.fullmatch(r'[\s\$\%,\-\+\.\d]+', label)
                if num_like:
                    continue
                rh_id = self.get_next_node_id()
                self.graph.add_node(rh_id,
                                    type="RowHeader",
                                    table_id=table_id,
                                    level=0,
                                    position=row_idx,
                                    text=label,
                                    span_start=row_idx,
                                    span_end=row_idx,
                                    span_size=1,
                                    is_summary=False,
                                    is_leaf=True,
                                    doc_id=doc_id)
                row_headers.append(rh_id)
                level_row_headers[0].append((row_idx, row_idx, label, rh_id, False))
                # 连接到表
                self.graph.add_edge(table_node_id, rh_id, edge_type="CONTAINS_ROW_HEADER")
                # 行路径
                row_to_group_path[row_idx] = [label]
            # 建立同层兄弟关系
            if level_row_headers.get(0):
                siblings = sorted(level_row_headers[0], key=lambda x: x[0])
                for i in range(len(siblings) - 1):
                    self.graph.add_edge(siblings[i][3], siblings[i+1][3], edge_type="NEXT_SIBLING_ROW_HEADER")
            return {
                "row_headers": row_headers,
                "row_to_group_path": row_to_group_path,
                "level_row_headers": level_row_headers
            }
        
        row_headers = []
        # 记录每个层级的RowHeader：level -> [(span_start, span_end, text, node_id, is_summary)]
        level_row_headers: Dict[int, List[Tuple[int, int, str, str, bool]]] = {0: [], 1: []}
        # 记录每行的分组路径：row_idx -> [group_texts]
        row_to_group_path: Dict[int, List[str]] = {}
        
        # 构建Level 0的RowHeader（主分组）
        for i, (group_row_idx, group_label, is_summary) in enumerate(row_groups):
            # 计算该分组的跨度
            span_start = group_row_idx
            # 找到下一个分组或表格结束
            if i < len(row_groups) - 1:
                span_end = row_groups[i + 1][0] - 1
            else:
                span_end = len(rows) - 1
            
            # 计算跨度大小
            span_size = span_end - span_start + 1
            
            # 创建RowHeader节点
            row_header_id = self.get_next_node_id()
            self.graph.add_node(row_header_id,
                               type="RowHeader",
                               table_id=table_id,
                               level=0,
                               position=i,
                               text=group_label,
                               span_start=span_start,
                               span_end=span_end,
                               span_size=span_size,
                               is_summary=is_summary,
                               doc_id=doc_id)
            
            row_headers.append(row_header_id)
            level_row_headers[0].append((span_start, span_end, group_label, row_header_id, is_summary))
            
            # 第一层RowHeader直接连接到Table
            self.graph.add_edge(table_node_id, row_header_id, edge_type="CONTAINS_ROW_HEADER")
            
            # 为该分组下的所有数据行建立分组路径
            for row_idx in range(span_start, span_end + 1):
                if row_idx != group_row_idx:  # 跳过分组标识行本身
                    # 识别是否为汇总行（包含"total", "sum"等关键词）
                    row_text = rows[row_idx][0].strip().lower() if row_idx < len(rows) else ""
                    is_row_summary = any(keyword in row_text 
                                        for keyword in ['total', 'sum', 'subtotal'])
                    
                    # 如果是数据行，添加Level 1的细分项
                    if rows[row_idx][0].strip() and not is_row_summary:
                        # 创建Level 1的RowHeader（子项）
                        sub_row_header_id = self.get_next_node_id()
                        sub_label = rows[row_idx][0].strip()
                        
                        self.graph.add_node(sub_row_header_id,
                                           type="RowHeader",
                                           table_id=table_id,
                                           level=1,
                                           position=row_idx - span_start - 1,
                                           text=sub_label,
                                           span_start=row_idx,
                                           span_end=row_idx,
                                           span_size=1,
                                           parent_id=row_header_id,
                                           is_summary=False,
                                           doc_id=doc_id)
                        
                        row_headers.append(sub_row_header_id)
                        level_row_headers[1].append((row_idx, row_idx, sub_label, sub_row_header_id, False))
                        
                        # 建立父子关系
                        self.graph.add_edge(row_header_id, sub_row_header_id, edge_type="CHILD_ROW_HEADER")
                        
                        # 记录分组路径
                        row_to_group_path[row_idx] = [group_label, sub_label]
                    elif is_row_summary:
                        # 汇总行：只有主分组路径
                        row_to_group_path[row_idx] = [group_label, row_text.title()]
                    else:
                        # 普通数据行
                        row_to_group_path[row_idx] = [group_label]
        
        # 标记叶节点并建立同层兄弟关系
        for level in [0, 1]:
            if level not in level_row_headers or not level_row_headers[level]:
                continue
            
            # 按span_start排序的同层节点
            level_nodes = sorted(level_row_headers[level], key=lambda x: x[0])
            
            for i, (span_start, span_end, text, node_id, is_summary) in enumerate(level_nodes):
                # 检查是否为叶节点（level 1总是叶节点，level 0检查是否有子节点）
                if level == 1:
                    is_leaf = True
                else:
                    # 检查是否有子节点
                    has_children = any(
                        child_start >= span_start and child_end <= span_end
                        for child_start, child_end, _, _, _ in level_row_headers.get(1, [])
                    )
                    is_leaf = not has_children
                
                self.graph.nodes[node_id]['is_leaf'] = is_leaf
                
                # 建立同层兄弟关系（NEXT_SIBLING_ROW_HEADER）
                if i < len(level_nodes) - 1:
                    next_node_id = level_nodes[i + 1][3]
                    self.graph.add_edge(node_id, next_node_id, edge_type="NEXT_SIBLING_ROW_HEADER")
        
        return {
            "row_headers": row_headers,
            "row_to_group_path": row_to_group_path,
            "level_row_headers": level_row_headers
        }
    
    def parse_html_table(self, html_str: str, table_id: int) -> Dict[str, Any]:
        """解析HTML表格：采用 grid_builder 将表格完全展开，再基于 grid 分割 header / data。"""
        from T3G.grid_builder import html_to_grid, analyze_grid  # local import to avoid circular

        grid, cell_meta = html_to_grid(html_str)
        if not grid:
            return {
                "table_id": table_id,
                "headers": [],
                "rows": [],
                "header_grid": [],
                "unit": None,
                "scale": 1.0
            }

        grid_info = analyze_grid(grid, cell_meta)
        header_rows_cnt = grid_info["header_rows"]
        col_paths = grid_info["col_paths"]

        num_columns = max(len(r) for r in grid) if grid else 0

        # Build header_grid (List[List[str]]) from header rows
        header_grid: List[List[str]] = []
        for r in range(header_rows_cnt):
            row_texts: List[str] = []
            for c in range(num_columns):
                cid = grid[r][c] if c < len(grid[r]) else -1
                txt = cell_meta[cid]["text"] if cid != -1 else ""
                row_texts.append(txt)
            header_grid.append(row_texts)

        # Build data_grid from remaining rows
        data_grid: List[List[str]] = []
        for r in range(header_rows_cnt, len(grid)):
            row_texts: List[str] = []
            for c in range(num_columns):
                cid = grid[r][c] if c < len(grid[r]) else -1
                txt = cell_meta[cid]["text"] if cid != -1 else ""
                row_texts.append(txt)
            data_grid.append(row_texts)

        # Column headers
        # Prefer the last header row text (closest to data). If it's empty, fall back to
        # the hierarchical path from col_paths, and only then use the synthetic col_i.
        headers: List[str] = []
        last_header_row = header_grid[-1] if header_grid else []
        for c in range(num_columns):
            txt = (last_header_row[c] if c < len(last_header_row) else "").strip()
            if txt:
                headers.append(txt)
                continue
            path = col_paths.get(c, [])
            if path:
                headers.append(" > ".join(path))
            else:
                headers.append(f"col_{c}")

        # unit detection from whole table text
        soup_text = " ".join([m["text"] for m in cell_meta.values()])
        unit, scale = self.parse_unit_from_text(soup_text)

        return {
            "table_id": table_id,
            "headers": headers,
            "rows": data_grid,
            "header_grid": header_grid,
            "header_rows_count": header_rows_cnt,
            "num_columns": num_columns,
            "unit": unit,
            "scale": scale,
            "col_paths": col_paths  # pass for later use
        }
    
    def build_table_graph(self, table_info: Dict[str, Any], doc_id: str) -> str:
        """构建表格子图"""
        table_id = table_info["table_id"]
        headers = table_info.get("headers", [])
        rows = table_info.get("rows", [])
        header_grid = table_info.get("header_grid", [])  # 新增：获取表头网格
        
        # 先准备 num_columns 供后续使用
        num_columns = table_info.get("num_columns")
        if not num_columns or num_columns <= 0:
            num_columns = max(len(r) for r in rows) if rows else 0
            num_columns = max(num_columns, len(headers))
            table_info["num_columns"] = num_columns
        
        # 若解析阶段未生成 header_grid，但 col_paths 已提供真实列头路径，则重建 header_grid，避免合成“col_0”
        if (not header_grid or len(header_grid) == 0) and table_info.get("col_paths"):
            col_paths = table_info["col_paths"]
            max_layers = max(len(p) for p in col_paths.values()) if col_paths else 0
            header_grid = [["" for _ in range(num_columns)] for _ in range(max_layers)]
            for c in range(num_columns):
                path = col_paths.get(c, [])
                offset = max_layers - len(path)
                for idx, txt in enumerate(path):
                    header_grid[offset + idx][c] = txt
            table_info["header_rows_count"] = max_layers
        num_columns = table_info.get("num_columns", len(headers))
        unit = table_info.get("unit")
        scale = table_info.get("scale", 1.0)
        
        # 统一列数
        if num_columns is None or num_columns <= 0:
            num_columns = max([len(r) for r in rows] + [len(headers)]) if (rows or headers) else 0
        
        # 补齐headers长度
        if len(headers) < num_columns:
            headers = headers + [f"col_{i}" for i in range(len(headers), num_columns)]
        elif len(headers) > num_columns:
            headers = headers[:num_columns]
        
        # 创建表格节点
        table_node_id = self.get_next_node_id()
        self.graph.add_node(table_node_id, 
                           type="Table", 
                           table_id=table_id,
                           doc_id=doc_id,
                           headers=headers,
                           unit=unit,
                           scale=scale)
        
        # 构建HeaderCell节点（如果有多层表头）
        col_to_header_path = {}
        col_to_unit = {}
        level_headers = {}
        if header_grid and len(header_grid) > 0:
            header_result = self.build_header_cells(header_grid, table_id, doc_id, table_node_id)
            col_to_header_path = header_result.get("col_to_header_path", {})
            col_to_unit = header_result.get("col_to_unit", {})
            level_headers = header_result.get("level_headers", {})
        else:
            # 回退策略：为每一列创建合成的HeaderCell，确保列语义与边关系完整
            synth_level = 0
            level_headers[synth_level] = []
            for c in range(num_columns):
                text = headers[c] if c < len(headers) else f"col_{c}"
                hc_id = self.get_next_node_id()
                self.graph.add_node(hc_id,
                                    type="HeaderCell",
                                    table_id=table_id,
                                    level=synth_level,
                                    position=c,
                                    text=text,
                                    span_start=c,
                                    span_end=c,
                                    span_size=1,
                                    is_leaf=True,
                                    doc_id=doc_id)
                self.graph.add_edge(table_node_id, hc_id, edge_type="CONTAINS_HEADER")
                level_headers[synth_level].append((c, c, text, hc_id))
                col_to_header_path[c] = [text]
        
        # 构建RowHeader节点（如果有多层行分组）
        row_to_group_path = {}
        level_row_headers = {}
        if rows and len(rows) > 0:
            row_header_result = self.build_row_headers(rows, table_id, doc_id, table_node_id, num_columns)
            row_to_group_path = row_header_result.get("row_to_group_path", {})
            level_row_headers = row_header_result.get("level_row_headers", {})
        
        # 优化：删除Column和Row节点，直接创建Cell节点
        # HeaderCell和RowHeader已经提供了完整的行列语义信息
        
        # 获取表头行数，用于计算原始TR索引
        header_rows_count = table_info.get("header_rows_count", 0)
        
        # 创建单元格节点（使用原始TR索引作为row坐标，以匹配数据集标注）
        for data_row_idx, row in enumerate(rows):
            # 原始TR索引 = 表头行数 + 数据行索引
            original_row_idx = header_rows_count + data_row_idx
            # 规范化行长度
            if len(row) < num_columns:
                row = row + [''] * (num_columns - len(row))
            elif len(row) > num_columns:
                row = row[:num_columns]
            
            # 创建单元格节点
            for col_idx in range(num_columns):
                cell_value = row[col_idx] if col_idx < len(row) else ''
                cell_node_id = self.get_next_node_id()
                
                # 获取该列的单位（优先列单位，其次表单位）
                cell_unit = unit
                cell_scale = scale
                if col_idx in col_to_unit:
                    cell_unit, cell_scale = col_to_unit[col_idx]
                
                # 标准化单元格值（保留原文本）
                numeric_value, original_text = self.normalize_cell_value(cell_value, cell_unit, cell_scale)
                
                # 获取该列的表头路径
                header_path = col_to_header_path.get(col_idx, [])
                col_path_str = ' > '.join(header_path) if header_path else ''
                
                # 获取该行的分组路径（使用数据行索引）
                row_group_path = row_to_group_path.get(data_row_idx, [])
                row_path_str = ' > '.join(row_group_path) if row_group_path else ''
                
                # 构建Cell节点属性
                cell_attrs = {
                    "type": "Cell",
                    "table_id": table_id,
                    "row": original_row_idx,  # 使用原始TR索引
                    "col": col_idx,
                    "value": original_text,
                    "header_path": header_path,  # 列表头路径（列表）
                    "row_group_path": row_group_path,  # 行分组路径（列表）
                    "col_path_str": col_path_str,  # 列表头路径（字符串，LLM友好）
                    "row_path_str": row_path_str,  # 行分组路径（字符串，LLM友好）
                    "doc_id": doc_id,
                    "full_path_list": row_group_path + header_path,
                    "full_path_str": (row_path_str + " | " if row_path_str else "") + col_path_str
                }
                
                # 添加数值相关属性
                if numeric_value is not None:
                    cell_attrs["numeric_value"] = numeric_value
                if cell_unit is not None:
                    cell_attrs["unit"] = cell_unit
                if cell_scale != 1.0:
                    cell_attrs["scale"] = cell_scale
                
                self.graph.add_node(cell_node_id, **cell_attrs)
                
                # 表格直接到单元格的关系（优化：删除Row/Column中介）
                self.graph.add_edge(table_node_id, cell_node_id, edge_type="CONTAINS_CELL")
                
                # HeaderCell到Cell的关系（提供列语义）
                if level_headers and len(level_headers) > 0:
                    # 优先选择叶子层覆盖当前列的HeaderCell；若未命中，向上回退
                    candidate_levels = sorted(level_headers.keys(), reverse=True)
                    linked = False
                    for lvl in candidate_levels:
                        for span_start, span_end, text, header_id in level_headers[lvl]:
                            if span_start <= col_idx <= span_end:
                                self.graph.add_edge(header_id, cell_node_id, edge_type="DEFINES_COLUMN")
                                linked = True
                                break
                        if linked:
                            break
                
                # RowHeader到Cell的关系（提供行语义，优化：直接连接Cell）
                if level_row_headers:
                    # 找到最底层（最具体）覆盖当前行的RowHeader（使用数据行索引匹配）
                    for level in sorted(level_row_headers.keys(), reverse=True):
                        for span_start, span_end, text, header_id, is_summary in level_row_headers[level]:
                            if span_start <= data_row_idx <= span_end:
                                self.graph.add_edge(header_id, cell_node_id, edge_type="DEFINES_ROW")
                                break
                        else:
                            continue
                        break
        
        return table_node_id
    
    def build_table_description_graph(self, table_descriptions: Dict[str, str], doc_id: str, 
                                    table_anchors: Dict[int, str]) -> List[str]:
        """构建表格描述子图"""
        description_nodes = []
        
        if not table_descriptions:
            return description_nodes
            
        for desc_key, desc_text in table_descriptions.items():
            # 解析描述键格式: "table_id-row-col" 
            if '-' in desc_key:
                try:
                    parts = desc_key.split('-')
                    if len(parts) >= 3:
                        table_id = int(parts[0])
                        row = int(parts[1])
                        col = int(parts[2])
                        
                        # 创建描述节点
                        desc_node_id = self.get_next_node_id()
                        self.graph.add_node(desc_node_id,
                                           type="TableDescription",
                                           table_id=table_id,
                                           row=row,
                                           col=col,
                                           description=desc_text,
                                           desc_key=desc_key,
                                           doc_id=doc_id)
                        
                        description_nodes.append(desc_node_id)
                        
                        # 优化：只连接到Cell节点（删除Table和Doc的冗余连接）
                        # 通过坐标定位对应的单元格
                        for node_id, node_data in self.graph.nodes(data=True):
                            if (node_data.get('type') == 'Cell' and 
                                node_data.get('table_id') == table_id and
                                node_data.get('row') == row and 
                                node_data.get('col') == col and
                                node_data.get('doc_id') == doc_id):
                                self.graph.add_edge(node_id, desc_node_id, edge_type="HAS_DESCRIPTION")
                                break
                                
                except (ValueError, IndexError) as e:
                    logger.warning(f"解析表格描述键失败: {desc_key}, 错误: {e}")
                    continue
        
        return description_nodes
    
    def build_text_graph(self, paragraphs: List[str], doc_id: str) -> List[str]:
        """构建文本子图 - 优化版：删除TextDoc中间节点"""
        para_nodes = []
        
        # 直接创建段落节点，不创建TextDoc
        for para_idx, paragraph in enumerate(paragraphs):
            para_node_id = self.get_next_node_id()
            self.graph.add_node(para_node_id,
                               type="Paragraph",
                               paragraph_index=para_idx,
                               index=para_idx,  # 保持兼容性
                               content=paragraph,
                               doc_id=doc_id)
            
            para_nodes.append(para_node_id)
        
        return para_nodes  # 返回段落节点列表
    
    def build_qa_graph(self, qa_data: Dict[str, Any], doc_id: str, table_anchors: Dict[int, str]) -> str:
        """构建问答子图

        规范化建模建议（默认启用 qa_decompose）：
        - QAInstance 节点代表一次问答样本实例
        - Question/Answer/Program/QuestionType 拆分为独立节点（便于检索/推理）
        - TableEvidence/TextEvidence 作为“证据声明”节点，再指向 Cell/Paragraph（便于监督训练）
        - 边命名统一使用动词短语、全大写、可读：
          HAS_QUESTION / HAS_ANSWER / HAS_PROGRAM / HAS_QUESTION_TYPE
          HAS_TABLE_EVIDENCE / HAS_TEXT_EVIDENCE
          POINTS_TO_CELL / POINTS_TO_PARAGRAPH / POINTS_TO_DESCRIPTION
          USES_TABLE
        """
        # 1) QAInstance (总入口)
        qa_instance_id = self.get_next_node_id()
        qa_instance_attrs = {
            "type": "QAInstance",
            "doc_id": doc_id
        }
        self.graph.add_node(qa_instance_id, **qa_instance_attrs)

        question_text = qa_data.get("question", "") or ""
        answer_obj = qa_data.get("answer", None)
        question_type = qa_data.get("question_type", "") or ""
        program = str(qa_data.get("program", "") or "")
        
        # 2) 拆分节点：Question / Answer / Program / QuestionType
        if self.qa_decompose:
            q_id = self.get_next_node_id()
            self.graph.add_node(q_id, type="Question", text=question_text, doc_id=doc_id)
            self.graph.add_edge(qa_instance_id, q_id, edge_type="HAS_QUESTION")

            a_id = self.get_next_node_id()
            # answer 可能是数值/字符串/列表等，这里统一存为 text/raw
            self.graph.add_node(a_id, type="Answer", text=str(answer_obj) if answer_obj is not None else "", raw=answer_obj, doc_id=doc_id)
            self.graph.add_edge(qa_instance_id, a_id, edge_type="HAS_ANSWER")

            if question_type:
                qt_id = self.get_next_node_id()
                self.graph.add_node(qt_id, type="QuestionType", name=str(question_type), doc_id=doc_id)
                self.graph.add_edge(qa_instance_id, qt_id, edge_type="HAS_QUESTION_TYPE")

            if program:
                p_id = self.get_next_node_id()
                self.graph.add_node(p_id, type="Program", text=program, doc_id=doc_id)
                self.graph.add_edge(qa_instance_id, p_id, edge_type="HAS_PROGRAM")

        # 3) 兼容旧节点：QuestionAnswer（可选保留）
        legacy_qa_id: Optional[str] = None
        if self.keep_legacy_qa_node:
            legacy_qa_id = self.get_next_node_id()
            qa_attrs = {
                "type": "QuestionAnswer",
                "question": question_text,
                "doc_id": doc_id,
                "answer": answer_obj,
                "question_type": question_type,
                "program": program,
                "table_evidence": qa_data.get("table_evidence", []) or [],
                "text_evidence": qa_data.get("text_evidence", []) or []
            }
            self.graph.add_node(legacy_qa_id, **qa_attrs)
            # 把 legacy QA 挂到 QAInstance 下（便于同时存在）
            self.graph.add_edge(qa_instance_id, legacy_qa_id, edge_type="HAS_LEGACY_QA")

        # 4) 证据节点建模（推荐）
        used_tables: set[int] = set()

        table_evi = qa_data.get("table_evidence", []) or []
        for ev in table_evi:
            if not (isinstance(ev, str) and '-' in ev):
                continue
            parts = ev.split('-')
            if len(parts) != 3:
                continue
            try:
                t_id, r, c = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue

            cell_id = self._find_cell_node(t_id, r, c, doc_id)
            if not cell_id:
                continue

            used_tables.add(t_id)

            # Evidence node
            if self.qa_decompose:
                te_id = self.get_next_node_id()
                self.graph.add_node(
                    te_id,
                    type="TableEvidence",
                    evidence_key=ev,
                    table_id=t_id,
                    row=r,
                    col=c,
                    doc_id=doc_id
                )
                self.graph.add_edge(qa_instance_id, te_id, edge_type="HAS_TABLE_EVIDENCE")
                self.graph.add_edge(te_id, cell_id, edge_type="POINTS_TO_CELL")

                desc_node_id = self._find_table_description_node(ev, doc_id)
                if desc_node_id:
                    self.graph.add_edge(te_id, desc_node_id, edge_type="POINTS_TO_DESCRIPTION")

            # 旧式直连边（保留用于兼容/快速监督抽取）
            if legacy_qa_id:
                self.graph.add_edge(legacy_qa_id, cell_id, edge_type="TABLE_EVIDENCE")
                desc_node_id = self._find_table_description_node(ev, doc_id)
                if desc_node_id:
                    self.graph.add_edge(legacy_qa_id, desc_node_id, edge_type="TABLE_DESCRIPTION_EVIDENCE")

        text_evi = qa_data.get("text_evidence", []) or []
        for idx in text_evi:
            if not isinstance(idx, int):
                continue
            para_id = self._find_paragraph_node(idx, doc_id)
            if not para_id:
                continue

            if self.qa_decompose:
                pe_id = self.get_next_node_id()
                self.graph.add_node(
                    pe_id,
                    type="TextEvidence",
                    paragraph_index=idx,
                    doc_id=doc_id
                )
                self.graph.add_edge(qa_instance_id, pe_id, edge_type="HAS_TEXT_EVIDENCE")
                self.graph.add_edge(pe_id, para_id, edge_type="POINTS_TO_PARAGRAPH")

            if legacy_qa_id:
                self.graph.add_edge(legacy_qa_id, para_id, edge_type="TEXT_EVIDENCE")

        # QAInstance / legacy QA -> Table（表格上下文）
        for t_id in used_tables:
            if t_id in table_anchors:
                if self.qa_decompose:
                    self.graph.add_edge(qa_instance_id, table_anchors[t_id], edge_type="USES_TABLE")
                if legacy_qa_id:
                    self.graph.add_edge(legacy_qa_id, table_anchors[t_id], edge_type="USES_TABLE")

        # 返回主入口：如果保留 legacy，则维持原来返回 legacy QA 以减少上游改动；否则返回 QAInstance
        return legacy_qa_id if legacy_qa_id is not None else qa_instance_id
    
    def _find_cell_node(self, table_id: int, row: int, col: int, doc_id: Optional[str] = None) -> Optional[str]:
        """
        在主图中查找指定坐标的Cell节点
        
        Args:
            table_id: 表格ID
            row: 行号
            col: 列号
            doc_id: 可选，文档ID，用于跨样本区分
            
        Returns:
            Cell节点ID，如果未找到则返回None
        """
        for node_id, node_data in self.graph.nodes(data=True):
            if (node_data.get('type') == 'Cell' and
                node_data.get('table_id') == table_id and
                node_data.get('row') == row and
                node_data.get('col') == col and
                (doc_id is None or node_data.get('doc_id') == doc_id)):
                return node_id
        return None
    
    def _find_paragraph_node(self, paragraph_index: int, doc_id: Optional[str] = None) -> Optional[str]:
        """在主图中查找指定段落索引的Paragraph节点"""
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('type') == 'Paragraph' and node_data.get('paragraph_index') == paragraph_index:
                if doc_id is None or node_data.get('doc_id') == doc_id:
                    return node_id
        return None

    def _find_table_description_node(self, desc_key: str, doc_id: Optional[str] = None) -> Optional[str]:
        """在主图中查找指定desc_key的TableDescription节点"""
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('type') == 'TableDescription' and node_data.get('desc_key') == desc_key:
                if doc_id is None or node_data.get('doc_id') == doc_id:
                    return node_id
        return None
    
    def _add_header_cell_to_gss(self, gss: nx.MultiDiGraph, main_hc_id: str, 
                                 hc_data: Dict, main_header_cells: Dict,
                                 gss_header_cells: Dict, gss_table_id: str,
                                 main_graph: nx.MultiDiGraph) -> str:
        """递归添加 HeaderCell 及其所有父节点到 GSS，同时保留NEXT_SIBLING关系"""
        # 如果已添加，直接返回
        if main_hc_id in gss_header_cells:
            return gss_header_cells[main_hc_id]
        
        # 复制当前 HeaderCell（包含所有新属性：is_leaf, span_size, role, time_value等）
        gss_hc_id = f"gss_hc_{len(gss.nodes())}"
        gss.add_node(gss_hc_id, **hc_data)
        gss_header_cells[main_hc_id] = gss_hc_id
        
        # 检查是否有父节点
        parent_id = hc_data.get('parent_id')
        if parent_id and parent_id in main_header_cells:
            # 递归添加父节点
            gss_parent_id = self._add_header_cell_to_gss(
                gss, parent_id, main_header_cells[parent_id],
                main_header_cells, gss_header_cells, gss_table_id, main_graph
            )
            # 建立 CHILD_HEADER 关系
            gss.add_edge(gss_parent_id, gss_hc_id, type="CHILD_HEADER")
        else:
            # 顶层节点，连接到 Table
            if gss_table_id:
                gss.add_edge(gss_table_id, gss_hc_id, type="CONTAINS_HEADER")
        
        # 检查并添加 NEXT_SIBLING_HEADER 关系
        for _, target, edge_data in main_graph.out_edges(main_hc_id, data=True):
            if edge_data.get('edge_type') == 'NEXT_SIBLING_HEADER':
                # 如果兄弟节点也在GSS中，建立关系
                if target in gss_header_cells:
                    gss.add_edge(gss_hc_id, gss_header_cells[target], type="NEXT_SIBLING_HEADER")
        
        return gss_hc_id
    
    def _add_row_header_to_gss(self, gss: nx.MultiDiGraph, main_rh_id: str,
                                rh_data: Dict, main_row_headers: Dict,
                                gss_row_headers: Dict, gss_table_id: str,
                                main_graph: nx.MultiDiGraph) -> str:
        """递归添加 RowHeader 及其所有父节点到 GSS，同时保留NEXT_SIBLING关系"""
        # 如果已添加，直接返回
        if main_rh_id in gss_row_headers:
            return gss_row_headers[main_rh_id]
        
        # 复制当前 RowHeader（包含所有新属性：is_leaf, span_size等）
        gss_rh_id = f"gss_rh_{len(gss.nodes())}"
        gss.add_node(gss_rh_id, **rh_data)
        gss_row_headers[main_rh_id] = gss_rh_id
        
        # 检查是否有父节点
        parent_id = rh_data.get('parent_id')
        if parent_id and parent_id in main_row_headers:
            # 递归添加父节点
            gss_parent_id = self._add_row_header_to_gss(
                gss, parent_id, main_row_headers[parent_id],
                main_row_headers, gss_row_headers, gss_table_id, main_graph
            )
            # 建立 CHILD_ROW_HEADER 关系
            gss.add_edge(gss_parent_id, gss_rh_id, type="CHILD_ROW_HEADER")
        else:
            # 顶层节点，连接到 Table
            if gss_table_id:
                gss.add_edge(gss_table_id, gss_rh_id, type="CONTAINS_ROW_HEADER")
        
        # 检查并添加 NEXT_SIBLING_ROW_HEADER 关系
        for _, target, edge_data in main_graph.out_edges(main_rh_id, data=True):
            if edge_data.get('edge_type') == 'NEXT_SIBLING_ROW_HEADER':
                # 如果兄弟节点也在GSS中，建立关系
                if target in gss_row_headers:
                    gss.add_edge(gss_rh_id, gss_row_headers[target], type="NEXT_SIBLING_ROW_HEADER")
        
        return gss_rh_id
    
    def build_gold_support_subgraph(self, sample: Dict[str, Any], main_graph_nodes: Dict[str, Any]) -> nx.MultiDiGraph:
        """构建金标准支持子图(GSS)"""
        gss = nx.MultiDiGraph()
        
        # 获取QA信息
        qa_info = sample.get('qa', {})
        if not qa_info:
            logger.warning(f"样本 {sample.get('uid', 'unknown')} 没有QA信息")
            return gss
        
        # 创建QA节点（补充所需元数据字段）
        qa_node_id = "gss_qa_1"
        uid = sample.get('uid', 'unknown')
        table_evidence = qa_info.get('table_evidence', []) or []
        text_evidence = qa_info.get('text_evidence', []) or []
        
        # 基于 table_evidence 收集 table_description 文本
        table_desc_obj = None
        if isinstance(sample.get('table_description'), dict):
            table_desc_obj = sample.get('table_description')
        elif isinstance(qa_info.get('table_description'), dict):
            table_desc_obj = qa_info.get('table_description')
        
        # 与 table_evidence 对齐的描述列表
        table_description_texts: List[str] = []
        if table_desc_obj and isinstance(table_evidence, list):
            for key in table_evidence:
                if isinstance(key, str) and key in table_desc_obj:
                    table_description_texts.append(str(table_desc_obj[key]))
                else:
                    table_description_texts.append("")
        
        # 简化QuestionAnswer节点属性 - 只保留核心字段
        qa_attrs = {
            "type": "QuestionAnswer",
            "question": qa_info.get("question", ""),
            "doc_id": sample.get('uid', 'unknown'),
            "answer": qa_info.get("answer", ""),
            "question_type": qa_info.get("question_type", ""),
            "program": str(qa_info.get("program", "")),
            "table_evidence": list(table_evidence),
            "text_evidence": list(text_evidence)
        }
        gss.add_node(qa_node_id, **qa_attrs)
        
        # 建立一个索引以便通过(table_id,row,col)定位主图Cell
        main_cells_by_coord: Dict[Tuple[int,int,int], Tuple[str, Dict[str, Any]]] = {}
        for node_id, node_data in main_graph_nodes.get('cell_nodes', []):
            try:
                key = (int(node_data.get('table_id')), int(node_data.get('row')), int(node_data.get('col')))
                main_cells_by_coord[key] = (node_id, node_data)
            except Exception:
                continue
        
        # ✨ 新增：收集主图中的 HeaderCell 和 RowHeader 节点
        main_header_cells: Dict[str, Dict[str, Any]] = {}
        main_row_headers: Dict[str, Dict[str, Any]] = {}
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('type') == 'HeaderCell':
                main_header_cells[node_id] = node_data
            elif node_data.get('type') == 'RowHeader':
                main_row_headers[node_id] = node_data
        
        # 记录已复制到GSS的Cell：coord -> gss_cell_id
        coord_to_gss_cell: Dict[Tuple[int,int,int], str] = {}
        # 记录已复制到GSS的TableDescription：evidence_key -> gss_desc_id
        evidence_to_gss_desc: Dict[str, str] = {}
        # 记录已复制到GSS的结构节点（优化：删除Row和Column）
        gss_table_nodes: Dict[int, str] = {}  # table_id -> gss_table_id
        # ✨ 新增：记录已复制到GSS的 HeaderCell 和 RowHeader
        gss_header_cells: Dict[str, str] = {}  # main_id -> gss_id
        gss_row_headers: Dict[str, str] = {}   # main_id -> gss_id
        
        # 获取table_description数据
        table_desc_obj = sample.get('table_description', {})
        
        # 建立结构节点索引（优化：只需Table节点）
        main_tables_by_id: Dict[int, Tuple[str, Dict[str, Any]]] = {}
        
        # 从主图中收集Table节点
        for node_id, node_data in main_graph_nodes.get('table_nodes', []):
            table_id = node_data.get('table_id')
            if table_id is not None:
                main_tables_by_id[table_id] = (node_id, node_data)
        
        # 处理表格证据
        if table_evidence:
            logger.info(f"表格证据: {table_evidence}")
            
            for evidence in table_evidence:
                # 解析证据格式 "table_id-row-col"
                if isinstance(evidence, str) and '-' in evidence:
                    parts = evidence.split('-')
                    if len(parts) == 3:
                        try:
                            table_id, row, col = int(parts[0]), int(parts[1]), int(parts[2])
                            coord = (table_id, row, col)
                            # 在主图中查找对应的Cell节点
                            if coord in main_cells_by_coord:
                                _src_id, node_data = main_cells_by_coord[coord]
                                # 若尚未复制该Cell到GSS，则复制并缓存
                                if coord not in coord_to_gss_cell:
                                    gss_cell_id = f"gss_cell_{len(gss.nodes())}"
                                    gss.add_node(gss_cell_id, **node_data)
                                    coord_to_gss_cell[coord] = gss_cell_id
                                
                                # 创建或获取Table节点（优化：删除Row/Column节点）
                                if table_id not in gss_table_nodes:
                                    if table_id in main_tables_by_id:
                                        _, table_data = main_tables_by_id[table_id]
                                        gss_table_id = f"gss_table_{len(gss.nodes())}"
                                        gss.add_node(gss_table_id, **table_data)
                                        gss_table_nodes[table_id] = gss_table_id
                                        logger.info(f"添加Table节点: table_{table_id}")
                                
                                # 建立关系（优化：Table直接连接Cell，删除Row/Column）
                                # Table -> Cell
                                if table_id in gss_table_nodes:
                                    gss.add_edge(gss_table_nodes[table_id], coord_to_gss_cell[coord], type="CONTAINS_CELL")
                                
                                # QuestionAnswer到Cell的证据关系
                                gss.add_edge(qa_node_id, coord_to_gss_cell[coord], type="SUPPORTS_ANSWER")
                                gss.add_edge(qa_node_id, coord_to_gss_cell[coord], type="TABLE_EVIDENCE")
                                
                                # QA到Table的上下文关系
                                if table_id in gss_table_nodes:
                                    gss.add_edge(qa_node_id, gss_table_nodes[table_id], type="USES_TABLE")
                                
                                logger.info(f"添加表格证据节点: {evidence}")
                                
                                # ✨ 新增：为该 Cell 添加关联的 HeaderCell
                                # 策略：通过列坐标找到所有覆盖该列的HeaderCell（所有层级）
                                added_header_for_cell = False
                                for main_hc_id, hc_data in main_header_cells.items():
                                    # 检查该HeaderCell是否覆盖当前列
                                    if (hc_data.get('table_id') == table_id and
                                        hc_data.get('span_start') <= col <= hc_data.get('span_end')):
                                        
                                        # 递归添加该 HeaderCell 及其所有祖先
                                        gss_hc_id = self._add_header_cell_to_gss(
                                            gss, main_hc_id, hc_data, main_header_cells,
                                            gss_header_cells, gss_table_nodes.get(table_id), self.graph
                                        )
                                        
                                        # 只为叶节点建立 HEADER_OF 关系
                                        if hc_data.get('is_leaf', False):
                                            gss.add_edge(gss_hc_id, coord_to_gss_cell[coord], type="DEFINES_COLUMN")
                                            logger.info(f"  添加 HeaderCell (叶节点): {hc_data.get('text', '')} (level={hc_data.get('level', 0)})")
                                            added_header_for_cell = True
                                        else:
                                            logger.info(f"  添加 HeaderCell (父节点): {hc_data.get('text', '')} (level={hc_data.get('level', 0)})")
                                
                                if not added_header_for_cell:
                                    logger.warning(f"  ⚠️  Cell ({table_id}, {row}, {col}) 没有找到对应的叶节点HeaderCell")
                                
                                # ✨ 新增：为该 Cell 添加关联的 RowHeader
                                # 策略：通过行坐标找到所有覆盖该行的RowHeader（所有层级）
                                added_row_header_for_cell = False
                                for main_rh_id, rh_data in main_row_headers.items():
                                    # 检查该RowHeader是否覆盖当前行
                                    # 注意：RowHeader的span是基于data_row_idx，需要转换
                                    if rh_data.get('table_id') == table_id:
                                        # 获取表头行数以转换坐标
                                        header_rows_count = 0
                                        for node_id, node_data in self.graph.nodes(data=True):
                                            if (node_data.get('type') == 'Table' and 
                                                node_data.get('table_id') == table_id):
                                                # 从table_info中获取header_rows_count（如果有的话）
                                                break
                                        
                                        # 使用原始行号匹配（因为Cell使用original_row_idx）
                                        span_start = rh_data.get('span_start', -1)
                                        span_end = rh_data.get('span_end', -1)
                                        
                                        # RowHeader的span是基于数据行索引，需要检查是否覆盖
                                        # 这里简化处理：如果RowHeader有parent_id，说明是子节点，直接检查span
                                        if span_start <= row <= span_end or (span_start == row == span_end):
                                            # 递归添加该 RowHeader 及其所有祖先
                                            gss_rh_id = self._add_row_header_to_gss(
                                                gss, main_rh_id, rh_data, main_row_headers,
                                                gss_row_headers, gss_table_nodes.get(table_id), self.graph
                                            )
                                            
                                            # 只为叶节点建立 ROW_HEADER_OF 关系
                                            if rh_data.get('is_leaf', False):
                                                gss.add_edge(gss_rh_id, coord_to_gss_cell[coord], type="DEFINES_ROW")
                                                logger.info(f"  添加 RowHeader (叶节点): {rh_data.get('text', '')} (level={rh_data.get('level', 0)})")
                                                added_row_header_for_cell = True
                                            else:
                                                logger.info(f"  添加 RowHeader (父节点): {rh_data.get('text', '')} (level={rh_data.get('level', 0)})")
                                
                                # 如果没有找到RowHeader，尝试从同行第0列的Cell获取行标签
                                if not added_row_header_for_cell:
                                    # 查找同行第0列的Cell，其value可能是行标签
                                    row_label_cell = self._find_cell_node(table_id, row, 0)
                                    if row_label_cell:
                                        row_label_data = self.graph.nodes[row_label_cell]
                                        row_label_text = row_label_data.get('value', '').strip()
                                        
                                        # 如果第0列有文本且不是数字，将其作为行标签
                                        if row_label_text and not row_label_text.replace('.', '').replace(',', '').replace('$', '').replace('-', '').replace('%', '').strip().isdigit():
                                            # 创建一个临时的RowHeader节点
                                            temp_rh_id = f"gss_temp_rh_{len(gss.nodes())}"
                                            temp_rh_attrs = {
                                                "type": "RowHeader",
                                                "table_id": table_id,
                                                "level": 0,
                                                "position": row,
                                                "text": row_label_text,
                                                "span_start": row,
                                                "span_end": row,
                                                "span_size": 1,
                                                "is_summary": False,
                                                "is_leaf": True,
                                                "doc_id": node_data.get("doc_id", ""),
                                                "is_temp": True  # 标记为临时创建的
                                            }
                                            gss.add_node(temp_rh_id, **temp_rh_attrs)
                                            
                                            # 连接到Table
                                            if table_id in gss_table_nodes:
                                                gss.add_edge(gss_table_nodes[table_id], temp_rh_id, type="CONTAINS_ROW_HEADER")
                                            
                                            # 连接到Cell
                                            gss.add_edge(temp_rh_id, coord_to_gss_cell[coord], type="DEFINES_ROW")
                                            
                                            logger.info(f"  ✨ 为Cell ({table_id}, {row}, {col}) 创建临时RowHeader: {row_label_text}")
                                            added_row_header_for_cell = True
                                    
                                    if not added_row_header_for_cell:
                                        logger.info(f"  ℹ️  Cell ({table_id}, {row}, {col}) 没有对应的RowHeader（可能是普通数据行）")
                                
                                # 添加对应的TableDescription节点（如果存在）
                                if evidence in table_desc_obj and evidence not in evidence_to_gss_desc:
                                    desc_text = table_desc_obj[evidence]
                                    gss_desc_id = f"gss_desc_{len(gss.nodes())}"
                                    
                                    # 创建TableDescription节点
                                    desc_attrs = {
                                        "type": "TableDescription",
                                        "table_id": table_id,
                                        "row": row,
                                        "col": col,
                                        "description": desc_text,
                                        "desc_key": evidence,
                                        "doc_id": node_data.get("doc_id", ""),
                                        "is_evidence": True,
                                        "evidence_type": "table_description"
                                    }
                                    gss.add_node(gss_desc_id, **desc_attrs)
                                    evidence_to_gss_desc[evidence] = gss_desc_id
                                    
                                    # 建立关系：Cell -> TableDescription
                                    gss.add_edge(coord_to_gss_cell[coord], gss_desc_id, type="HAS_DESCRIPTION")
                                    
                                    # 建立关系：QA -> TableDescription（优化：只保留一个语义证据关系）
                                    gss.add_edge(qa_node_id, gss_desc_id, type="SEMANTIC_EVIDENCE")
                                    
                                    # 删除TableDescription到结构节点的冗余关系（坐标属性已包含位置信息）
                                    
                                    logger.info(f"添加表格描述节点: {evidence} -> {desc_text[:50]}...")
                                    
                            else:
                                # 证据坐标不存在，尝试提供更详细的信息
                                logger.warning(f"❌ 未找到表格证据节点: {evidence}")
                                logger.warning(f"   可能原因: 数据集标注错误或表格解析问题")
                                
                                # 尝试查找附近的单元格
                                nearby_cells = []
                                for check_coord in main_cells_by_coord.keys():
                                    if check_coord[0] == table_id and abs(check_coord[1] - row) <= 1 and abs(check_coord[2] - col) <= 1:
                                        nearby_cells.append(check_coord)
                                
                                if nearby_cells:
                                    logger.warning(f"   附近存在的单元格坐标: {nearby_cells[:5]}")
                                    logger.warning(f"   建议检查原始数据或手动修正证据标注")
                                else:
                                    logger.warning(f"   Table {table_id} 的所有Cell坐标范围未知，请检查表格解析结果")
                        except ValueError as e:
                            logger.warning(f"解析表格证据失败: {evidence}, 错误: {e}")
        
        # 处理文本证据
        if text_evidence:
            logger.info(f"文本证据: {text_evidence}")
            
            for evidence in text_evidence:
                if isinstance(evidence, int):
                    # 在主图中查找对应的Paragraph节点
                    for node_id, node_data in main_graph_nodes.get('paragraph_nodes', []):
                        if node_data.get('paragraph_index') == evidence:
                            # 复制Paragraph节点到GSS并增强属性
                            gss_para_id = f"gss_paragraph_{len(gss.nodes())}"
                            enhanced_para_data = dict(node_data)
                            enhanced_para_data.update({
                                "is_evidence": True,
                                "evidence_type": "text_evidence",
                                "paragraph_length": len(node_data.get('content', '')),
                                "evidence_index": evidence
                            })
                            gss.add_node(gss_para_id, **enhanced_para_data)
                            
                            # 创建QuestionAnswer到Paragraph的证据关系（优化：删除冗余关系）
                            gss.add_edge(qa_node_id, gss_para_id, type="SUPPORTS_ANSWER")
                            gss.add_edge(qa_node_id, gss_para_id, type="TEXT_EVIDENCE")
                            # 删除REFERENCES关系（与EVIDENCE重复）
                            
                            logger.info(f"添加文本证据节点: {evidence}")
                            break
                    else:
                        logger.warning(f"未找到文本证据节点: {evidence}")
        
        # 创建Doc -> QA的关系
        for node_id, node_data in main_graph_nodes.get('doc_nodes', []):
            gss_doc_id = f"gss_doc_{len(gss.nodes())}"
            # 只保留type和doc_id属性
            simple_doc_data = {
                "type": "Doc",
                "doc_id": node_data.get("doc_id", "unknown")
            }
            gss.add_node(gss_doc_id, **simple_doc_data)
            
            # Doc到QA的关系（优化：只保留一个核心关系，删除冗余）
            gss.add_edge(gss_doc_id, qa_node_id, type="HAS_QUESTION")
        
        logger.info(f"GSS构建完成: {gss.number_of_nodes()} 个节点, {gss.number_of_edges()} 条边")
        return gss
    
    def export_gss_to_neo4j_cypher(self, gss: nx.MultiDiGraph, filepath: str, sample_prefix: str = None):
        """导出GSS为Neo4j Cypher格式（包含关系）
        包含新增的属性和边关系，优化给LLM使用
        
        Args:
            gss: GSS图对象
            filepath: 输出文件路径
            sample_prefix: 样本前缀（如 "gss_1_7d840731..."），用于生成全局唯一的节点ID
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("// Gold Standard Support Subgraph (GSS) - Enhanced for LLM\n")
            f.write("// Includes: is_leaf, span_size, role, time_value, col_path_str, row_path_str\n")
            f.write("// Edge types: NEXT_SIBLING_HEADER, NEXT_SIBLING_ROW_HEADER\n\n")
            
            # 创建节点（为每个节点写入唯一id属性，使用带前缀的node_id）
            for node_id, node_data in gss.nodes(data=True):
                props = []
                node_type = node_data.get('type', 'Unknown')
                
                # 生成带前缀的全局唯一 ID
                if sample_prefix:
                    global_id = f"{sample_prefix}|{node_id}"
                else:
                    global_id = str(node_id)
                
                # 为QuestionAnswer节点使用简化格式
                if node_type == 'QuestionAnswer':
                    # 按指定顺序添加QA节点属性（使用全局唯一ID）
                    props.append(f"id: {repr(global_id)}")
                    if 'question' in node_data and node_data['question']:
                        question = self.clean_string_for_cypher(node_data['question'])
                        props.append(f'question: "{question}"')
                    if 'doc_id' in node_data and node_data['doc_id']:
                        props.append(f'doc_id: "{node_data["doc_id"]}"')
                    if 'answer' in node_data and node_data['answer'] is not None:
                        answer = node_data['answer']
                        if isinstance(answer, (int, float)):
                            props.append(f'answer: {answer}')
                        else:
                            answer_str = self.clean_string_for_cypher(str(answer))
                            props.append(f'answer: "{answer_str}"')
                    if 'question_type' in node_data and node_data['question_type']:
                        props.append(f'question_type: "{node_data["question_type"]}"')
                    if 'program' in node_data and node_data['program']:
                        program = self.clean_string_for_cypher(node_data['program'])
                        # 限制program长度
                        if len(program) > 300:
                            program = program[:300] + '...'
                        props.append(f'program: "{program}"')
                    if 'table_evidence' in node_data and node_data['table_evidence']:
                        # 将列表转换为用" | "分隔的字符串
                        evidence_str = ' | '.join([str(item) for item in node_data['table_evidence']])
                        props.append(f'table_evidence: "{evidence_str}"')
                    if 'text_evidence' in node_data and node_data['text_evidence']:
                        # 将列表转换为用" | "分隔的字符串
                        text_evidence_str = ' | '.join([str(item) for item in node_data['text_evidence']])
                        props.append(f'text_evidence: "{text_evidence_str}"')
                else:
                    # 其他节点类型保持原有格式（使用全局唯一ID）
                    props.append(f"id: {repr(global_id)}")
                    for key, value in node_data.items():
                        if key == 'type' or value is None:
                            continue
                        
                        # 跳过冗余属性：Table的headers（HeaderCell已包含），scale可选
                        if node_type == 'Table' and key in ['headers', 'scale']:
                            continue
                        
                        if isinstance(value, str):
                            value = self.clean_string_for_cypher(value)
                            # 限制字符串长度为500字符，避免单行过长
                            if len(value) > 500:
                                value = value[:500] + '...'
                            props.append(f'{key}: "{value}"')
                        elif isinstance(value, list):
                            # 将列表转换为用" | "分隔的字符串，避免过长换行问题
                            if len(value) > 10:
                                # 列表太长，跳过
                                continue
                            if len(value) == 0:
                                # 空列表用空字符串表示
                                props.append(f'{key}: ""')
                            else:
                                # 将列表元素用" | "连接成字符串
                                cleaned_items = [self.clean_string_for_cypher(str(item)[:100]) for item in value]
                                list_str = ' | '.join(cleaned_items)
                                # 限制总长度
                                if len(list_str) > 300:
                                    list_str = list_str[:300] + '...'
                                props.append(f'{key}: "{list_str}"')
                        elif isinstance(value, bool):
                            # 将Python布尔值转换为Neo4j格式
                            props.append(f'{key}: {str(value).lower()}')
                        elif isinstance(value, dict):
                            # 跳过字典类型（通常过于复杂）
                            continue
                        else:
                            props.append(f'{key}: {value}')
                
                props_str = ', '.join(props)
                f.write(f'CREATE (:{node_type} {{{props_str}}});\n')
            
            f.write("\n// 创建证据关系\n")
            # 创建关系，通过全局唯一 id 匹配
            for source, target, edge_data in gss.edges(data=True):
                rel_type = edge_data.get('type', 'RELATED_TO').upper()
                # 生成带前缀的源和目标ID
                if sample_prefix:
                    source_id = f"{sample_prefix}|{source}"
                    target_id = f"{sample_prefix}|{target}"
                else:
                    source_id = str(source)
                    target_id = str(target)
                f.write(f"MATCH (a {{id: {repr(source_id)}}}), (b {{id: {repr(target_id)}}}) CREATE (a)-[:{rel_type}]->(b);\n")
    
    def convert_sample(self, sample: Dict[str, Any]) -> nx.MultiDiGraph:
        """转换单个样本为图"""
        doc_id = sample.get("uid", f"doc_{self.node_counter}")
        # 设置当前样本的前缀，确保 node_id 在全局唯一
        self.current_doc_prefix = self._sanitize_doc_id_for_prefix(doc_id)
        # 重置节点计数器，使每个样本的节点编号从1开始（前缀保证全局唯一性）
        self.node_counter = 0
        
        # 创建文档节点
        doc_node_id = self.get_next_node_id()
        self.graph.add_node(doc_node_id, type="Doc", doc_id=doc_id)
        
        # These will store mappings from evidence identifiers to graph node IDs for this sample
        cell_nodes_map = {} # key: "table_id-row-col", value: node_id
        para_nodes_map = {} # key: paragraph_index, value: node_id
        
        # 构建表格子图
        table_anchors = {}
        if "tables" in sample:
            for table_id, table_html in enumerate(sample["tables"]):
                table_info = self.parse_html_table(table_html, table_id)
                table_node_id = self.build_table_graph(table_info, doc_id)
                table_anchors[table_id] = table_node_id
                
                # 文档到表格的关系
                self.graph.add_edge(doc_node_id, table_node_id, edge_type="HAS_TABLE")
        
        # 构建文本子图 - 优化：直接连接Paragraph到Doc
        if "paragraphs" in sample:
            para_nodes = self.build_text_graph(sample["paragraphs"], doc_id)
            # 文档到段落的直接关系（删除TextDoc中间层）
            for para_node_id in para_nodes:
                self.graph.add_edge(doc_node_id, para_node_id, edge_type="CONTAINS_PARAGRAPH")
        
        # 构建表格描述子图 - 优化：删除Doc到TableDescription的冗余连接
        if "table_description" in sample:
            self.build_table_description_graph(sample["table_description"], doc_id, table_anchors)
            # TableDescription只通过Cell连接，不需要Doc的直接连接
        
        # 构建问答子图
        if "qa" in sample:
            qa_node_id = self.build_qa_graph(sample["qa"], doc_id, table_anchors)
            # 文档到问答的关系
            self.graph.add_edge(doc_node_id, qa_node_id, edge_type="HAS_QUESTION")
        
        return self.graph
    
    def convert_samples(self, samples: List[Dict[str, Any]]) -> nx.MultiDiGraph:
        """转换多个样本为统一图"""
        self.graph = nx.MultiDiGraph()  # 重置图
        self.node_counter = 0
        
        for sample in samples:
            self.convert_sample(sample)
        
        return self.graph
    
    def clean_string_for_cypher(self, text: str) -> str:
        """清理字符串中的特殊字符，使其适合Cypher脚本"""
        if not isinstance(text, str):
            return str(text)
        
        # 移除或替换Unicode控制字符
        import re
        # 移除常见的控制字符，包括\u0002等
        text = re.sub(r'[\u0000-\u001F\u007F-\u009F]', '', text)
        
        # 归一化常见的Unicode标点，避免高亮器/解析器误判
        replacements = {
            '\u00A0': ' ',   # NBSP -> space
            '\u2010': '-',   # hyphen
            '\u2011': '-',   # non-breaking hyphen
            '\u2012': '-',   # figure dash
            '\u2013': '-',   # en dash
            '\u2014': '-',   # em dash
            '\u2015': '-',   # horizontal bar
            '\u2026': '...', # ellipsis
            '\u2022': '-',   # bullet
            '\u2018': "'",  # left single quote
            '\u2019': "'",  # right single quote
            '\u201C': '"',  # left double quote
            '\u201D': '"',  # right double quote
        }
        for src, dst in replacements.items():
            text = text.replace(src, dst)
        
        # 转义Cypher中的特殊字符
        text = text.replace('\\', '\\\\')  # 反斜杠
        text = text.replace('"', '\\"')    # 双引号
        text = text.replace('\n', '\\n')   # 换行符
        text = text.replace('\r', '\\r')   # 回车符
        text = text.replace('\t', '\\t')   # 制表符
        
        return text

    def export_to_edgelist(self, filepath: str):
        """导出为边列表格式"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for u, v, data in self.graph.edges(data=True):
                edge_type = data.get('edge_type', 'RELATED_TO')
                f.write(f"{u}\t{v}\t{edge_type}\n")
    
    def export_to_neo4j_cypher(self, filepath: str):
        """导出为Neo4j Cypher格式（按样本分组：每个样本的所有节点后跟该样本的所有边关系）
        包含新增的属性：is_leaf, span_size, role, time_value, col_path_str, row_path_str等
        包含新增的边关系：NEXT_SIBLING_HEADER, NEXT_SIBLING_ROW_HEADER
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("// Neo4j Cypher Script - Enhanced Graph Structure for LLM\n")
            f.write("// Includes: is_leaf, span_size, role, time_value, col_path_str, row_path_str\n")
            f.write("// Edge types: NEXT_SIBLING_HEADER, NEXT_SIBLING_ROW_HEADER\n\n")
            
            # 按doc_id分组节点和边
            samples_data = {}
            
            # 收集所有节点，按doc_id分组
            for node_id, node_data in self.graph.nodes(data=True):
                doc_id = node_data.get('doc_id', 'unknown')
                if doc_id not in samples_data:
                    samples_data[doc_id] = {'nodes': [], 'edges': []}
                samples_data[doc_id]['nodes'].append((node_id, node_data))
            
            # 收集所有边，按源节点的doc_id分组
            for u, v, edge_data in self.graph.edges(data=True):
                u_data = self.graph.nodes[u]
                doc_id = u_data.get('doc_id', 'unknown')
                if doc_id in samples_data:
                    samples_data[doc_id]['edges'].append((u, v, edge_data))
            
            # 按样本顺序输出
            sample_items = list(samples_data.items())
            for sample_idx, (doc_id, sample_data) in enumerate(sample_items, 1):
                f.write(f"// ========== Sample {sample_idx}: {doc_id} ==========\n")
                f.write(f"// Sample {sample_idx} - Creating all nodes\n")
                
                # 输出该样本的所有节点
                for node_id, node_data in sample_data['nodes']:
                    props = []
                    node_type = node_data.get('type', 'Unknown')
                    
                    for key, value in node_data.items():
                        if key != 'type' and value is not None:
                            # 跳过冗余属性：Table的headers（HeaderCell已包含）
                            if node_type == 'Table' and key in ['headers']:
                                continue
                            
                            if isinstance(value, str):
                                # 转义字符串中的引号和换行符，并限制长度避免过长
                                value = self.clean_string_for_cypher(value)
                                # 限制字符串长度为500字符，避免单行过长
                                if len(value) > 500:
                                    value = value[:500] + '...'
                                props.append(f'{key}: "{value}"')
                            elif isinstance(value, list):
                                # 将列表转换为用" | "分隔的字符串，避免过长换行问题
                                if len(value) > 10:
                                    # 列表太长，跳过
                                    continue
                                if len(value) == 0:
                                    # 空列表用空字符串表示
                                    props.append(f'{key}: ""')
                                else:
                                    # 将列表元素用" | "连接成字符串
                                    cleaned_items = [self.clean_string_for_cypher(str(item)[:100]) for item in value]
                                    list_str = ' | '.join(cleaned_items)
                                    # 限制总长度
                                    if len(list_str) > 300:
                                        list_str = list_str[:300] + '...'
                                    props.append(f'{key}: "{list_str}"')
                            elif isinstance(value, bool):
                                # 将Python布尔值转换为Neo4j格式
                                props.append(f'{key}: {str(value).lower()}')
                            elif isinstance(value, dict):
                                # 跳过字典类型（通常过于复杂）
                                continue
                            else:
                                props.append(f'{key}: {value}')
                    
                    props_str = ', '.join(props)
                    f.write(f'CREATE (:{node_type} {{id: {repr(node_id)}, {props_str}}});\n')
                
                f.write(f"\n// Sample {sample_idx} - Creating all relationships\n")
                
                # 输出该样本的所有边关系
                for u, v, edge_data in sample_data['edges']:
                    edge_type = edge_data.get('edge_type', 'RELATED_TO').upper()
                    f.write(f"MATCH (a {{id: {repr(u)}}}), (b {{id: {repr(v)}}}) CREATE (a)-[:{edge_type}]->(b);\n")
                
                # Only add comment between samples, not after the last one  
                if sample_idx < len(sample_items):
                    f.write(f"\n// End of Sample {sample_idx}\n\n")
                else:
                    f.write(f"// End of Sample {sample_idx}\n")
    def process_dataset_batch(self, dataset_path: str, output_dir: str, dataset_name: str, 
                            generate_gss: bool = True, max_samples: int = None) -> Dict[str, Any]:
        """
        批量处理数据集
        
        Args:
            dataset_path: 数据集文件路径
            output_dir: 输出目录
            dataset_name: 数据集名称 (train/dev/test)
            generate_gss: 是否生成GSS子图
            max_samples: 最大处理样本数，None表示处理全部
            
        Returns:
            处理结果统计信息
        """
        logger.info(f"开始处理 {dataset_name} 数据集: {dataset_path}")
        
        # 读取数据
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
            
        logger.info(f"数据集 {dataset_name} 包含 {len(data)} 个样本")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 转换为图
        self.graph = nx.MultiDiGraph()  # 重置图
        self.node_counter = 0
        
        # 处理所有样本
        for i, sample in enumerate(data):
            if i % 100 == 0:
                logger.info(f"处理进度: {i+1}/{len(data)}")
            self.convert_sample(sample)
        
        logger.info(f"图构建完成: {self.graph.number_of_nodes()} 个节点, {self.graph.number_of_edges()} 条边")
        
        # 导出主图
        graph_edgelist_path = os.path.join(output_dir, f"{dataset_name}_graph_edgelist.txt")
        graph_cypher_path = os.path.join(output_dir, f"{dataset_name}_graph.cypher")
        
        self.export_to_edgelist(graph_edgelist_path)
        self.export_to_neo4j_cypher(graph_cypher_path)
        
        result = {
            "dataset_name": dataset_name,
            "total_samples": len(data),
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges(),
            "graph_edgelist_path": graph_edgelist_path,
            "graph_cypher_path": graph_cypher_path,
            "gss_generated": False,
            "gss_cypher_path": None,
            "gss_statistics": None
        }
        
        # 生成GSS子图（如果需要）
        if generate_gss:
            logger.info("开始生成GSS子图...")
            gss_cypher_path = os.path.join(output_dir, f"{dataset_name}_gss_all.cypher")
            gss_stats_path = os.path.join(output_dir, f"{dataset_name}_gss_statistics.json")
            
            gss_statistics = self.generate_all_gss(data, gss_cypher_path, gss_stats_path)
            
            result.update({
                "gss_generated": True,
                "gss_cypher_path": gss_cypher_path,
                "gss_statistics_path": gss_stats_path,
                "gss_statistics": gss_statistics
            })
        
        logger.info(f"数据集 {dataset_name} 处理完成")
        return result

    def generate_all_gss(self, samples: List[Dict[str, Any]], output_cypher_path: str, 
                        stats_path: str) -> Dict[str, Any]:
        """
        为所有样本生成GSS子图并合并到单个Cypher文件
        
        Args:
            samples: 样本列表
            output_cypher_path: 输出的合并Cypher文件路径
            stats_path: 统计信息文件路径
            
        Returns:
            GSS生成统计信息
        """
        all_gss_content = []
        statistics = {
            "total_samples": len(samples),
            "successful_gss": 0,
            "failed_gss": 0,
            "gss_details": []
        }
        
        # 为每个样本生成GSS
        for i, sample in enumerate(samples):
            if i % 100 == 0:
                logger.info(f"GSS生成进度: {i+1}/{len(samples)}")
                
            try:
                # 重新转换单个样本以获取节点信息
                temp_converter = TableToGraphConverter()
                temp_graph = temp_converter.convert_sample(sample)
                
                # 收集节点信息（优化：删除Row和Column节点）
                main_graph_nodes = {
                    'cell_nodes': [],
                    'paragraph_nodes': [],
                    'qa_nodes': [],
                    'doc_nodes': [],
                    'table_nodes': [],
                    'text_doc_nodes': []
                }
                
                for node_id, node_data in temp_graph.nodes(data=True):
                    node_type = node_data.get('type', '')
                    if node_type == 'Cell':
                        main_graph_nodes['cell_nodes'].append((node_id, node_data))
                    elif node_type == 'Paragraph':
                        main_graph_nodes['paragraph_nodes'].append((node_id, node_data))
                    elif node_type == 'QuestionAnswer':
                        main_graph_nodes['qa_nodes'].append((node_id, node_data))
                    elif node_type == 'Doc':
                        main_graph_nodes['doc_nodes'].append((node_id, node_data))
                    elif node_type == 'Table':
                        main_graph_nodes['table_nodes'].append((node_id, node_data))
                    elif node_type == 'TextDoc':
                        main_graph_nodes['text_doc_nodes'].append((node_id, node_data))
                
                # 构建GSS
                gss = temp_converter.build_gold_support_subgraph(sample, main_graph_nodes)
                
                if gss.number_of_nodes() > 0:
                    # 生成Cypher内容
                    cypher_lines = []
                    uid = sample.get('uid', 'unknown')
                    prefix = f"gss_{i+1}_{uid}"
                    cypher_lines.append(f"// GSS for sample {i+1} (uid: {uid})")
                    
                    # 节点id映射（避免跨样本冲突）
                    # original_node_id -> combined_id
                    id_map: Dict[str, str] = {}
                    for node_id, node_data in gss.nodes(data=True):
                        combined_id = f"{prefix}|{str(node_id)}"
                        id_map[str(node_id)] = combined_id
                        props = []
                        node_type = node_data.get('type', 'Unknown')
                        
                        # 为QuestionAnswer节点使用简化格式
                        if node_type == 'QuestionAnswer':
                            props.append(f"id: {repr(combined_id)}")
                            if 'question' in node_data and node_data['question']:
                                question = self.clean_string_for_cypher(node_data['question'])
                                props.append(f'question: "{question}"')
                            if 'doc_id' in node_data and node_data['doc_id']:
                                props.append(f'doc_id: "{node_data["doc_id"]}"')
                            if 'answer' in node_data and node_data['answer'] is not None:
                                answer = node_data['answer']
                                if isinstance(answer, (int, float)):
                                    props.append(f'answer: {answer}')
                                else:
                                    answer_str = self.clean_string_for_cypher(str(answer))
                                    props.append(f'answer: "{answer_str}"')
                            if 'question_type' in node_data and node_data['question_type']:
                                props.append(f'question_type: "{node_data["question_type"]}"')
                            if 'program' in node_data and node_data['program']:
                                program = self.clean_string_for_cypher(node_data['program'])
                                # 限制program长度
                                if len(program) > 300:
                                    program = program[:300] + '...'
                                props.append(f'program: "{program}"')
                            if 'table_evidence' in node_data and node_data['table_evidence']:
                                # 将列表转换为用" | "分隔的字符串
                                evidence_str = ' | '.join([str(item) for item in node_data['table_evidence']])
                                props.append(f'table_evidence: "{evidence_str}"')
                            if 'text_evidence' in node_data and node_data['text_evidence']:
                                # 将列表转换为用" | "分隔的字符串
                                text_evidence_str = ' | '.join([str(item) for item in node_data['text_evidence']])
                                props.append(f'text_evidence: "{text_evidence_str}"')
                        else:
                            # 其他节点类型保持原有格式
                            props.append(f"id: {repr(combined_id)}")
                            for key, value in node_data.items():
                                if key == 'type' or value is None:
                                    continue
                                if isinstance(value, str):
                                    value = self.clean_string_for_cypher(value)
                                    props.append(f'{key}: "{value}"')
                                elif isinstance(value, list):
                                    cleaned_items = [self.clean_string_for_cypher(str(item)) for item in value]
                                    list_str = ', '.join([f'"{item}"' for item in cleaned_items])
                                    props.append(f'{key}: [{list_str}]')
                                elif isinstance(value, bool):
                                    # 将Python布尔值转换为Neo4j格式
                                    props.append(f'{key}: {str(value).lower()}')
                                elif isinstance(value, dict):
                                    # 将Python字典转换为Neo4j格式（使用双引号）
                                    dict_str = json.dumps(value)
                                    props.append(f'{key}: {dict_str}')
                                else:
                                    props.append(f'{key}: {value}')
                        
                        props_str = ', '.join(props)
                        cypher_lines.append(f'CREATE (:{node_type} {{{props_str}}});')
                    
                    # 关系：用 combined id 匹配并创建
                    for source, target, edge_data in gss.edges(data=True):
                        rel_type = edge_data.get('type', 'RELATED_TO').upper()
                        s_id = id_map[str(source)]
                        t_id = id_map[str(target)]
                        cypher_lines.append(f'MATCH (a {{id: {repr(s_id)}}}), (b {{id: {repr(t_id)}}}) CREATE (a)-[:{rel_type}]->(b);')
                    
                    cypher_lines.append("")  # 空行分隔
                    
                    all_gss_content.extend(cypher_lines)
                    statistics["successful_gss"] += 1
                    
                    # 记录详细信息
                    gss_detail = {
                        "sample_index": i,
                        "uid": uid,
                        "nodes": gss.number_of_nodes(),
                        "edges": gss.number_of_edges(),
                        "status": "success"
                    }
                    statistics["gss_details"].append(gss_detail)
                    
                else:
                    statistics["failed_gss"] += 1
                    gss_detail = {
                        "sample_index": i,
                        "uid": sample.get('uid', 'unknown'),
                        "nodes": 0,
                        "edges": 0,
                        "status": "failed - no nodes"
                    }
                    statistics["gss_details"].append(gss_detail)
                    
            except Exception as e:
                logger.warning(f"样本 {i} GSS生成失败: {str(e)}")
                statistics["failed_gss"] += 1
                gss_detail = {
                    "sample_index": i,
                    "uid": sample.get('uid', 'unknown'),
                    "nodes": 0,
                    "edges": 0,
                    "status": f"failed - {str(e)}"
                }
                statistics["gss_details"].append(gss_detail)
        
        # 写入合并的Cypher文件
        with open(output_cypher_path, 'w', encoding='utf-8') as f:
            f.write("// Combined GSS Cypher file\n")
            f.write(f"// Total samples: {len(samples)}\n")
            f.write(f"// Successful GSS: {statistics['successful_gss']}\n")
            f.write(f"// Failed GSS: {statistics['failed_gss']}\n\n")
            f.write('\n'.join(all_gss_content))
        
        # 保存统计信息
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"GSS生成完成: 成功 {statistics['successful_gss']}, 失败 {statistics['failed_gss']}")
        return statistics


def process_all_datasets():
    """处理所有MultiHiertt数据集"""
    base_path = "/home/cqjtu/NLP-Group/LZH/TTGR_llm/dataset/MultiHiertt"
    output_base = "/home/cqjtu/NLP-Group/LZH/TTGR_llm/TextTableToGraph/outputs"
    
    datasets = [
        {"name": "train", "path": f"{base_path}/train.json", "generate_gss": True},
        {"name": "dev", "path": f"{base_path}/dev.json", "generate_gss": True},
        {"name": "test", "path": f"{base_path}/test.json", "generate_gss": False}
    ]
    
    results = {}
    
    for dataset_config in datasets:
        dataset_name = dataset_config["name"]
        dataset_path = dataset_config["path"]
        generate_gss = dataset_config["generate_gss"]
        
        output_dir = f"{output_base}/{dataset_name}"
        
        logger.info(f"开始处理数据集: {dataset_name}")
        
        converter = TableToGraphConverter()
        result = converter.process_dataset_batch(
            dataset_path=dataset_path,
            output_dir=output_dir,
            dataset_name=dataset_name,
            generate_gss=generate_gss
        )
        
        results[dataset_name] = result
        
        logger.info(f"数据集 {dataset_name} 处理完成")
        logger.info(f"  - 样本数: {result['total_samples']}")
        logger.info(f"  - 图节点数: {result['graph_nodes']}")
        logger.info(f"  - 图边数: {result['graph_edges']}")
        if result['gss_generated']:
            gss_stats = result['gss_statistics']
            logger.info(f"  - GSS成功: {gss_stats['successful_gss']}")
            logger.info(f"  - GSS失败: {gss_stats['failed_gss']}")
    
    # 保存总体结果
    os.makedirs(output_base, exist_ok=True)
    summary_path = f"{output_base}/processing_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"所有数据集处理完成，结果保存到: {summary_path}")
    return results


def test_enhanced_gss():
    """测试增强版GSS构建"""
    # 读取数据
    data_path = "/home/cqjtu/NLP-Group/LZH/TTTGR_LLM/dataset/MultiHiertt/dev.json"
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 取第一个样本
    first_sample = data[0]
    logger.info(f"测试增强版GSS构建，样本UID: {first_sample.get('uid', 'unknown')}")
    
    # 转换为主图
    converter = TableToGraphConverter()
    main_graph = converter.convert_sample(first_sample)
    
    # 收集主图节点信息
    main_graph_nodes = {
        'cell_nodes': [],
        'paragraph_nodes': [],
        'qa_nodes': [],
        'doc_nodes': [],
        'table_nodes': [],
        'text_doc_nodes': []
    }
    
    for node_id, node_data in main_graph.nodes(data=True):
        node_type = node_data.get('type', '')
        if node_type == 'Cell':
            main_graph_nodes['cell_nodes'].append((node_id, node_data))
        elif node_type == 'Paragraph':
            main_graph_nodes['paragraph_nodes'].append((node_id, node_data))
        elif node_type == 'QuestionAnswer':
            main_graph_nodes['qa_nodes'].append((node_id, node_data))
        elif node_type == 'Doc':
            main_graph_nodes['doc_nodes'].append((node_id, node_data))
        elif node_type == 'Table':
            main_graph_nodes['table_nodes'].append((node_id, node_data))
        elif node_type == 'TextDoc':
            main_graph_nodes['text_doc_nodes'].append((node_id, node_data))
    
    # 构建增强版GSS
    logger.info("=" * 50)
    logger.info("构建增强版GSS（包含TableDescription节点）")
    logger.info("=" * 50)
    
    enhanced_gss = converter.build_gold_support_subgraph(first_sample, main_graph_nodes)
    
    logger.info(f"增强版GSS构建完成: {enhanced_gss.number_of_nodes()} 个节点, {enhanced_gss.number_of_edges()} 条边")
    
    # 分析增强版GSS节点类型
    gss_node_types = {}
    for node_id, node_data in enhanced_gss.nodes(data=True):
        node_type = node_data.get('type', 'Unknown')
        gss_node_types[node_type] = gss_node_types.get(node_type, 0) + 1
    
    logger.info("增强版GSS节点类型分布:")
    for node_type, count in gss_node_types.items():
        logger.info(f"  {node_type}: {count}")
    
    # 分析GSS边类型
    gss_edge_types = {}
    for u, v, edge_data in enhanced_gss.edges(data=True):
        edge_type = edge_data.get('type', 'Unknown')
        gss_edge_types[edge_type] = gss_edge_types.get(edge_type, 0) + 1
    
    logger.info("增强版GSS边类型分布:")
    for edge_type, count in gss_edge_types.items():
        logger.info(f"  {edge_type}: {count}")
    
    # 详细分析节点和关系
    logger.info("\n增强版GSS详细结构:")
    for node_id, node_data in enhanced_gss.nodes(data=True):
        node_type = node_data.get('type', 'Unknown')
        if node_type == 'QuestionAnswer':
            logger.info(f"QuestionAnswer节点: {node_id}")
            logger.info(f"  问题: {node_data.get('question', '')}")
            logger.info(f"  答案: {node_data.get('answer', '')}")
            logger.info(f"  表格证据: {node_data.get('table_evidence', [])}")
            logger.info(f"  文本证据: {node_data.get('text_evidence', [])}")
        elif node_type == 'Cell':
            coord = f"{node_data.get('table_id', '?')}-{node_data.get('row', '?')}-{node_data.get('col', '?')}"
            logger.info(f"Cell节点: {node_id} (坐标: {coord}, 值: \"{node_data.get('value', '')}\")")
        elif node_type == 'TableDescription':
            desc_key = node_data.get('desc_key', '')
            description = node_data.get('description', '')[:80] + '...' if len(node_data.get('description', '')) > 80 else node_data.get('description', '')
            logger.info(f"TableDescription节点: {node_id} (键: {desc_key})")
            logger.info(f"  描述: {description}")
        elif node_type == 'Paragraph':
            para_content = node_data.get('content', '')[:100] + '...' if len(node_data.get('content', '')) > 100 else node_data.get('content', '')
            logger.info(f"Paragraph节点: {node_id} (索引: {node_data.get('paragraph_index', '?')})")
            logger.info(f"  内容: {para_content}")
        elif node_type == 'Doc':
            logger.info(f"Doc节点: {node_id}")
    
    # 显示关系结构
    logger.info("\n增强版GSS关系结构:")
    for u, v, edge_data in enhanced_gss.edges(data=True):
        edge_type = edge_data.get('type', 'Unknown')
        u_type = enhanced_gss.nodes[u].get('type', 'Unknown')
        v_type = enhanced_gss.nodes[v].get('type', 'Unknown')
        logger.info(f"  {u}({u_type}) --{edge_type}--> {v}({v_type})")
    
    # 导出增强版GSS
    output_dir = "/home/cqjtu/NLP-Group/LZH/TTTGR_LLM/TextTableToGraph/enhanced_gss_output"
    os.makedirs(output_dir, exist_ok=True)
    
    enhanced_gss_cypher_path = os.path.join(output_dir, "enhanced_gss.cypher")
    converter.export_gss_to_neo4j_cypher(enhanced_gss, enhanced_gss_cypher_path)
    
    logger.info(f"\n增强版GSS已导出: {enhanced_gss_cypher_path}")
    
    return enhanced_gss, first_sample

    def combine_gss_graphs(self, gss_graphs):
        """合并多个GSS图"""
        import networkx as nx
        
        if not gss_graphs:
            return nx.MultiDiGraph()
        
        if len(gss_graphs) == 1:
            return gss_graphs[0]
        
        # 创建合并图
        combined_gss = nx.MultiDiGraph()
        
        # 合并所有节点和边
        for gss in gss_graphs:
            # 添加节点
            for node_id, node_data in gss.nodes(data=True):
                if not combined_gss.has_node(node_id):
                    combined_gss.add_node(node_id, **node_data)
            
            # 添加边
            for u, v, edge_data in gss.edges(data=True):
                combined_gss.add_edge(u, v, **edge_data)
        
        return combined_gss