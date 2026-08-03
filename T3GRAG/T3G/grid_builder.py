#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility to transform an HTML <table> to an explicit integer grid.
Each unique cell gets an incremental integer id (starting at 0).
The grid is a list of rows; each position stores that cell id.

Returns
-------
grid : List[List[int]]
    Fully expanded matrix after applying rowspan/colspan.
cell_meta : Dict[int, Dict[str, Any]]
    Mapping from cell_id to metadata {tag, text, rowspan, colspan, is_th}.
"""
from typing import List, Dict, Any, Tuple
from bs4 import BeautifulSoup


def html_to_grid(html: str) -> Tuple[List[List[int]], Dict[int, Dict[str, Any]]]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return [], {}

    grid: List[List[int]] = []
    cell_meta: Dict[int, Dict[str, Any]] = {}
    cell_id = 0

    # we maintain for each column the remaining rows of rowspan to skip
    rowspan_left: List[int] = []

    for tr in table.find_all("tr"):
        row: List[int] = []
        col = 0
        # adapt rowspan_left length to current grid width
        if len(rowspan_left) < len(grid[0]) if grid else 0:
            rowspan_left.extend([0] * (len(grid[0]) - len(rowspan_left)))

        td_tags = tr.find_all(["td", "th"])
        tag_idx = 0
        # iterate until we've consumed all cells in row
        while tag_idx < len(td_tags):
            # skip occupied cells due to rowspan from previous rows
            while col < len(rowspan_left) and rowspan_left[col] > 0:
                row.append(-1)
                rowspan_left[col] -= 1
                col += 1
            cell = td_tags[tag_idx]
            tag_idx += 1
            rs = int(cell.get("rowspan", 1))
            cs = int(cell.get("colspan", 1))
            text = cell.get_text(strip=True)
            meta = {
                "tag": cell.name,
                "text": text,
                "rowspan": rs,
                "colspan": cs,
                "is_th": cell.name == "th",
            }
            cell_meta[cell_id] = meta
            # ensure rowspan_left long enough
            while len(rowspan_left) < col + cs:
                rowspan_left.append(0)
            for _c in range(cs):
                row.append(cell_id)
                # mark rowspan skip for future rows
                if rs > 1:
                    rowspan_left[col] = rs - 1
                col += 1
            cell_id += 1
        # fill trailing rowspan skips
        while col < len(rowspan_left) and rowspan_left[col] > 0:
            row.append(-1)
            rowspan_left[col] -= 1
            col += 1
        grid.append(row)

    return grid, cell_meta


def _row_texts(grid: List[List[int]], cell_meta: Dict[int, Dict[str, Any]], r: int, num_cols: int) -> List[str]:
    row = grid[r]
    texts: List[str] = []
    for c in range(num_cols):
        cid = row[c] if c < len(row) else -1
        txt = cell_meta[cid]["text"].strip() if cid != -1 else ""
        texts.append(txt)
    return texts


def _row_has_span(grid: List[List[int]], r: int) -> bool:
    """Detect colspan/rowspan effect in expanded grid: repeated cell ids within a row or -1 gaps."""
    row = grid[r]
    if any(cid == -1 for cid in row):
        return True
    seen = set()
    for cid in row:
        if cid in seen:
            return True
        seen.add(cid)
    return False


def analyze_grid(grid: List[List[int]], cell_meta: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Return header_row_count and for each column header layers (list top->bottom)."""
    if not grid:
        return {"header_rows": 0, "col_paths": {}}

    num_rows = len(grid)
    num_cols = max(len(r) for r in grid)

    # Improved header-row detection:
    # - Always include the first row as header if it has any structure.
    # - Prefer detecting true multi-header by looking for (within early rows):
    #   - merged cells (colspan/rowspan -> repeated ids / -1 in expanded grid)
    #   - empty first column with non-empty other columns
    #   - a "unit" row spanning multiple columns (often first col empty and later columns have same cell id)
    # - Stop header when we reach a typical MultiHiertt group-separator row:
    #   - first col has text AND all other cols are empty
    # - For simple tables where each row has values across columns, keep header_rows=1.

    if num_rows == 0:
        header_rows = 0
    else:
        header_rows = 1

    # inspect only first few rows for header patterns
    max_scan = min(num_rows, 8)
    for r in range(1, max_scan):
        texts = _row_texts(grid, cell_meta, r, num_cols)
        if not any(texts):
            # blank row: include in header if we're still scanning header region
            header_rows = r + 1
            continue

        first = texts[0]
        rest_non_empty = sum(1 for t in texts[1:] if t)

        # group separator (year row etc.)
        if first and rest_non_empty == 0:
            break

        # multi-header signals
        multi_signal = False
        if _row_has_span(grid, r):
            multi_signal = True
        if (not first) and rest_non_empty >= 1:
            multi_signal = True

        # also consider that second row like units might be all in one spanning cell
        if rest_non_empty >= 1 and len(set([cid for cid in grid[r] if cid != -1])) <= 2:
            # e.g., ['', (in millions) spanning 3 cols]
            multi_signal = True

        if multi_signal:
            header_rows = r + 1
            continue

        # If we reached a normal row with values across columns, we treat it as data for simple table.
        break

    # collect paths
    col_paths: Dict[int, List[str]] = {c: [] for c in range(num_cols)}
    for c in range(num_cols):
        for r in range(header_rows):
            if len(grid[r]) <= c:
                continue
            cid = grid[r][c]
            if cid == -1:
                # inherit from left non -1 in same row
                left = c - 1
                while left >= 0 and (len(grid[r]) <= left or grid[r][left] == -1):
                    left -= 1
                if left >= 0 and len(grid[r]) > left:
                    cid = grid[r][left]
            if cid != -1:
                text = cell_meta[cid]["text"]
                if text and (not col_paths[c] or col_paths[c][-1] != text):
                    col_paths[c].append(text)

    return {
        "header_rows": header_rows,
        "col_paths": col_paths,
    }
