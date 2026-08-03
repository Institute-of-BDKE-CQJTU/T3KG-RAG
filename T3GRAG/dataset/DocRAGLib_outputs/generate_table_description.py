import argparse
import json
from pathlib import Path

from bs4 import BeautifulSoup

EMPTY = "[EMPTY]"


def _clean_text(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").split())


def _normalize_cell_text(cell) -> str:
    text = cell.get_text(" ", strip=True)
    if not text:
        return EMPTY
    return _clean_text(text)


def parse_table_to_matrix(html_string: str):
    """
    Parse HTML table into a complete matrix with rowspan/colspan expanded.
    Returns:
      matrix: List[List[str]]
      header_rows: List[int] (top header row indices)
      top_header_nonexist_flag: int
    """
    soup = BeautifulSoup(html_string, features="html.parser")

    for sup in soup.select("sup"):
        sup.extract()
    for sub in soup.select("sub"):
        sub.extract()

    table = soup.find("table")
    if table is None:
        return [], [], 1

    trs = table.find_all("tr")
    matrix = []
    pending_rowspan = {}  # col_idx -> {"rows_left": int, "value": str}

    for tr in trs:
        row = []
        col = 0

        cells = tr.find_all(["td", "th"])
        cell_idx = 0

        while cell_idx < len(cells) or col in pending_rowspan:
            while col in pending_rowspan and pending_rowspan[col]["rows_left"] > 0:
                row.append(pending_rowspan[col]["value"])
                pending_rowspan[col]["rows_left"] -= 1
                if pending_rowspan[col]["rows_left"] == 0:
                    del pending_rowspan[col]
                col += 1

            if cell_idx >= len(cells):
                continue

            cell = cells[cell_idx]
            cell_idx += 1

            text = _normalize_cell_text(cell)
            rowspan = int(cell.get("rowspan", 1) or 1)
            colspan = int(cell.get("colspan", 1) or 1)

            for i in range(colspan):
                row.append(text)
                if rowspan > 1:
                    pending_rowspan[col + i] = {"rows_left": rowspan - 1, "value": text}
            col += colspan

        matrix.append(row)

    if not matrix:
        return [], [], 1

    max_cols = max(len(r) for r in matrix)
    for i in range(len(matrix)):
        if len(matrix[i]) < max_cols:
            matrix[i].extend([EMPTY] * (max_cols - len(matrix[i])))

    header_rows = []
    for r_idx, row in enumerate(matrix):
        first_col = row[0] if row else EMPTY
        others = row[1:] if len(row) > 1 else []
        has_other_content = any(v != EMPTY for v in others)

        # Top header heuristic: leading index col is empty and header cols have content.
        if (first_col == EMPTY or first_col == "") and has_other_content:
            header_rows.append(r_idx)
        else:
            break

    if not header_rows:
        top_header_nonexist_flag = 1
    else:
        top_header_nonexist_flag = 0

    return matrix, header_rows, top_header_nonexist_flag


def _resolve_unnamed_label(labels, j: int) -> str:
    """Find nearest non-empty/non-unnamed label for column j."""
    def is_unnamed(v: str) -> bool:
        sv = str(v)
        return (sv == EMPTY) or sv.startswith("Unnamed")

    tmp = j
    while tmp < len(labels) and is_unnamed(labels[tmp]):
        tmp += 1
    if tmp < len(labels):
        return str(labels[tmp])

    tmp = j
    while tmp >= 0 and is_unnamed(labels[tmp]):
        tmp -= 1
    if tmp >= 0:
        return str(labels[tmp])

    return f"data {j}"


def get_table_context(paragraphs, table_idx: int, window: int = 2) -> str:
    marker = f"## Table {table_idx} ##"
    pos = -1
    for i, p in enumerate(paragraphs):
        if marker in p:
            pos = i
            break

    if pos == -1:
        return ""

    left = max(0, pos - window)
    right = min(len(paragraphs), pos + window + 1)
    context_parts = []
    for i in range(left, right):
        if i == pos:
            continue
        text = _clean_text(paragraphs[i])
        if text:
            context_parts.append(text)
    return " ".join(context_parts)


def generate_cell_descriptions(matrix, header_rows, top_header_nonexist_flag: int, table_idx: int):
    describe_dict = {}
    if not matrix or len(matrix[0]) <= 1:
        return describe_dict

    header_count = len(header_rows) if header_rows else 1
    data_start = len(header_rows)

    # Build header levels for data columns (excluding first index column)
    col_header_levels = []
    if not top_header_nonexist_flag:
        for h in header_rows:
            col_header_levels.append(matrix[h][1:])

    row_labels = []
    values_matrix = []
    for r in range(data_start, len(matrix)):
        row = matrix[r]
        row_labels.append(row[0] if row else EMPTY)
        values_matrix.append(row[1:])

    if top_header_nonexist_flag:
        synthetic = [f"data{j}" for j in range(len(values_matrix[0]) if values_matrix else 0)]
        col_header_levels = [synthetic]

    for i in range(len(values_matrix)):
        row_vals = values_matrix[i]
        for j in range(len(row_vals)):
            value_str = str(row_vals[j])
            if (
                value_str.startswith("Unnamed")
                or value_str == EMPTY
                or value_str == "-"
                or value_str == "—"
                or value_str.strip() == ""
            ):
                continue

            base_row = row_labels[i]
            row_semantic = "total" if base_row in (EMPTY, "") else str(base_row)

            # Hierarchical row context: previous row with empty data cells acts as group label
            temp_i = i - 1
            while temp_i >= 0:
                if all(str(v) == EMPTY for v in values_matrix[temp_i]):
                    group_label = row_labels[temp_i]
                    if group_label not in (EMPTY, ""):
                        row_semantic += f" {group_label}"
                    break
                temp_i -= 1

            col_semantic = ""
            if col_header_levels:
                level0 = col_header_levels[0]
                col_semantic = _resolve_unnamed_label(level0, j)

                prev = col_semantic
                for lv in col_header_levels[1:]:
                    cur = str(lv[j]) if j < len(lv) else EMPTY
                    if cur.startswith("Unnamed") or cur == EMPTY or cur == prev:
                        continue
                    col_semantic += f" {cur}"
                    prev = cur

            if col_semantic:
                sentence = f"Table {table_idx} shows {row_semantic} of {col_semantic} is {value_str} ."
            else:
                sentence = f"Table {table_idx} shows {row_semantic} is {value_str} ."

            x_index = i + header_count
            if top_header_nonexist_flag == 1:
                x_index -= 1
            y_index = j + 1
            describe_dict[f"{table_idx}-{x_index}-{y_index}"] = _clean_text(sentence)

    return describe_dict


def build_table_description(sample: dict):
    paragraphs = sample.get("paragraphs", [])
    tables = sample.get("tables", [])

    all_cell_descriptions = {}
    for table_idx, table_html in enumerate(tables):
        try:
            matrix, header_rows, top_header_nonexist_flag = parse_table_to_matrix(table_html)
            _ = get_table_context(paragraphs, table_idx, window=2)
            cell_descriptions = generate_cell_descriptions(
                matrix=matrix,
                header_rows=header_rows,
                top_header_nonexist_flag=top_header_nonexist_flag,
                table_idx=table_idx,
            )
            all_cell_descriptions.update(cell_descriptions)
        except Exception:
            continue

    return all_cell_descriptions


def main():
    parser = argparse.ArgumentParser(description="Generate table_description for DocRAGLib merged dataset.")
    base_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--input",
        type=str,
        default=str(base_dir / "docraglib_dev_merged.json"),
        help="Input json file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(base_dir / "docraglib_dev_merged_with_table_description.json"),
        help="Output json file",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    for sample in dataset:
        sample["table_description"] = build_table_description(sample)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Done. Wrote {len(dataset)} samples to: {output_path}")


if __name__ == "__main__":
    main()
