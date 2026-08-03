"""
将预生成的 .cypher 文件快速批量导入 Neo4j。

优化点：
- 事务级批量提交（每批一次 commit）
- 不逐条 consume 结果，减少网络往返开销
- 仅显示进度条与总导入耗时
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from pathlib import Path
from typing import List

from tqdm import tqdm

from connect_neo4j import Neo4jConnection


def configure_quiet_runtime() -> None:
    """关闭 Neo4j 驱动与 Python warnings 的噪声输出，只保留进度条和耗时。"""
    warnings.filterwarnings("ignore")

    quiet_loggers = [
        "neo4j",
        "neo4j.notifications",
        "neo4j.io",
        "neo4j.pool",
        "neo4j.bolt",
    ]
    for name in quiet_loggers:
        logger = logging.getLogger(name)
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False


def parse_cypher_statements(file_path: Path) -> List[str]:
    """
    读取并解析 .cypher 文件，按分号拆分语句。

    处理规则：
    - 跳过 // 单行注释
    - 跳过 /* */ 多行注释
    - 支持字符串中的分号（不会错误切分）
    - 去除每条语句首尾空白
    - 跳过空语句
    """
    content = file_path.read_text(encoding="utf-8")

    statements: List[str] = []
    current: List[str] = []

    in_single_quote = False
    in_double_quote = False
    in_line_comment = False
    in_block_comment = False

    i = 0
    length = len(content)

    while i < length:
        ch = content[i]
        nxt = content[i + 1] if i + 1 < length else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        if not in_single_quote and not in_double_quote:
            if ch == "/" and nxt == "/":
                in_line_comment = True
                i += 2
                continue
            if ch == "/" and nxt == "*":
                in_block_comment = True
                i += 2
                continue

        if ch == "'" and not in_double_quote:
            prev = content[i - 1] if i > 0 else ""
            if prev != "\\":
                in_single_quote = not in_single_quote
            current.append(ch)
            i += 1
            continue

        if ch == '"' and not in_single_quote:
            prev = content[i - 1] if i > 0 else ""
            if prev != "\\":
                in_double_quote = not in_double_quote
            current.append(ch)
            i += 1
            continue

        if ch == ";" and not in_single_quote and not in_double_quote:
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
            i += 1
            continue

        current.append(ch)
        i += 1

    tail = "".join(current).strip()
    if tail:
        statements.append(tail)

    return statements


def execute_batch_fast(session, batch: List[str]) -> None:
    """快速模式：一个批次使用单事务提交，不逐条 consume。"""
    tx = session.begin_transaction()
    try:
        for statement in batch:
            tx.run(statement)
        tx.commit()
    except Exception:
        tx.rollback()
        raise


def import_cypher(split: str, batch_size: int) -> None:
    project_root = Path(__file__).resolve().parents[1]
    cypher_file = project_root / "T3G" / "outputs" / split / f"{split}_graph.cypher"

    if not cypher_file.exists():
        raise FileNotFoundError(f"未找到 Cypher 文件：{cypher_file}")

    statements = parse_cypher_statements(cypher_file)
    total = len(statements)

    if total == 0:
        print("无可执行语句，耗时: 0.00s")
        return

    start = time.perf_counter()
    neo4j = Neo4jConnection()

    try:
        with neo4j.driver.session() as session:
            with tqdm(total=total, desc=f"Import {split}", unit="stmt") as pbar:
                for i in range(0, total, batch_size):
                    batch = statements[i : i + batch_size]
                    execute_batch_fast(session, batch)
                    pbar.update(len(batch))
    finally:
        neo4j.close()

    elapsed = time.perf_counter() - start
    print(f"导入耗时: {elapsed:.2f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 .cypher 文件快速批量导入 Neo4j")
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="数据集划分名称（如 train/dev/test），默认 dev",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2000,
        help="事务批次大小，默认 2000（更快）",
    )
    return parser.parse_args()


def main() -> None:
    configure_quiet_runtime()
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size 必须为正整数")

    import_cypher(split=args.split, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
