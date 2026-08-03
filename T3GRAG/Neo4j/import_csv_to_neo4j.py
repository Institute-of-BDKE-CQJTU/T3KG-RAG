#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 neo4j-admin database import 将 CSV 批量导入 Neo4j（Docker 版本）。

支持两种数据集快捷参数：
- --MultiHiertt
- --DocRAGLib

行为：
1) 根据参数选择 CSV 目录（也可手工 --csv_dir 覆盖）
2) 停止容器
3) 先清空（删除）目标数据库
4) neo4j-admin full import 重建数据库
5) 重启容器
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import List


DEFAULT_CONTAINER = "neo4j-container"
DEFAULT_DATABASE = "neo4j-grag"
DEFAULT_IMAGE = "neo4j:latest"

PROJECT_ROOT = Path(__file__).parent.parent
MULTIHIERTT_CSV_DIR = PROJECT_ROOT / "T3G_csv" / "outputs" / "dev"
DOCRAGLIB_CSV_DIR = PROJECT_ROOT / "T3G_csv_DocLibRAG" / "outputs" / "dev_csv_with_desc"


def run(cmd: List[str], allow_fail: bool = False) -> None:
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0 and not allow_fail:
        raise RuntimeError(f"命令执行失败: {' '.join(cmd)}")


def build_import_args(import_dir: Path) -> List[str]:
    node_files = sorted(import_dir.glob("nodes_*.csv"))
    rel_files = sorted(import_dir.glob("edges_*.csv"))

    if not node_files:
        raise FileNotFoundError(f"未找到 nodes_*.csv: {import_dir}")
    if not rel_files:
        raise FileNotFoundError(f"未找到 edges_*.csv: {import_dir}")

    args: List[str] = []
    for nf in node_files:
        args.extend(["--nodes", str(nf)])
    for rf in rel_files:
        args.extend(["--relationships", str(rf)])
    return args


def resolve_csv_dir(args: argparse.Namespace) -> Path:
    if args.csv_dir:
        return (PROJECT_ROOT / args.csv_dir).resolve() if not Path(args.csv_dir).is_absolute() else Path(args.csv_dir).resolve()

    if args.MultiHiertt:
        return MULTIHIERTT_CSV_DIR.resolve()
    if args.DocRAGLib:
        return DOCRAGLIB_CSV_DIR.resolve()

    return MULTIHIERTT_CSV_DIR.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="neo4j-admin database import (docker)")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--MultiHiertt",
        action="store_true",
        help=f"导入 MultiHiertt（默认目录: {MULTIHIERTT_CSV_DIR.resolve()}）",
    )
    mode_group.add_argument(
        "--DocRAGLib",
        action="store_true",
        help=f"导入 DocRAGLib（默认目录: {DOCRAGLIB_CSV_DIR.resolve()}）",
    )

    parser.add_argument(
        "--csv_dir",
        type=str,
        default=None,
        help="CSV 输入目录（nodes_*.csv / edges_*.csv）。设置后会覆盖 --MultiHiertt/--DocRAGLib 默认目录",
    )
    parser.add_argument(
        "--container",
        type=str,
        default=DEFAULT_CONTAINER,
        help="Neo4j Docker 容器名称",
    )
    parser.add_argument(
        "--database",
        type=str,
        default=DEFAULT_DATABASE,
        help="目标数据库名称（会先删除再重建）",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=DEFAULT_IMAGE,
        help="Neo4j 镜像（用于 neo4j-admin）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_dir = resolve_csv_dir(args)
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV 目录不存在: {csv_dir}")

    import_args = build_import_args(csv_dir)

    if shutil.which("docker") is None:
        raise EnvironmentError("未找到 docker 命令，请确认 Docker 已安装")

    # 1) Stop container
    run(["docker", "stop", args.container])

    # 2) 先清空旧数据库（防止混入历史数据）
    clear_cmd = [
        "docker",
        "run",
        "--rm",
        "--volumes-from",
        args.container,
        args.image,
        "neo4j-admin",
        "database",
        "delete",
        "--force",
        args.database,
    ]
    run(clear_cmd, allow_fail=True)

    # 3) Run neo4j-admin full import in a throwaway container with volumes
    #    Mount CSV directory as /import in the temporary container
    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--volumes-from",
        args.container,
        "-v",
        f"{csv_dir}:/import",
        args.image,
        "neo4j-admin",
        "database",
        "import",
        "full",
        "--overwrite-destination=true",
        "--verbose",
    ]

    # Replace file paths with /import paths (inside container)
    import_args_in_container: List[str] = []
    for arg in import_args:
        if arg.startswith(str(csv_dir)):
            import_args_in_container.append(arg.replace(str(csv_dir), "/import"))
        else:
            import_args_in_container.append(arg)

    docker_cmd.extend(import_args_in_container)
    docker_cmd.append("--")
    docker_cmd.append(args.database)
    run(docker_cmd)

    # 4) Restart container
    run(["docker", "start", args.container])

    mode = "MultiHiertt" if args.MultiHiertt else ("DocRAGLib" if args.DocRAGLib else "default")
    print("导入完成")
    print("模式:", mode)
    print("CSV目录:", csv_dir)
    print("数据库:", args.database)


if __name__ == "__main__":
    main()
