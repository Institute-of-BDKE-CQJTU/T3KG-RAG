#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行批量图谱生成脚本
可以用于测试（少量样本）或完整生成
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from batch_graph_generation_parallel import ParallelBatchGraphGenerator
import multiprocessing as mp

def main():
    """主函数"""
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    # 创建并行批量生成器（使用8个进程）
    generator = ParallelBatchGraphGenerator(num_workers=8)
    
    # 测试模式：只处理dev集的前10个样本
    # 取消下面的注释来启用测试模式
    # results = generator.process_dataset_parallel("dev", max_samples=10)
    
    # 完整运行：处理所有数据集
    results = generator.run_parallel_batch_generation(max_samples_per_dataset=None)
    
    return results

if __name__ == "__main__":
    results = main()

