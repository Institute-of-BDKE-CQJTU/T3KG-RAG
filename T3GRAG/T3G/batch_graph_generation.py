#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiHiertt数据集批量图结构生成器
为train.json、dev.json、test.json生成完整的图结构和GSS子图
"""

import json
import os
import time
from datetime import datetime
import logging
from pathlib import Path
from table_to_graph_converter import TableToGraphConverter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = Path(__file__).resolve().with_name('batch_generation.log')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BatchGraphGenerator:
    """批量图结构生成器"""
    
    def __init__(self):
        self.base_dataset_path = str(PROJECT_ROOT / 'dataset' / 'MultiHiertt')
        self.output_base_path = str(PROJECT_ROOT / 'T3G' / 'outputs')
        
        # 数据集配置
        self.dataset_configs = {
            "train": {
                "file": "train.json",
                "generate_gss": True,
                "expected_samples": 7830,
                "description": "训练集"
            },
            "dev": {
                "file": "dev.json", 
                "generate_gss": True,
                "expected_samples": 1044,
                "description": "验证集"
            },
            "test": {
                "file": "test.json",
                "generate_gss": False,  # test集没有答案，不生成GSS
                "expected_samples": 1566,
                "description": "测试集"
            }
        }
        
    def create_output_directories(self):
        """创建输出目录结构"""
        directories = [
            self.output_base_path,
            f"{self.output_base_path}/train",
            f"{self.output_base_path}/dev", 
            f"{self.output_base_path}/test",
            f"{self.output_base_path}/statistics",
            f"{self.output_base_path}/logs"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"创建目录: {directory}")
    
    def load_dataset(self, dataset_name):
        """加载数据集"""
        config = self.dataset_configs[dataset_name]
        dataset_path = os.path.join(self.base_dataset_path, config["file"])
        
        logger.info(f"加载{config['description']}: {dataset_path}")
        start_time = time.time()
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            load_time = time.time() - start_time
            actual_samples = len(data)
            expected_samples = config["expected_samples"]
            
            logger.info(f"加载完成: {actual_samples} 个样本 (预期: {expected_samples})")
            logger.info(f"加载耗时: {load_time:.2f} 秒")
            
            if actual_samples != expected_samples:
                logger.warning(f"样本数量不匹配: 实际 {actual_samples} vs 预期 {expected_samples}")
            
            return data
        
        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            raise
    
    def generate_main_graph(self, dataset_name, data, max_samples=None):
        """生成主图结构"""
        config = self.dataset_configs[dataset_name]
        output_dir = os.path.join(self.output_base_path, dataset_name)
        
        if max_samples:
            data = data[:max_samples]
            logger.info(f"限制处理样本数: {max_samples}")
        
        logger.info(f"开始生成{config['description']}主图结构: {len(data)} 个样本")
        start_time = time.time()
        
        # 使用TableToGraphConverter进行批量处理
        converter = TableToGraphConverter()
        result = converter.process_dataset_batch(
            dataset_path=os.path.join(self.base_dataset_path, config["file"]),
            output_dir=output_dir,
            dataset_name=dataset_name,
            generate_gss=config["generate_gss"],
            max_samples=max_samples
        )
        
        generation_time = time.time() - start_time
        logger.info(f"{config['description']}主图生成完成")
        logger.info(f"生成耗时: {generation_time:.2f} 秒")
        logger.info(f"图统计: {result['graph_nodes']} 个节点, {result['graph_edges']} 条边")
        
        return result
    
    def generate_individual_gss(self, dataset_name, data, main_graph_result, max_samples=None):
        """为每个样本生成独立的GSS子图文件"""
        if not self.dataset_configs[dataset_name]["generate_gss"]:
            logger.info(f"{dataset_name} 不生成GSS子图")
            return None
        
        if max_samples:
            data = data[:max_samples]
        
        logger.info(f"开始为{dataset_name}生成独立GSS子图文件")
        start_time = time.time()
        
        output_dir = os.path.join(self.output_base_path, dataset_name, "individual_gss")
        os.makedirs(output_dir, exist_ok=True)
        
        gss_statistics = {
            "total_samples": len(data),
            "successful_gss": 0,
            "failed_gss": 0,
            "individual_files": []
        }
        
        converter = TableToGraphConverter()
        
        for i, sample in enumerate(data):
            if i % 100 == 0:
                logger.info(f"GSS生成进度: {i+1}/{len(data)}")
            
            try:
                # 为单个样本转换图
                temp_converter = TableToGraphConverter()
                temp_graph = temp_converter.convert_sample(sample)
                
                # 收集主图节点信息
                main_graph_nodes = {
                    'cell_nodes': [],
                    'paragraph_nodes': [],
                    'qa_nodes': [],
                    'doc_nodes': [],
                    'table_nodes': [],
                    'row_nodes': [],
                    'column_nodes': []
                }
                
                for node_id, node_data in temp_graph.nodes(data=True):
                    node_type = node_data.get('type', '')
                    if node_type == 'Cell':
                        main_graph_nodes['cell_nodes'].append((node_id, node_data))
                    elif node_type == 'Paragraph':
                        main_graph_nodes['paragraph_nodes'].append((node_id, node_data))
                    elif node_type == 'QA':
                        main_graph_nodes['qa_nodes'].append((node_id, node_data))
                    elif node_type == 'Doc':
                        main_graph_nodes['doc_nodes'].append((node_id, node_data))
                    elif node_type == 'Table':
                        main_graph_nodes['table_nodes'].append((node_id, node_data))
                    elif node_type == 'Row':
                        main_graph_nodes['row_nodes'].append((node_id, node_data))
                    elif node_type == 'Column':
                        main_graph_nodes['column_nodes'].append((node_id, node_data))
                
                # 构建GSS
                gss = temp_converter.build_gold_support_subgraph(sample, main_graph_nodes)
                
                if gss.number_of_nodes() > 0:
                    # 导出独立GSS文件
                    uid = sample.get('uid', f'sample_{i}')
                    gss_file = os.path.join(output_dir, f"gss_{i+1}_{uid}.cypher")
                    temp_converter.export_gss_to_neo4j_cypher(gss, gss_file)
                    
                    gss_statistics["successful_gss"] += 1
                    gss_statistics["individual_files"].append({
                        "index": i,
                        "uid": uid,
                        "file": gss_file,
                        "nodes": gss.number_of_nodes(),
                        "edges": gss.number_of_edges()
                    })
                    
                else:
                    logger.warning(f"样本 {i} GSS为空")
                    gss_statistics["failed_gss"] += 1
                    
            except Exception as e:
                logger.error(f"样本 {i} GSS生成失败: {e}")
                gss_statistics["failed_gss"] += 1
        
        generation_time = time.time() - start_time
        logger.info(f"独立GSS生成完成: 成功 {gss_statistics['successful_gss']}, 失败 {gss_statistics['failed_gss']}")
        logger.info(f"GSS生成耗时: {generation_time:.2f} 秒")
        
        # 保存统计信息
        stats_file = os.path.join(output_dir, "gss_statistics.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(gss_statistics, f, indent=2, ensure_ascii=False)
        
        return gss_statistics
    
    def generate_combined_graph_with_edges(self, dataset_name, main_result):
        """生成带有边关系连接的完整图结构文件"""
        logger.info(f"为{dataset_name}生成完整图结构+边关系文件")
        
        output_dir = os.path.join(self.output_base_path, dataset_name)
        
        # 读取主图Cypher文件
        main_cypher_path = main_result["graph_cypher_path"]
        combined_file_path = os.path.join(output_dir, f"{dataset_name}_complete_graph_with_edges.cypher")
        
        try:
            with open(main_cypher_path, 'r', encoding='utf-8') as f:
                cypher_content = f.read()
            
            # 在文件开头添加统计信息和说明
            header = f"""// ================================================
// MultiHiertt {dataset_name.upper()} 数据集完整图结构 + 边关系
// ================================================
// 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
// 数据集: {dataset_name}
// 样本数量: {main_result['total_samples']}
// 图节点数: {main_result['graph_nodes']}
// 图边数: {main_result['graph_edges']}
// ================================================

// 说明:
// 1. 本文件包含完整的异构图结构
// 2. 节点类型: Doc, Table, Row, Column, Cell, TableDescription, TextDoc, Paragraph, QA
// 3. 边关系类型: HAS_TABLE, HAS_ROW, HAS_COLUMN, HAS_CELL, HAS_DESCRIPTION, 
//               HAS_TABLE_DESCRIPTION, HAS_TEXT, HAS_PARAGRAPH, HAS_QA
// 4. 每个节点后面紧跟其相关的边关系连接
// ================================================

"""
            
            # 组合完整内容
            complete_content = header + cypher_content
            
            # 写入完整文件
            with open(combined_file_path, 'w', encoding='utf-8') as f:
                f.write(complete_content)
            
            logger.info(f"完整图结构文件已生成: {combined_file_path}")
            return combined_file_path
            
        except Exception as e:
            logger.error(f"生成完整图结构文件失败: {e}")
            return None
    
    def generate_dataset_summary(self, results):
        """生成数据集处理总结"""
        summary_file = os.path.join(self.output_base_path, "batch_generation_summary.json")
        
        summary = {
            "generation_info": {
                "timestamp": datetime.now().isoformat(),
                "total_datasets": len(results),
                "datasets_processed": list(results.keys())
            },
            "datasets": results,
            "overall_statistics": {
                "total_samples": sum(r.get("total_samples", 0) for r in results.values() if r and "error" not in r),
                "total_graph_nodes": sum(r.get("graph_nodes", 0) for r in results.values() if r and "error" not in r),
                "total_graph_edges": sum(r.get("graph_edges", 0) for r in results.values() if r and "error" not in r),
                "total_gss_successful": sum(
                    r.get("gss_statistics", {}).get("successful_gss", 0) if r.get("gss_statistics") else 0 
                    for r in results.values() if r and "error" not in r
                ),
                "total_gss_failed": sum(
                    r.get("gss_statistics", {}).get("failed_gss", 0) if r.get("gss_statistics") else 0 
                    for r in results.values() if r and "error" not in r
                )
            }
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"生成总结报告: {summary_file}")
        return summary
    
    def run_batch_generation(self, max_samples_per_dataset=None):
        """运行批量生成"""
        logger.info("=" * 80)
        logger.info("开始MultiHiertt数据集批量图结构生成")
        logger.info("=" * 80)
        
        total_start_time = time.time()
        
        # 创建输出目录
        self.create_output_directories()
        
        results = {}
        
        for dataset_name in ["train", "dev", "test"]:
            config = self.dataset_configs[dataset_name]
            logger.info(f"\n处理{config['description']}: {dataset_name}")
            logger.info("-" * 50)
            
            try:
                # 1. 加载数据集
                data = self.load_dataset(dataset_name)
                
                # 2. 生成主图结构
                main_result = self.generate_main_graph(dataset_name, data, max_samples_per_dataset)
                
                # 3. 生成完整图结构+边关系文件
                complete_graph_file = self.generate_combined_graph_with_edges(dataset_name, main_result)
                main_result["complete_graph_file"] = complete_graph_file
                
                # 4. 生成独立GSS文件 (如果需要)
                if config["generate_gss"]:
                    gss_stats = self.generate_individual_gss(dataset_name, data, main_result, max_samples_per_dataset)
                    main_result["individual_gss_statistics"] = gss_stats
                
                results[dataset_name] = main_result
                
                logger.info(f"{config['description']}处理完成")
                
            except Exception as e:
                logger.error(f"{dataset_name}处理失败: {e}")
                results[dataset_name] = {"error": str(e)}
        
        # 5. 生成总结报告
        summary = self.generate_dataset_summary(results)
        
        total_time = time.time() - total_start_time
        logger.info("=" * 80)
        logger.info("批量生成完成")
        logger.info(f"总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
        logger.info(f"总样本数: {summary['overall_statistics']['total_samples']}")
        logger.info(f"总图节点数: {summary['overall_statistics']['total_graph_nodes']}")
        logger.info(f"总图边数: {summary['overall_statistics']['total_graph_edges']}")
        logger.info(f"总GSS成功数: {summary['overall_statistics']['total_gss_successful']}")
        logger.info("=" * 80)
        
        return results


def main():
    """主函数"""
    generator = BatchGraphGenerator()
    
    # 运行批量生成
    # 可以设置max_samples_per_dataset来限制每个数据集的处理样本数（用于测试）
    results = generator.run_batch_generation(max_samples_per_dataset=None)  # None表示处理全部
    
    return results


if __name__ == "__main__":
    results = main() 