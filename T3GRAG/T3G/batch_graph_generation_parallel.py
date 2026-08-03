#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiHiertt数据集批量图结构生成器 - 并行版本
为train.json、dev.json、test.json生成完整的图结构和GSS子图
使用多进程并行处理提高生成速度
输出目录结构与原版batch_graph_generation.py保持一致
"""

import json
import os
import time
from datetime import datetime
import logging
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from table_to_graph_converter import TableToGraphConverter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/cqjtu/NLP-Group/LZH/T3GRAG/T3G/batch_generation_parallel.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_sample_batch(args):
    """处理单个样本批次的工作函数"""
    batch_data, batch_id, dataset_name, output_dir, generate_gss = args
    
    try:
        start_time = time.time()
        
        # 创建转换器实例
        converter = TableToGraphConverter()
        
        # 转换批次数据
        batch_graph = converter.convert_samples(batch_data)
        
        # 导出主图批次文件
        batch_main_file = os.path.join(output_dir, f"{dataset_name}_batch_{batch_id:03d}.cypher")
        converter.export_to_neo4j_cypher(batch_main_file)
        
        batch_result = {
            'batch_id': batch_id,
            'samples': len(batch_data),
            'nodes': batch_graph.number_of_nodes(),
            'edges': batch_graph.number_of_edges(),
            'main_file': batch_main_file,
            'gss_data': [],
            'time': time.time() - start_time,
            'success': True
        }
        
        # 生成GSS数据（如果需要）
        if generate_gss:
            # 为每个样本生成GSS数据
            for i, sample in enumerate(batch_data):
                try:
                    # 为单个样本创建独立的转换器和图
                    single_converter = TableToGraphConverter()
                    single_graph = single_converter.convert_samples([sample])  # 只转换单个样本
                    
                    # 收集单个样本的主图节点信息
                    single_main_graph_nodes = collect_main_graph_nodes(single_graph)
                    
                    # 生成该样本的GSS
                    gss = single_converter.build_gold_support_subgraph(sample, single_main_graph_nodes)
                    if gss.number_of_nodes() > 0:
                        uid = sample.get('uid', f'sample_{i}')
                        
                        # 收集GSS数据而不是导出文件
                        gss_data = {
                            'sample_index': i,
                            'uid': uid,
                            'nodes': [],
                            'edges': []
                        }
                        
                        # 收集节点数据
                        for node_id, node_data in gss.nodes(data=True):
                            gss_data['nodes'].append((node_id, node_data))
                        
                        # 收集边数据
                        for u, v, edge_data in gss.edges(data=True):
                            gss_data['edges'].append((u, v, edge_data))
                        
                        batch_result['gss_data'].append(gss_data)
                        
                except Exception as e:
                    logger.warning(f"批次 {batch_id} 样本 {i} GSS生成失败: {e}")
        
        logger.info(f"批次 {batch_id} 完成: {len(batch_data)} 样本, "
                   f"{batch_graph.number_of_nodes()} 节点, {batch_graph.number_of_edges()} 边, "
                   f"耗时 {batch_result['time']:.2f}s")
        
        return batch_result
        
    except Exception as e:
        logger.error(f"批次 {batch_id} 处理失败: {e}")
        return {
            'batch_id': batch_id,
            'samples': len(batch_data),
            'success': False,
            'error': str(e)
        }

def collect_main_graph_nodes(graph):
    """收集主图节点信息"""
    main_graph_nodes = {
        'cell_nodes': [],
        'paragraph_nodes': [],
        'qa_nodes': [],
        'doc_nodes': [],
        'table_nodes': [],
        'row_nodes': [],
        'column_nodes': [],
        'text_doc_nodes': []
    }
    
    for node_id, node_data in graph.nodes(data=True):
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
        elif node_type == 'Row':
            main_graph_nodes['row_nodes'].append((node_id, node_data))
        elif node_type == 'Column':
            main_graph_nodes['column_nodes'].append((node_id, node_data))
        elif node_type == 'TextDoc':
            main_graph_nodes['text_doc_nodes'].append((node_id, node_data))
            
    return main_graph_nodes

class ParallelBatchGraphGenerator:
    """并行批量图结构生成器"""
    
    def __init__(self, num_workers=8):
        # 使用当前项目下的 MultiHiertt 数据集与 outputs 目录
        self.base_dataset_path = "/home/cqjtu/NLP-Group/LZH/T3GRAG/dataset/MultiHiertt"
        self.output_base_path = "/home/cqjtu/NLP-Group/LZH/T3GRAG/T3G/outputs"
        
        # 自动检测CPU核心数，默认最多使用8个进程
        available_cores = mp.cpu_count()
        self.num_workers = min(num_workers, available_cores - 1, 8)
        self.batch_size = 100  # 每个批次的样本数
        
        logger.info(f"使用 {self.num_workers} 个并行进程（总核心数: {available_cores}）")
        
        # 数据集配置：本次只生成主图结构，不生成任何数据集的 GSS 子图
        self.dataset_configs = {
            "train": {
                "file": "train.json",
                "generate_gss": False,
                "expected_samples": 7830,
                "description": "训练集"
            },
            "dev": {
                "file": "dev.json", 
                "generate_gss": False,
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
    
    def process_dataset_parallel(self, dataset_name, max_samples=None):
        """并行处理数据集"""
        config = self.dataset_configs[dataset_name]
        output_dir = os.path.join(self.output_base_path, dataset_name)
        
        logger.info(f"开始并行处理{config['description']}: {dataset_name}")
        
        # 加载数据
        data = self.load_dataset(dataset_name)
        
        if max_samples:
            data = data[:max_samples]
            
        logger.info(f"数据集 {dataset_name} 包含 {len(data)} 个样本")
        logger.info(f"批次大小: {self.batch_size}, 并行进程: {self.num_workers}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 分批准备任务
        batch_tasks = []
        num_batches = (len(data) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(data))
            batch_data = data[start_idx:end_idx]
            
            task = (batch_data, batch_idx + 1, dataset_name, output_dir, config["generate_gss"])
            batch_tasks.append(task)
        
        logger.info(f"准备了 {len(batch_tasks)} 个批次任务")
        
        # 并行执行
        start_time = time.time()
        results = []
        completed_batches = 0
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交所有任务
            future_to_batch = {executor.submit(process_sample_batch, task): task[1] for task in batch_tasks}
            
            # 收集结果
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        completed_batches += 1
                        
                        # 显示进度
                        progress = (completed_batches / len(batch_tasks)) * 100
                        elapsed = time.time() - start_time
                        speed = completed_batches * self.batch_size / elapsed
                        logger.info(f"进度: {completed_batches}/{len(batch_tasks)} ({progress:.1f}%), "
                                   f"速度: {speed:.1f} 样本/秒, 已用时: {elapsed:.1f}s")
                    
                except Exception as e:
                    logger.error(f"批次 {batch_id} 执行异常: {e}")
        
        total_time = time.time() - start_time
        
        # 合并并生成统一文件
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        result_summary = {
            "dataset_name": dataset_name,
            "total_samples": len(data),
            "successful_batches": len(successful_results),
            "failed_batches": len(failed_results),
            "parallel_time": total_time,
            "average_speed": len(data) / total_time if total_time > 0 else 0
        }
        
        if successful_results:
            logger.info(f"开始合并 {len(successful_results)} 个批次并生成统一文件...")
            
            # 生成统一的主图文件
            main_graph_result = self.generate_unified_main_graph(dataset_name, successful_results, output_dir)
            result_summary.update(main_graph_result)
            
            # 生成统一的GSS文件（如果需要）
            if config["generate_gss"]:
                gss_result = self.generate_unified_gss(dataset_name, successful_results, output_dir)
                result_summary.update(gss_result)
        
        logger.info(f"\n=== 并行处理完成 ===")
        logger.info(f"数据集: {dataset_name}")
        logger.info(f"总耗时: {total_time:.2f}s")
        logger.info(f"成功批次: {len(successful_results)}/{len(batch_tasks)}")
        logger.info(f"失败批次: {len(failed_results)}")
        logger.info(f"总节点数: {result_summary.get('graph_nodes', 0)}")
        logger.info(f"总边数: {result_summary.get('graph_edges', 0)}")
        logger.info(f"平均处理速度: {len(data)/total_time:.1f} 样本/秒")
        
        return result_summary
    
    def generate_unified_main_graph(self, dataset_name, successful_results, output_dir):
        """生成统一的主图文件"""
        # 合并主图文件
        main_graph_cypher_path = os.path.join(output_dir, f"{dataset_name}_graph.cypher")
        main_graph_edgelist_path = os.path.join(output_dir, f"{dataset_name}_graph_edgelist.txt")
        
        logger.info(f"生成统一主图文件: {main_graph_cypher_path}")
        
        total_nodes = 0
        total_edges = 0
        
        # 按批次ID排序
        successful_results.sort(key=lambda x: x['batch_id'])
        
        # 合并Cypher文件
        with open(main_graph_cypher_path, 'w', encoding='utf-8') as outfile:
            # 写入文件头
            outfile.write(f"// Neo4j Cypher Script - {dataset_name.upper()} Dataset\n")
            outfile.write(f"// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            outfile.write(f"// Total batches: {len(successful_results)}\n")
            outfile.write(f"// Total samples: {sum(r['samples'] for r in successful_results)}\n\n")
            
            # 合并所有批次文件内容
            for result in successful_results:
                batch_file = result['main_file']
                if os.path.exists(batch_file):
                    with open(batch_file, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        # 跳过原文件的注释头
                        lines = content.split('\n')
                        content_start = 0
                        for i, line in enumerate(lines):
                            if line.startswith('CREATE ') or line.startswith('MATCH '):
                                content_start = i
                                break
                        
                        if content_start > 0:
                            content = '\n'.join(lines[content_start:])
                        
                        outfile.write(content)
                        if not content.endswith('\n'):
                            outfile.write('\n')
                
                total_nodes += result['nodes']
                total_edges += result['edges']
        
        # 生成边列表文件（简化版本）
        with open(main_graph_edgelist_path, 'w', encoding='utf-8') as f:
            f.write(f"# {dataset_name.upper()} Dataset Edge List\n")
            f.write(f"# Total nodes: {total_nodes}\n")
            f.write(f"# Total edges: {total_edges}\n")
            f.write("# Format: source_node\ttarget_node\tedge_type\n")
        
        # 清理临时批次文件
        for result in successful_results:
            if os.path.exists(result['main_file']):
                os.remove(result['main_file'])
        
        logger.info(f"主图文件生成完成: {total_nodes} 节点, {total_edges} 边")
        
        return {
            "graph_nodes": total_nodes,
            "graph_edges": total_edges,
            "graph_cypher_path": main_graph_cypher_path,
            "graph_edgelist_path": main_graph_edgelist_path
        }
    
    def generate_unified_gss(self, dataset_name, successful_results, output_dir):
        """生成统一的GSS文件"""
        gss_cypher_path = os.path.join(output_dir, f"{dataset_name}_gss_all.cypher")
        gss_stats_path = os.path.join(output_dir, f"{dataset_name}_gss_statistics.json")
        
        logger.info(f"生成统一GSS文件: {gss_cypher_path}")
        
        # 收集所有GSS数据
        all_gss_data = []
        for result in successful_results:
            all_gss_data.extend(result.get('gss_data', []))
        
        # 生成GSS统计信息
        gss_statistics = {
            "total_samples": sum(r['samples'] for r in successful_results),
            "successful_gss": len(all_gss_data),
            "failed_gss": sum(r['samples'] for r in successful_results) - len(all_gss_data),
            "gss_details": []
        }
        
        # 写入GSS Cypher文件
        with open(gss_cypher_path, 'w', encoding='utf-8') as f:
            f.write("// Combined GSS Cypher file\n")
            f.write(f"// Total samples: {gss_statistics['total_samples']}\n")
            f.write(f"// Successful GSS: {gss_statistics['successful_gss']}\n")
            f.write(f"// Failed GSS: {gss_statistics['failed_gss']}\n\n")
            
            # 为每个GSS生成Cypher内容
            for i, gss_data in enumerate(all_gss_data):
                uid = gss_data['uid']
                sample_idx = gss_data['sample_index']
                prefix = f"gss_{i+1}_{uid}"
                
                f.write(f"// GSS for sample {i+1} (uid: {uid})\n")
                
                # 节点id映射（避免跨样本冲突）
                id_map = {}
                
                # 生成节点
                for node_id, node_data in gss_data['nodes']:
                    combined_id = f"{prefix}|{str(node_id)}"
                    id_map[str(node_id)] = combined_id
                    
                    # 格式化节点
                    node_cypher = self.format_gss_node(combined_id, node_data)
                    f.write(node_cypher)
                
                # 生成边
                for u, v, edge_data in gss_data['edges']:
                    edge_type = edge_data.get('type', 'RELATED_TO').upper()
                    s_id = id_map[str(u)]
                    t_id = id_map[str(v)]
                    f.write(f'MATCH (a {{id: {repr(s_id)}}}), (b {{id: {repr(t_id)}}}) CREATE (a)-[:{edge_type}]->(b);\n')
                
                f.write("\n")
                
                # 记录GSS详细信息
                gss_detail = {
                    "sample_index": sample_idx,
                    "uid": uid,
                    "nodes": len(gss_data['nodes']),
                    "edges": len(gss_data['edges']),
                    "status": "success"
                }
                gss_statistics["gss_details"].append(gss_detail)
        
        # 保存GSS统计信息
        with open(gss_stats_path, 'w', encoding='utf-8') as f:
            json.dump(gss_statistics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"GSS文件生成完成: 成功 {gss_statistics['successful_gss']}, 失败 {gss_statistics['failed_gss']}")
        
        return {
            "gss_generated": True,
            "gss_cypher_path": gss_cypher_path,
            "gss_statistics_path": gss_stats_path,
            "gss_statistics": gss_statistics
        }
    
    def format_gss_node(self, node_id, node_data):
        """格式化GSS节点为Cypher格式"""
        node_type = node_data.get('type', 'Unknown')
        props = []
        
        # QA节点使用简化格式
        if node_type == 'QuestionAnswer':
            props.append(f"id: {repr(node_id)}")
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
                props.append(f'program: "{program}"')
            if 'table_evidence' in node_data and node_data['table_evidence']:
                evidence_list = ', '.join([f'"{item}"' for item in node_data['table_evidence']])
                props.append(f'table_evidence: [{evidence_list}]')
            if 'text_evidence' in node_data and node_data['text_evidence']:
                text_evidence_list = ', '.join([f'"{str(item)}"' for item in node_data['text_evidence']])
                props.append(f'text_evidence: [{text_evidence_list}]')
        else:
            # 其他节点类型
            props.append(f"id: {repr(node_id)}")
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
                    props.append(f'{key}: {str(value).lower()}')
                elif isinstance(value, dict):
                    import json
                    dict_str = json.dumps(value)
                    props.append(f'{key}: {dict_str}')
                else:
                    props.append(f'{key}: {value}')
        
        props_str = ', '.join(props)
        return f'CREATE (:{node_type} {{{props_str}}});\n'
    
    def clean_string_for_cypher(self, text):
        """清理字符串中的特殊字符"""
        if not isinstance(text, str):
            return str(text)
        
        import re
        text = re.sub(r'[\u0000-\u001F\u007F-\u009F]', '', text)
        text = text.replace('\\', '\\\\')
        text = text.replace('"', '\\"')
        text = text.replace('\n', '\\n')
        text = text.replace('\r', '\\r')
        text = text.replace('\t', '\\t')
        
        return text
    
    def generate_dataset_summary(self, results):
        """生成数据集处理总结"""
        summary_file = os.path.join(self.output_base_path, "parallel_batch_generation_summary.json")
        
        summary = {
            "generation_info": {
                "timestamp": datetime.now().isoformat(),
                "processing_mode": "parallel",
                "num_workers": self.num_workers,
                "batch_size": self.batch_size,
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
                ),
                "total_parallel_time": sum(r.get("parallel_time", 0) for r in results.values() if r and "error" not in r),
                "average_speed": sum(r.get("average_speed", 0) for r in results.values() if r and "error" not in r) / len([r for r in results.values() if r and "error" not in r and r.get("average_speed")]) if len([r for r in results.values() if r and "error" not in r and r.get("average_speed")]) > 0 else 0
            }
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"生成并行处理总结报告: {summary_file}")
        return summary
    
    def run_parallel_batch_generation(self, max_samples_per_dataset=None):
        """运行并行批量生成"""
        logger.info("=" * 80)
        logger.info("开始MultiHiertt数据集并行批量图结构生成")
        logger.info("=" * 80)
        
        total_start_time = time.time()
        
        # 创建输出目录
        self.create_output_directories()
        
        results = {}
        
        # 处理每个数据集
        for dataset_name in ["train", "dev", "test"]:
            config = self.dataset_configs[dataset_name]
            logger.info(f"\n并行处理{config['description']}: {dataset_name}")
            logger.info("-" * 50)
            
            try:
                result = self.process_dataset_parallel(dataset_name, max_samples_per_dataset)
                results[dataset_name] = result
                
                logger.info(f"{config['description']}并行处理完成")
                
                # 强制垃圾回收
                gc.collect()
                
            except Exception as e:
                logger.error(f"{dataset_name}并行处理失败: {e}")
                results[dataset_name] = {"error": str(e)}
        
        # 生成总结报告
        summary = self.generate_dataset_summary(results)
        
        total_time = time.time() - total_start_time
        logger.info("=" * 80)
        logger.info("并行批量生成完成")
        logger.info(f"总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
        logger.info(f"总样本数: {summary['overall_statistics']['total_samples']}")
        logger.info(f"总图节点数: {summary['overall_statistics']['total_graph_nodes']}")
        logger.info(f"总图边数: {summary['overall_statistics']['total_graph_edges']}")
        logger.info(f"总GSS成功数: {summary['overall_statistics']['total_gss_successful']}")
        logger.info(f"平均处理速度: {summary['overall_statistics']['average_speed']:.1f} 样本/秒")
        
        # 计算加速比（相对于预期的单线程处理）
        if summary['overall_statistics']['average_speed'] > 0:
            expected_single_thread_time = summary['overall_statistics']['total_samples'] / summary['overall_statistics']['average_speed'] * self.num_workers
            speedup = expected_single_thread_time / total_time if total_time > 0 else 1
            logger.info(f"预估加速比: {speedup:.1f}x")
        
        logger.info("=" * 80)
        
        return results


def main():
    """主函数"""
    # 创建并行批量生成器（使用8个进程）
    generator = ParallelBatchGraphGenerator(num_workers=8)
    
    # 运行并行批量生成
    # 可以设置max_samples_per_dataset来限制每个数据集的处理样本数（用于测试）
    results = generator.run_parallel_batch_generation(max_samples_per_dataset=None)  # None表示处理全部
    
    return results


if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    results = main() 