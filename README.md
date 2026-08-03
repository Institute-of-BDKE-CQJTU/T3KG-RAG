# T3KG-RAG


# 0. Abstract

Large language models (LLMs) have achieved remarkable progress in various natural language understanding and generation tasks. However, question answering (QA) over text-table hybrid contexts remains a challenging task. Existing open-source LLMs and retrieval-augmented generation (RAG) methods still have shortcomings in hybrid context understanding, efficient evidence retrieval and complex reasoning. How to construct a unified representation model for heterogeneous text-table information and achieve efficient hybrid retrieval based on this representation model remains an unsolved issue. In this paper, we propose T3KG-RAG, a novel text-table QA method via RAG over hybrid context knowledge graph (KG). First, our method automatically constructs KG from text-table hybrid contexts, achieving unified representation of heterogeneous data. Next, we propose a multi-level semantic enhanced hybrid retrieval method which enables efficient evidence perception over the hybrid context KG. Finally, we incorporate the hybrid evidence into prompts, and invoke LLM to generate the answer. Experimental results on two public datasets MultiHiertt and DocRAGLib demonstrate that our approach outperforms state-of-the-art baseline models and achieves better computational efficiency.

# 1. Methodology
![image](https://github.com/Institute-of-BDKE-CQJTU/T3KG-RAG/blob/main/Fig1.pdf)
Fig. 1. The overall architecture of the proposed T3KG-RAG framework, which consists of three modules: (1) automated KG construction from text-table hybrid contexts, (2) multi-level semantic enhanced evidence hybrid retrieval, and (3) hybrid evidence enhanced answer generation via LLM.
# 2. Dataset and Model Downloads

Download links for the MultiHiertt dataset and the DocRAGLib dataset:[MultiHiertt](https://github.com/psunlpgroup/MultiHiertt) and [DocRAGLib](https://github.com/ChiZhang-bit/Mixture-of-RAG).

Download link for the encoder in ‘Multi-level Semantic Enhanced Evidence Hybrid Retrieval’:[all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)

Download links for the Llama 3.1-8B large language model:[Llama 3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

Please place the MultHiertt dataset in the following directory: T3GRAG/dataset/MultiHiertt;

Please place the DocRAGLib dataset in the following directory: T3GRAG/dataset/DocRAGLib.



# 3. **Main Files Structures**

```python
# Project Structure
T3GRAG/
├── GRAG/                       # Main Workflow for Knowledge Graph retrieval-augmented generation
│
├── T3G/                        # Algorithms for Constructing Basic Knowledge Graph Structures

├── T3G_csv/                    # MultiHiertt Dataset Knowledge Graph Construction Process
│
├── T3G_csv_DocRAGLib/          # DocRAGLib Dataset Knowledge Graph Construction Process
│
├── Neo4j/                      # Neo4j Import, Connection and Deployment Tools
├── dataset/                    # Dataset
│   ├── MultiHiertt/            # MultiHiertt Raw Dataset
│   └── DocRAGLib/              # DocRAGLib Raw Dataset
├── dataset_process_DocRAGLib/  # Pre-processing the DocRAGLib dataset
├── model/                      # Place the model in this pat
│   ├── all-mpnet-base-v2/      # Put the all-mpnet-base-v2 model in this path and change the code path
│   └── Llama-3.1-8B-Instruct/  # Put the Llama-3.1-8B-Instruct model in this path and change the code path

```



# 4. Run the code



## 4.1 **Environment Setup**



```python
conda create -n t3kg_rag python=3.9
conda activate t3kg_rag
pip install -r requirements.txt
```

## 4.2 Run the MultiHiertt dataset



```python
# 1 Automatic Construction of Hybrid-context KG
cd T3GRAG/T3G_csv
python T3G_csv/batch_graph_generation_csv_parallel.py \
  --dataset_root T3GRAG/dataset/MultiHiertt \
  --output_root T3GRAG/T3G_csv/outputs \
  --splits train,dev,test \
  --num_workers 32 \
  --chunk_size 8

# Output directory
T3G_csv/outputs/
  train/
    nodes_{Label}.csv
    edges_{TYPE}.csv
  dev/
    nodes_{Label}.csv
    edges_{TYPE}.csv
  test/
    nodes_{Label}.csv
    edges_{TYPE}.csv
```



```python
# 2 Neo4j connection
T3GRAG/Neo4j/
├── docker-compose.yml      # Docker Compose Configuration file
├── connect_neo4j.py        # Neo4j connection code
├── requirements.txt        
└── README.md              
# 
cd T3GRAG/Neo4j
docker compose up -d
```

Open Neo4j via the link:http://your-server-ip:7474

```python
# 3 importing the MultiHiertt dataset into Neo4j
cd T3GRAG
python Neo4j/import_csv_to_neo4j.py --MultiHiertt --database neo4j
```



```python
# 4 Multi-level Semantic Enhanced Evidence Hybrid Retrieval


# Modify this line of model path
Line 34 MODEL_PATH = "sentence-transformers/all-mpnet-base-v2"

cd T3GRAG/GRAG
python build_graph_vector_index.py --MultiHiertt
# Output directory
T3GRAG/GRAG/output_vector/MultiHiertt/

cd T3GRAG/GRAG
python graph_retriever.py --MultiHiertt
# Output directory
T3GRAG/GRAG/output_retriever/MultiHiertt
```



```python
# 5 Hybrid Evidence-enhanced Answer Generation via LLM

# Modify this line of model path
DEFAULT_MODEL_PATH = Path("/LLMs/Llama-3.1-8B-Instruct")

# Run using a pre-trained large language model
cd T3GRAG/GRAG
python generate_answers.py --MultiHiertt
# Output directory
T3GRAG/GRAG/output_answer/MultiHiertt

```



## 4.3 Run the DocRAGLib dataset



```python
# 0 Pre-processing the DocRAGLib dataset

cd T3GRAG/dataset/DocRAGLib_outputs
python3 generate_table_description.py

cd T3GRAG/dataset_process_DocRAGLib
python merge_docraglib_dev.py

# 1 Automatic Construction of Hybrid-context KG
cd T3GRAG/T3G_csv_DocRAGLib
python T3G_csv_DocRAGLib/batch_graph_generation_csv_parallel_doclibrag.py \
  --input_json T3GRAG/dataset/DocLibRAG_outputs/doclibrag_dev_merged_with_table_description.json \
  --output_dir T3GRAG/T3G_csv_DocLibRAG/outputs/dev_csv_with_desc \
  --num_workers 8 \
  --chunk_size 8
```



```python
# 2 Neo4j connection
T3GRAG/Neo4j/
├── docker-compose.yml      # Docker Compose Configuration file
├── connect_neo4j.py        # Neo4j connection code
├── requirements.txt        
└── README.md              
# 
cd T3GRAG/Neo4j
docker compose up -d
```

Open Neo4j via the link:http://your-server-ip:7474

```python
# 3 importing the DocRAGLib dataset into Neo4j
cd T3GRAG
python Neo4j/import_csv_to_neo4j.py --DocRAGLib --database neo4j
```



```python
# 4 Multi-level Semantic Enhanced Evidence Hybrid Retrieval

# Modify this line of model path
Line 34 MODEL_PATH = "sentence-transformers/all-mpnet-base-v2"

cd T3GRAG/GRAG
python build_graph_vector_index.py --DocRAGLib
# Output directory
T3GRAG/GRAG/output_vector/DocRAGLib/

cd T3GRAG/GRAG
python graph_retriever.py --DocRAGLib
# Output directory
T3GRAG/GRAG/output_retriever/DocRAGLib
```



```python
# 5 Hybrid Evidence-enhanced Answer Generation via LLM

# Modify this line of model path
DEFAULT_MODEL_PATH = Path("/LLMs/Llama-3.1-8B-Instruct")

# Run using a pre-trained large language model
cd T3GRAG/GRAG
python generate_answers.py --DocRAGLib
# Output directory
T3GRAG/GRAG/output_answer/DocRAGLib

```











