# GRAG 

## Run MultiHiertt dataset



```bash
# 4 Multi-level Semantic Enhanced Evidence Hybrid Retrieval
cd T3GRAG/GRAG
python build_graph_vector_index.py --MultiHiertt
# Output directory
T3GRAG/GRAG/output_vector/MultiHiertt/

cd T3GRAG/GRAG
python graph_retriever.py --MultiHiertt
# Output directory
T3GRAG/GRAG/output_retriever/MultiHiertt
```



```bash
# 5 Hybrid Evidence-enhanced Answer Generation via LLM

# Modify this line of model path
DEFAULT_MODEL_PATH = Path("/LLMs/Llama-3.1-8B-Instruct")

# Run using a pre-trained large language model
cd T3GRAG/GRAG
python generate_answers.py --MultiHiertt
# Output directory
T3GRAG/GRAG/output_answer/MultiHiertt
```



## Run DocRAGLib dataset



```bash
# 4 Multi-level Semantic Enhanced Evidence Hybrid Retrieval
cd T3GRAG/GRAG
python build_graph_vector_index.py --DocRAGLib
# Output directory
T3GRAG/GRAG/output_vector/DocRAGLib/

cd T3GRAG/GRAG
python graph_retriever.py --DocRAGLib
# Output directory
T3GRAG/GRAG/output_retriever/DocRAGLib
```



```bash
# 5 Hybrid Evidence-enhanced Answer Generation via LLM

# Modify this line of model path
DEFAULT_MODEL_PATH = Path("/LLMs/Llama-3.1-8B-Instruct")

# Run using a pre-trained large language model
cd T3GRAG/GRAG
python generate_answers.py --DocRAGLib
# Output directory
T3GRAG/GRAG/output_answer/DocRAGLib
```

