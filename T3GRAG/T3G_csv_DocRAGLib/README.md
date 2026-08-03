# Operation method

```bash
# 1 Automatic Construction of Hybrid-context KG
cd T3GRAG/T3G_csv_DocRAGLib
python T3G_csv_DocRAGLib/batch_graph_generation_csv_parallel_doclibrag.py \
  --input_json T3GRAG/dataset/DocLibRAG_outputs/doclibrag_dev_merged_with_table_description.json \
  --output_dir T3GRAG/T3G_csv_DocLibRAG/outputs/dev_csv_with_desc \
  --num_workers 8 \
  --chunk_size 8
```

