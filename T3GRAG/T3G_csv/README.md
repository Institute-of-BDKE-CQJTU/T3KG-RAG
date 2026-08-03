# T3G_csv

## Output Structure
```
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

## Operation method
```bash
# 1 Automatic Construction of Hybrid-context KG
cd T3GRAG/T3G_csv
python T3G_csv/batch_graph_generation_csv_parallel.py \
  --dataset_root T3GRAG/dataset/MultiHiertt \
  --output_root T3GRAG/T3G_csv/outputs \
  --splits train,dev,test \
  --num_workers 32 \
  --chunk_size 8
```

