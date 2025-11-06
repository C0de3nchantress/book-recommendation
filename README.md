# LightGCN Book Recommendation

Minimal repo to train a LightGCN recommender on the supplied BookCrossing/Goodreads-style dataset.

Quick steps

1. Install dependencies:

   pip install -r requirements.txt

2. Preprocess raw ratings into contiguous indices and a leave-one-out split:

   python3 src/preprocess_to_indices.py --data-dir data --ratings-file ratings.csv --out-dir data

3. Train (quick smoke run):

   python3 scripts/run_train.py --processed-dir data --epochs 5 --device cpu --out-dir models

Notes
- `src/preprocess_to_indices.py` creates `data/processed_train.csv`, `data/processed_test.csv`, and `data/user_map.json`/`data/item_map.json`.
- `scripts/run_train.py` will save the best model to `models/best_model.pth` (based on NDCG@10 during training).
