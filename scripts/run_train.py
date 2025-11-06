#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import torch

# Ensure project root is on sys.path so `src` package can be imported when running this script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.train_lightgcn import train_lightgcn


def main(processed_dir: str = 'data', epochs: int = 10, device: str = 'cpu', out_dir: str = 'models'):
    train_path = os.path.join(processed_dir, 'processed_train.csv')
    test_path = os.path.join(processed_dir, 'processed_test.csv')
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise RuntimeError(f'Processed train/test not found in {processed_dir}. Run preprocess_to_indices.py first.')

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    num_users = int(max(train_df['u'].max(), test_df['u'].max()) + 1)
    num_items = int(max(train_df['i'].max(), test_df['i'].max()) + 1)

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'best_model.pth')

    device = device if torch.cuda.is_available() or device == 'cpu' else 'cpu'

    model = train_lightgcn(train_df, test_df, num_users, num_items, epochs=epochs, device=device, save_path=save_path, eval_every=1)
    print('Training finished. Best model (if any) saved to', save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-dir', default='data', help='Directory with processed_train.csv and processed_test.csv')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out-dir', default='models')
    parser.add_argument('--batch-size', type=int, default=1024, help='Training batch size (positive samples per batch)')
    parser.add_argument('--eval-every', type=int, default=1, help='Evaluate every N epochs (set large to skip frequent eval)')
    parser.add_argument('--eval-sample-users', type=int, default=1000, help='Number of users to sample for evaluation')
    args = parser.parse_args()
    main(processed_dir=args.processed_dir, epochs=args.epochs, device=args.device, out_dir=args.out_dir)
    # call train with tunable params
    # Note: main above already calls train; keep CLI args available if you want to call train directly.
