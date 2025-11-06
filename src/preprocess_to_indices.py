
import json
import os
import argparse
import numpy as np
import pandas as pd
from data.lightgcn_data import train_test_split_leave_one_out


def _normalize_val(v):
    """Convert unhashable sequence-like values to tuples, leave scalars as-is."""
    if isinstance(v, (list, tuple, np.ndarray, pd.Index)):
        return tuple(v)
    return v


def build_maps(df: pd.DataFrame, user_col='user_id', item_col='book_id'):
    users = df[user_col].apply(_normalize_val).unique()
    items = df[item_col].apply(_normalize_val).unique()
    # keep original ordering deterministic by sorting string representations
    user_keys = sorted(users, key=lambda x: str(x))
    item_keys = sorted(items, key=lambda x: str(x))
    user_map = {k: int(idx) for idx, k in enumerate(user_keys)}
    item_map = {k: int(idx) for idx, k in enumerate(item_keys)}
    return user_map, item_map


def map_df(df: pd.DataFrame, user_map, item_map, user_col='user_id', item_col='book_id'):
    df = df.copy()
    df['_u_raw'] = df[user_col].apply(_normalize_val)
    df['_i_raw'] = df[item_col].apply(_normalize_val)
    # map with .apply to avoid pandas trying to use Index.get_indexer on complex objects
    df['u'] = df['_u_raw'].apply(lambda x: user_map.get(x, -1)).astype(int)
    df['i'] = df['_i_raw'].apply(lambda x: item_map.get(x, -1)).astype(int)
    # drop rows where mapping failed
    df = df[(df['u'] >= 0) & (df['i'] >= 0)].copy()
    return df[['u', 'i']]


def main(data_dir: str = 'data', ratings_file: str = 'ratings.csv', out_dir: str = 'data'):
    ratings_path = os.path.join(data_dir, ratings_file)
    df = pd.read_csv(ratings_path)

    # Expect columns like user_id and book_id; try common variants
    if 'user_id' not in df.columns or 'book_id' not in df.columns:
        # attempt to detect
        possible_user = [c for c in df.columns if 'user' in c]
        possible_item = [c for c in df.columns if 'book' in c or 'item' in c]
        if possible_user and possible_item:
            user_col = possible_user[0]
            item_col = possible_item[0]
        else:
            raise RuntimeError('Could not find user/book columns in ratings CSV')
    else:
        user_col = 'user_id'
        item_col = 'book_id'

    user_map, item_map = build_maps(df, user_col=user_col, item_col=item_col)
    mapped = map_df(df, user_map, item_map, user_col=user_col, item_col=item_col)

    # leave-one-out split
    train, test = train_test_split_leave_one_out(mapped, seed=42)

    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, 'processed_train.csv')
    test_path = os.path.join(out_dir, 'processed_test.csv')
    user_map_path = os.path.join(out_dir, 'user_map.json')
    item_map_path = os.path.join(out_dir, 'item_map.json')

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    # JSON requires keys to be primitive types; convert map keys to strings
    def _make_serializable_map(m: dict):
        ser = {}
        for k, v in m.items():
            # try to convert numeric-like keys to int if possible
            try:
                if isinstance(k, (int,)):
                    sk = str(int(k))
                else:
                    sk = str(k)
            except Exception:
                sk = str(k)
            ser[sk] = int(v)
        return ser

    user_map_ser = _make_serializable_map(user_map)
    item_map_ser = _make_serializable_map(item_map)

    # Also save reverse maps (index -> original value as string) which is easier to read later
    user_map_rev = {int(v): str(k) for k, v in user_map.items()}
    item_map_rev = {int(v): str(k) for k, v in item_map.items()}

    with open(user_map_path, 'w') as f:
        json.dump({'map': user_map_ser, 'reverse': user_map_rev}, f, indent=2)
    with open(item_map_path, 'w') as f:
        json.dump({'map': item_map_ser, 'reverse': item_map_rev}, f, indent=2)

    print(f'Wrote {train_path}, {test_path}, and maps to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data', help='Directory containing ratings.csv')
    parser.add_argument('--ratings-file', default='ratings.csv', help='Ratings CSV filename')
    parser.add_argument('--out-dir', default='data', help='Output directory for processed files')
    args = parser.parse_args()
    main(data_dir=args.data_dir, ratings_file=args.ratings_file, out_dir=args.out_dir)
