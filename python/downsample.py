from datetime import datetime
import os

import click
import numpy as np
import pandas as pd

@click.command()
@click.option('--input-paths', required=True, nargs=13)
@click.option('--train-path', required=True)
@click.option('--test-path', required=True)
def main(input_paths, train_path, test_path):

    # this is kind of arbitrary but enforces separation of test/train groups
    train_df = pd.concat([pd.read_parquet(f) for f in input_paths[4:11]])
    test_df = pd.concat([pd.read_parquet(f) for f in input_paths[:4]+input_paths[11:]])

    train_df_neg = train_df[~train_df.target]
    train_df_pos = train_df[train_df.target].sample(train_df_neg.shape[0])
    pd.concat([train_df_neg, train_df_pos]).to_parquet(train_path)

    test_df_neg = test_df[~test_df.target]
    test_df_pos = test_df[test_df.target].sample(test_df_neg.shape[0])
    pd.concat([test_df_neg, test_df_pos]).to_parquet(test_path)

if __name__=="__main__":
    main()
