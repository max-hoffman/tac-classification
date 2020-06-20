from bisect import bisect
from datetime import datetime
import os

import click
import holidays
import numpy as np
import pandas as pd

US_HOLIDAYS = holidays.US()

def is_inebriated(tacs, timestamp, limit = 8e-3):
    idx = bisect(tacs[:,0], timestamp)

    if idx == 0:
        return tacs[0,1] >= limit
    elif idx >= len(tacs):
        return tacs[len(tacs)-1,1] >= limit
    else:
        first_frac = (timestamp - tacs[idx-1,0])/(tacs[idx,0] - tacs[idx-1,0])
        avg = first_frac*tacs[idx-1,1] + (1-first_frac)*tacs[idx,1]
        return avg >= limit

@click.command()
@click.option('--accelerometer-path', required=True)
@click.option('--tac-path', required=True)
@click.option('--output-path', required=True)
@click.option('--pid', required=True)
@click.option('--step-size', required=False, default=400)
@click.option('--sparsity_factor', required=False, default=2)
def main(accelerometer_path, tac_path, output_path, pid, step_size, sparsity_factor):

    acc_df = pd.read_csv(accelerometer_path)

    tacs = pd.read_csv(tac_path).values
    accs = acc_df[acc_df.pid == pid].sort_values('time').values

    ids = []
    features = []
    targets = []
    for i in range(step_size, accs.shape[0], step_size):
        ts = accs[i,0] / 1000
        date = datetime.fromtimestamp(ts)
        is_holiday = date in US_HOLIDAYS
        row = (
            accs[i-step_size:i,2:5],
            np.full((step_size,1), date.year),
            np.full((step_size,1), date.month),
            np.full((step_size,1), date.day),
            np.full((step_size,1), is_holiday),
        )
        seq = np.concatenate(row, axis=1)
        subseq = seq[range(0,step_size,sparsity_factor),:]
        ids.append(int(i/step_size))
        features.append(subseq.reshape(-1))
        targets.append(is_inebriated(tacs, ts))

    output = pd.DataFrame(data={'features': features, 'target': targets}, index=ids)
    output.to_parquet(output_path)

if __name__=="__main__":
    main()
