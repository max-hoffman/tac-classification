from bisect import bisect
from datetime import datetime
import os

import click
import numpy as np
import pandas as pd
import scipy.signal as signal

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
@click.option('--sparsity_factor', required=False, default=1)
def main(accelerometer_path, tac_path, output_path, pid, step_size, sparsity_factor):

    acc_df = pd.read_csv(accelerometer_path)

    tacs = pd.read_csv(tac_path).values
    accs = acc_df[acc_df.pid == pid].sort_values('time').values
    i_start = 0
    i = 0

    ids = []
    freq_feats = []
    window_feats = []
    step_feats = []
    targets = []

    window = 100
    step = 20
    fmin = 1  # smallest frequency in spectrogram channels
    fmax = 500  # largest frequency in spectrogram channels
    f = np.linspace(fmin, fmax, 32)

    while i_start < len(accs):
        x_pgrams = []
        y_pgrams = []
        z_pgrams = []
        window_feat = []
        dt_row = []
        dt_sum = 0

        while i < len(accs)-2 and dt_sum < 10*1000:
            dt = accs[i+1,0] - accs[i,0]
            dt_row.append(dt)
            dt_sum += dt
            i += 1

        if dt_sum < 10*1000:
            break

        dt_row.append(accs[i+1,0] - accs[i,0])

        n_steps = (i - i_start - window) // step
        for k in range(n_steps):
            t = accs[(i_start+k*step):(i_start+k*step+window),0]
            x = accs[(i_start+k*step):(i_start+k*step+window),2]
            y = accs[(i_start+k*step):(i_start+k*step+window),3]
            z = accs[(i_start+k*step):(i_start+k*step+window),4]

            x_pgram = signal.lombscargle(t, x - np.mean(x), f, normalize=True)
            y_pgram = signal.lombscargle(t, y - np.mean(y), f, normalize=True)
            z_pgram = signal.lombscargle(t, z - np.mean(z), f, normalize=True)

            x_pgrams.append(x_pgram)
            y_pgrams.append(y_pgram)
            z_pgrams.append(z_pgram)
            window_feat.append(
                [np.mean(x), np.mean(y), np.mean(z)] +\
                [np.std(x), np.std(y), np.std(z)] +\
                [np.max(y), np.max(y), np.max(z)] +\
                [np.min(y), np.min(y), np.min(z)]
            )

        ids.append(accs[i_start, 0])
        freq_feats.append([x_pgrams,x_pgrams,x_pgrams])
        window_feats.append(window_feat)
        step_feats.append([len(x_pgrams), max(accs[i_start:i,2]), max(accs[i_start:i,3]), max(accs[i_start:i,4])])

        start_drunk = is_inebriated(tacs, accs[i_start,0] / 1000)
        end_drunk = is_inebriated(tacs, accs[i,0] / 1000)
        targets.append((start_drunk or end_drunk) | 0)

        i_start = i

    output = pd.DataFrame({'freq': freq_feats, 'step': step_feats, 'window': window_feats, 'target': targets}, index=ids)
    output.to_parquet(output_path)

if __name__=="__main__":
    main()
