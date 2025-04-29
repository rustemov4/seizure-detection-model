#!/usr/bin/env python3
"""
edf_to_csv_sample.py

Reads an EDF file, takes only 5% of its samples (uniformly spaced),
and writes out a CSV with a 'time' column plus one column per channel.
"""

import argparse
import numpy as np
import pandas as pd
import mne

def edf_to_csv_sample(edf_path, csv_path, fraction=0.05):
    # ------------------------------------------------------------------
    # 1. Load EDF (this will preload all data into memory)
    # ------------------------------------------------------------------
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # data: array, shape (n_channels, n_samples)
    # times: array, shape (n_samples,)
    data, times = raw[:]

    n_channels, n_samples = data.shape
    n_sampled = int(n_samples * fraction)
    if n_sampled < 1:
        raise ValueError(f"Requested fraction {fraction} yields <1 sample (n_samples={n_samples}).")

    # ------------------------------------------------------------------
    # 2. Select sample indices
    # ------------------------------------------------------------------
    # Uniform sampling: pick indices evenly across the recording
    indices = np.linspace(0, n_samples - 1, n_sampled, dtype=int)

    # If you prefer *random* sampling (uncomment below):
    # rng = np.random.default_rng()
    # indices = rng.choice(n_samples, size=n_sampled, replace=False)
    # indices.sort()

    # Subsample
    sub_data = data[:, indices]      # shape: (n_channels, n_sampled)
    sub_times = times[indices]       # shape: (n_sampled,)

    # ------------------------------------------------------------------
    # 3. Build DataFrame and write CSV
    # ------------------------------------------------------------------
    # Make DataFrame: each row = one time-point
    df = pd.DataFrame(sub_data.T, columns=raw.ch_names)
    df.insert(0, 'time', sub_times)

    # Write out
    df.to_csv(csv_path, index=False)
    print(f"Wrote {n_sampled} samples ({fraction*100:.1f}%) to '{csv_path}'")

def main():
    p = argparse.ArgumentParser(description="Convert EDF→CSV with only a fraction of samples")
    p.add_argument("edf_file",  help="Path to input .edf file")
    p.add_argument("csv_file",  help="Path to output .csv file")
    p.add_argument(
        "--fraction", "-f",
        type=float, default=0.05,
        help="Fraction of total samples to keep (default: 0.05 → 5%%)",
    )
    args = p.parse_args()
    edf_to_csv_sample(args.edf_file, args.csv_file, fraction=args.fraction)

if __name__ == "__main__":
    main()
