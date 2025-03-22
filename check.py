#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np
import pandas as pd
import pyedflib
from datetime import datetime, timedelta
from scipy.signal import butter, filtfilt
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Constants
RES_DIR = "res"
DEBUG = True

# New constants for 30-second blocks
SAMPLE_RATE = 512         # Expected sampling rate used during training
BLOCK_DURATION_SEC = 30   # Each block is 30 seconds
BLOCK_SIZE = BLOCK_DURATION_SEC * SAMPLE_RATE  # 30 * 512 = 7680 samples
SUBWINDOW_SIZE = 512      # The model was trained on windows of 512 samples
NUM_SUBWINDOWS = BLOCK_SIZE // SUBWINDOW_SIZE  # 7680 / 512 = 30
SEIZURE_BLOCK_THRESHOLD = 0.7  # If 70% or more subwindows are seizure, mark the block as seizure

def debug_print(message):
    if DEBUG:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[DEBUG {now}] {message}")

# ------------------------------
# Preprocessing Functions
# ------------------------------

def bandpass_filter(data, lowcut=1, highcut=50.0, fs=SAMPLE_RATE, order=5):
    """Apply a Butterworth bandpass filter to the signal."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)

def load_common_channels():
    """
    Load common EEG channels from the saved CSV file.
    If not found, return an empty set.
    """
    common_channels_file = os.path.join(RES_DIR, "common_channels.csv")
    if os.path.exists(common_channels_file):
        df = pd.read_csv(common_channels_file)
        channels = set(df["Channels"].str.lower())
        debug_print(f"Loaded {len(channels)} common channels from {common_channels_file}.")
        return channels
    else:
        debug_print("No common_channels.csv found. Will use channels with 'eeg' in label from EDF.")
        return set()

def load_edf_file(edf_path, common_channels):
    """
    Load an EDF file and return a DataFrame of filtered EEG signals and the sampling frequency.
    Only channels whose lower-case name is in common_channels are retained.
    If common_channels is empty, retain all channels with 'eeg' in their name.
    """
    debug_print(f"Loading EDF file: {edf_path}")
    with pyedflib.EdfReader(edf_path) as f:
        sample_freq = f.getSampleFrequencies()[0]
        signals = {}
        for i, label in enumerate(f.getSignalLabels()):
            label_lc = label.lower()
            if common_channels:
                if label_lc in common_channels:
                    raw_signal = f.readSignal(i).astype(np.float32)
                    filtered = bandpass_filter(raw_signal, fs=sample_freq).astype(np.float32)
                    signals[label_lc] = filtered
            else:
                if "eeg" in label_lc:
                    raw_signal = f.readSignal(i).astype(np.float32)
                    filtered = bandpass_filter(raw_signal, fs=sample_freq).astype(np.float32)
                    signals[label_lc] = filtered
        if not signals:
            debug_print("No matching EEG channels found in EDF file.")
            sys.exit("Exiting: No EEG channels found.")
        df = pd.DataFrame(signals)
        debug_print(f"EDF file loaded with {df.shape[1]} EEG channels and {df.shape[0]} samples.")
        return df, sample_freq

def extract_blocks_for_inference(df, block_size=BLOCK_SIZE, overlap=0):
    """
    Split the dataframe into non-overlapping blocks of size 'block_size' (in samples).
    Returns:
      - blocks: a NumPy array of shape (num_blocks, block_size, num_channels)
      - start_indices: a list of starting sample indices for each block.
    """
    step = block_size - int(overlap * block_size)
    blocks = []
    start_indices = []
    for start in range(0, len(df) - block_size + 1, step):
        block = df.values[start:start + block_size, :]
        blocks.append(block)
        start_indices.append(start)
    debug_print(f"Extracted {len(blocks)} blocks (each {BLOCK_DURATION_SEC} sec) for inference.")
    return np.array(blocks), start_indices

def predict_seizure_for_block(block, model):
    """
    For a given block (30 sec, BLOCK_SIZE samples), split it into subwindows of SUBWINDOW_SIZE,
    predict each subwindow with the model, and return 1 if >= SEIZURE_BLOCK_THRESHOLD fraction of subwindows are seizure.
    """
    # Split the block into subwindows.
    subwindows = np.split(block, NUM_SUBWINDOWS)
    subwindows = np.array(subwindows)  # Shape: (NUM_SUBWINDOWS, SUBWINDOW_SIZE, num_channels)
    preds_prob = model.predict(subwindows)  # Shape: (NUM_SUBWINDOWS, 1)
    preds = (preds_prob > 0.5).astype(int).flatten()  # binary predictions for each subwindow
    fraction_seizure = np.mean(preds)
    debug_print(f"Block prediction: {fraction_seizure*100:.1f}% seizure")
    return 1 if fraction_seizure >= SEIZURE_BLOCK_THRESHOLD else 0

def group_seizure_intervals(preds, start_indices, block_size, sample_freq, gap_tolerance=1.0):
    """
    Group consecutive blocks (that are marked as seizure) into intervals.
    Returns a list of tuples: (start_time_in_sec, end_time_in_sec)
    """
    block_duration = block_size / sample_freq
    intervals = []
    current_interval = None

    for pred, start in zip(preds, start_indices):
        if pred == 1:
            block_start = start / sample_freq
            block_end = block_start + block_duration
            if current_interval is None:
                current_interval = [block_start, block_end]
            else:
                if block_start - current_interval[1] <= gap_tolerance:
                    current_interval[1] = block_end
                else:
                    intervals.append(tuple(current_interval))
                    current_interval = [block_start, block_end]
        else:
            if current_interval is not None:
                intervals.append(tuple(current_interval))
                current_interval = None
    if current_interval is not None:
        intervals.append(tuple(current_interval))
    return intervals

def format_time(seconds):
    """Convert seconds to HH:MM:SS format."""
    return str(timedelta(seconds=int(seconds)))

def print_intervals(model_name, intervals):
    if intervals:
        print(f"\nModel: {model_name} - Detected Seizure Intervals:")
        for interval in intervals:
            start, end = interval
            duration = end - start
            print(f"  From {format_time(start)} ({start:.2f}s) to {format_time(end)} ({end:.2f}s) - Duration: {duration:.2f} sec")
    else:
        print(f"\nModel: {model_name} - No seizure detected.")

def main():
    parser = argparse.ArgumentParser(description="Test EEG Seizure Detection Models on an EDF file")
    parser.add_argument("edf_file", type=str, help="Path to the EDF file to test")
    args = parser.parse_args()

    edf_file = args.edf_file
    if not os.path.exists(edf_file):
        sys.exit(f"EDF file '{edf_file}' does not exist.")

    # Load common channels from training, if available.
    common_channels = load_common_channels()

    # Load the EDF file.
    df, sample_freq = load_edf_file(edf_file, common_channels)
    # Use only EEG channels for inference.

    # For a 30-second block: BLOCK_SIZE samples.
    block_size = BLOCK_SIZE
    debug_print(f"Using block size: {block_size} samples (expected duration: {block_size/sample_freq:.2f} sec)")

    # Extract blocks and record start indices.
    blocks, start_indices = extract_blocks_for_inference(df, block_size=block_size, overlap=0)
    debug_print(f"Each block duration: {block_size/sample_freq:.2f} sec.")

    # Load the three models.
    model_paths = {
        "CNN Model": os.path.join(RES_DIR, "cnn_model.h5"),
        "LSTM Model": os.path.join(RES_DIR, "gru_model.h5"),  # This file was saved from a function using LSTM.
        "CNN+LSTM Model": os.path.join(RES_DIR, "cnn+lstm_model.h5")
    }
    models = {}
    for name, path in model_paths.items():
        if os.path.exists(path):
            models[name] = load_model(path)
            debug_print(f"Loaded {name} from {path}.")
        else:
            print(f"Model file {path} not found. Exiting.")
            sys.exit(1)

    # For each model, process each 30-second block.
    for model_name, model in models.items():
        block_preds = []
        for block in blocks:
            pred = predict_seizure_for_block(block, model)
            block_preds.append(pred)
        block_preds = np.array(block_preds)
        intervals = group_seizure_intervals(block_preds, start_indices, block_size, sample_freq, gap_tolerance=1.0)
        print_intervals(model_name, intervals)

if __name__ == "__main__":
    main()
