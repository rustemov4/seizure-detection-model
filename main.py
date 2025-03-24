import os
import glob

import matplotlib
import numpy as np
import pandas as pd
import pyedflib
from datetime import datetime
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             mean_squared_error, mean_absolute_error, r2_score, confusion_matrix)
from sklearn.decomposition import FastICA  # For ICA
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Flatten, Dense, Dropout,
                                     BatchNormalization, LSTM, Input)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Use TkAgg backend for compatibility
matplotlib.use('TkAgg')

# Constants and results directory
RES_DIR = "res"
PLOTS_DIR = "plots"
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
STD_MIN = 1e-5
STD_MAX = 1e5
CORRELATION_THRESHOLD = 0.88
WINDOW_SIZE = 512  # Window length in samples
OVERLAP = 0  # Overlap between windows (in samples)
SEIZURE_THRESHOLD = 0.6  # Window label = 1 if mean seizure value > threshold
DEBUG = True


def debug_print(message):
    if DEBUG:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[DEBUG {now}] {message}")


# ------------------------------
# Custom Attention Layer
# ------------------------------
class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch, time_steps, features)
        self.W = self.add_weight(name="att_weight",
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer="random_normal",
                                 trainable=True)
        self.b = self.add_weight(name="att_bias",
                                 shape=(input_shape[-1],),
                                 initializer="zeros",
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        # Calculate attention scores
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        # Multiply attention weights with the inputs to get context vector
        output = tf.reduce_sum(x * a, axis=1)
        return output


# ------------------------------
# Signal Preprocessing Functions
# ------------------------------

def bandpass_filter(data, lowcut=1, highcut=50.0, fs=512, order=5):
    """Apply a Butterworth bandpass filter to the signal."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def apply_ica_to_df(df, n_components=None):
    """
    Applies Independent Component Analysis (ICA) to the EEG channels in the DataFrame.
    The EEG columns (all except "Seizure") are processed and then reconstructed.
    """
    debug_print("Applying ICA to remove artifacts.")
    eeg_columns = [col for col in df.columns if col != "Seizure"]
    signals = df[eeg_columns].values
    # Set n_components to number of channels if not provided
    n_components = n_components if n_components is not None else len(eeg_columns)
    ica = FastICA(n_components=n_components, random_state=42)
    sources = ica.fit_transform(signals)
    signals_reconstructed = ica.inverse_transform(sources)
    df_ica = pd.DataFrame(signals_reconstructed, columns=eeg_columns)
    df_ica["Seizure"] = df["Seizure"].values
    debug_print("ICA applied, signals reconstructed.")
    return df_ica


# ------------------------------
# Data Processing Functions
# ------------------------------

def drop_highly_correlated_features(df, threshold=CORRELATION_THRESHOLD):
    debug_print("Calculating correlation matrix for dropping highly correlated features.")
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    if to_drop:
        debug_print(f"Dropping columns due to high correlation: {to_drop}")
        df = df.drop(columns=to_drop)
    else:
        debug_print("No highly correlated columns found.")
    return df


def load_seizure_data(csv_path):
    debug_print(f"Loading seizure data from {csv_path}")
    BASE_DATE = datetime(2024, 1, 1)

    def convert_time(time_str):
        for fmt in ("%H:%M:%S", "%H:%M.%S", "%H.%M:%S", "%H.%M.%S"):
            try:
                t = datetime.strptime(time_str, fmt)
                return t.replace(year=BASE_DATE.year, month=BASE_DATE.month, day=BASE_DATE.day)
            except ValueError:
                continue
        return None

    df = pd.read_csv(csv_path)
    df["File Name"] = df["File Name"].str.strip().str.lower()
    df["Registration Start Time"] = df["Registration Start Time"].apply(convert_time)
    df["Seizure Start Time"] = df["Seizure Start Time"].apply(convert_time)
    df["Seizure End Time"] = df["Seizure End Time"].apply(convert_time)
    df.dropna(inplace=True)
    debug_print(f"Seizure data loaded with {len(df)} records.")
    return df


def get_common_channels(folder_path, common_channels_csv="common_channels.csv"):
    debug_print(f"Searching for common EEG channels in folder: {folder_path}")
    edf_files = glob.glob(os.path.join(folder_path, '**/*.edf'), recursive=True)
    common_channels = None
    for edf_file in edf_files:
        with pyedflib.EdfReader(edf_file) as f:
            labels = {ch for ch in f.getSignalLabels() if 'eeg' in ch.lower()}
            if common_channels is None:
                common_channels = labels
            else:
                common_channels.intersection_update(labels)
    if common_channels:
        channels_df = pd.DataFrame({"Channels": sorted(common_channels)})
        channels_df.to_csv(os.path.join(RES_DIR, common_channels_csv), index=False)
        debug_print(f"Common EEG channels saved to {os.path.join(RES_DIR, common_channels_csv)}")
        return list(common_channels)
    else:
        debug_print("No common EEG channels found.")
        return []


def process_and_save_edf(folder_path, seizure_data, common_channels, output_dir=RES_DIR):
    """
    Process each EDF file to create a DataFrame with EEG channels and a "Seizure" column.
    Applies bandpass filtering only.
    """
    debug_print("Starting EDF processing (without window-level balancing).")
    os.makedirs(output_dir, exist_ok=True)
    dfs = []
    edf_files = glob.glob(os.path.join(folder_path, '**/*.edf'), recursive=True)
    debug_print(f"Found {len(edf_files)} EDF files.")
    for edf_file in edf_files:
        file_name = os.path.basename(edf_file).lower()
        debug_print(f"Processing file: {file_name}")
        try:
            with pyedflib.EdfReader(edf_file) as f:
                sample_freq = f.getSampleFrequencies()[0]
                data_dict = {}
                for i, label in enumerate(f.getSignalLabels()):
                    if label in common_channels:
                        raw_signal = f.readSignal(i).astype(np.float32)
                        # Only apply bandpass filter
                        filtered = bandpass_filter(raw_signal, fs=sample_freq).astype(np.float32)
                        data_dict[label] = filtered
                if not data_dict:
                    debug_print(f"No matching EEG channels in {file_name}. Skipping.")
                    continue
                df = pd.DataFrame(data_dict)
                columns_to_drop = [col for col in df.columns if df[col].std() < STD_MIN or df[col].std() > STD_MAX]
                if columns_to_drop:
                    debug_print(f"Dropping columns in {file_name} due to abnormal std: {columns_to_drop}")
                df.drop(columns=columns_to_drop, inplace=True)
                # Add "Seizure" column from CSV
                df["Seizure"] = 0
                matching_rows = seizure_data[seizure_data["File Name"] == file_name]
                if matching_rows.empty:
                    debug_print(f"File '{file_name}' - No seizure events found in CSV.")
                else:
                    for _, row in matching_rows.iterrows():
                        reg_start_ts = row["Registration Start Time"].timestamp()
                        seiz_start_ts = row["Seizure Start Time"].timestamp()
                        seiz_end_ts = row["Seizure End Time"].timestamp()
                        start_idx = max(0, int((seiz_start_ts - reg_start_ts) * sample_freq))
                        end_idx = min(len(df) - 1, int((seiz_end_ts - reg_start_ts) * sample_freq))
                        df.iloc[start_idx:end_idx + 1, df.columns.get_loc("Seizure")] = 1
                        debug_print(
                            f"File '{file_name}': Labeling seizure event from "
                            f"{row['Seizure Start Time'].strftime('%H:%M:%S')} to "
                            f"{row['Seizure End Time'].strftime('%H:%M:%S')} (Registration start: "
                            f"{row['Registration Start Time'].strftime('%H:%M:%S')}). "
                            f"Marking indices {start_idx} to {end_idx} as seizure."
                        )
                # Keep only the common channels and "Seizure"
                expected_columns = set(common_channels)
                expected_columns.add("Seizure")
                for col in list(df.columns):
                    if col not in expected_columns:
                        debug_print(f"Removing unneeded column {col} from {file_name}")
                        df.drop(columns=[col], inplace=True)
                # Only keep files with some seizure data
                if df["Seizure"].sum() == 0:
                    debug_print(f"No seizure data found in {file_name}. Skipping file.")
                    continue
                dfs.append(df)
        except Exception as e:
            debug_print(f"Error processing {file_name}: {e}")
    if not dfs:
        debug_print("No data was processed. Exiting.")
        return None
    merged_df = pd.concat(dfs, ignore_index=True)
    debug_print(f"Merged data shape: {merged_df.shape}")
    # Optionally drop highly correlated channels
    eeg_columns = [col for col in merged_df.columns if col != "Seizure"]
    original_feature_count = len(eeg_columns)
    features_df = merged_df[eeg_columns]
    features_df_dropped = drop_highly_correlated_features(features_df, threshold=CORRELATION_THRESHOLD)
    dropped_feature_count = original_feature_count - features_df_dropped.shape[1]
    if dropped_feature_count / original_feature_count > 0.5:
        debug_print("Warning: Dropping highly correlated features would remove >50% channels. Skipping drop.")
    else:
        merged_df = pd.concat([features_df_dropped, merged_df["Seizure"]], axis=1)
        debug_print(f"After correlation drop, data has {merged_df.shape[1] - 1} EEG channels.")
    return merged_df


def extract_raw_windows_from_df(df, window_size=WINDOW_SIZE, overlap=OVERLAP, seizure_threshold=SEIZURE_THRESHOLD):
    """
    Splits the merged dataframe into raw EEG windows and computes a window label based on
    the mean of the "Seizure" column within that window.
    """
    debug_print("Extracting raw EEG windows and computing window labels.")
    step = window_size - int(overlap * window_size)
    eeg_columns = [col for col in df.columns if col != "Seizure"]
    X = df[eeg_columns].values
    y = df["Seizure"].values
    raw_windows = []
    labels_list = []
    for i, start in enumerate(range(0, len(df) - window_size + 1, step)):
        window = X[start:start + window_size, :]
        raw_windows.append(window)
        window_seizure = y[start:start + window_size]
        label = 1 if np.mean(window_seizure) > seizure_threshold else 0
        labels_list.append(label)
        if i < 5:
            debug_print(f"Window {i + 1}: Start index {start}, mean seizure value: {np.mean(window_seizure):.2f}, "
                        f"labeled as {'Seizure' if label == 1 else 'No Seizure'}.")
    debug_print(f"Extracted {len(raw_windows)} windows.")
    return np.array(raw_windows), np.array(labels_list)


def balance_windows(windows, labels, ratio=3):
    """
    Balances the window dataset by keeping all seizure windows (label 1) and randomly downsampling
    non-seizure windows (label 0) so that their count is at most ratio times the number of seizure windows.
    """
    seizure_idx = np.where(labels == 1)[0]
    nonseizure_idx = np.where(labels == 0)[0]
    n_seizure = len(seizure_idx)
    n_nonseizure = len(nonseizure_idx)
    max_nonseizure = n_seizure * ratio
    debug_print(f"Window counts before balancing: Seizure: {n_seizure}, Non-seizure: {n_nonseizure}")
    if n_nonseizure > max_nonseizure:
        selected_nonseizure_idx = np.random.choice(nonseizure_idx, size=max_nonseizure, replace=False)
    else:
        selected_nonseizure_idx = nonseizure_idx
    selected_indices = np.concatenate([seizure_idx, selected_nonseizure_idx])
    np.random.shuffle(selected_indices)
    balanced_windows = windows[selected_indices]
    balanced_labels = labels[selected_indices]
    debug_print(f"Window counts after balancing: {dict(zip(*np.unique(balanced_labels, return_counts=True)))}")
    return balanced_windows, balanced_labels


# ------------------------------
# Model Building Functions (with Attention)
# ------------------------------

def build_cnn_model(raw_input_shape):
    debug_print("Building CNN model with attention for raw EEG input.")
    raw_input = Input(shape=raw_input_shape)
    x = Conv1D(64, kernel_size=3, activation='relu')(raw_input)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    # Add attention layer to focus on important features across time steps
    x = Attention()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=raw_input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_gru_model(raw_input_shape):
    debug_print("Building LSTM (replacing GRU) model with attention for raw EEG input.")
    raw_input = Input(shape=raw_input_shape)
    x = LSTM(64, return_sequences=True)(raw_input)
    x = Attention()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=raw_input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_cnn_lstm_model(raw_input_shape):
    debug_print("Building CNN+LSTM model with attention for raw EEG input.")
    raw_input = Input(shape=raw_input_shape)
    x = Conv1D(64, kernel_size=3, activation='relu')(raw_input)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    # Use return_sequences=True to enable attention after LSTM
    x = LSTM(50, return_sequences=True)(x)
    x = Attention()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=raw_input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ------------------------------
# Helper: Plot Confusion Matrix
# ------------------------------

def plot_confusion_matrix(cm, classes, model_name, save_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ------------------------------
# Training and Evaluation Functions
# ------------------------------

def train_and_evaluate_model(build_model_fn, model_name, raw_data, labels):
    debug_print(f"Starting training for {model_name}.")
    X_train, X_test, y_train, y_test = train_test_split(
        raw_data, labels, test_size=0.2, random_state=42, stratify=labels
    )
    raw_input_shape = X_train.shape[1:]  # (window_size, num_channels)
    model = build_model_fn(raw_input_shape)
    debug_print(f"Training {model_name} with input shape {raw_input_shape}.")

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
    callbacks = [early_stopping, reduce_lr]

    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=48,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }
    debug_print(f"{model_name} metrics: {metrics}")
    model_filename = os.path.join(RES_DIR, model_name.lower().replace(" ", "_") + ".h5")
    model.save(model_filename)
    debug_print(f"Saved {model_name} as {model_filename}")
    results_filename = os.path.join(RES_DIR, model_name.lower().replace(" ", "_") + "_results.csv")
    pd.DataFrame([metrics]).to_csv(results_filename, index=False)
    debug_print(f"Saved {model_name} results to {results_filename}")

    plt.figure(figsize=(10, 4))
    n_plot = min(50, len(y_test))
    plt.plot(y_test[:n_plot], label="Actual", marker="o")
    plt.plot(y_pred[:n_plot], label="Predicted", marker="x")
    plt.title(f"Actual vs Predicted - {model_name} (First {n_plot} samples)")
    plt.xlabel("Sample index")
    plt.ylabel("Seizure label")
    plt.legend()
    plt.grid(True)
    actual_vs_pred_path = os.path.join(PLOTS_DIR, f"{model_name.lower().replace(' ', '_')}_actual_vs_pred.png")
    plt.savefig(actual_vs_pred_path)
    plt.close()
    debug_print(f"Saved actual vs predicted plot for {model_name} to {actual_vs_pred_path}")

    cm = confusion_matrix(y_test, y_pred)
    cm_path = os.path.join(PLOTS_DIR, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plot_confusion_matrix(cm, classes=["Non-Seizure", "Seizure"], model_name=model_name, save_path=cm_path)
    debug_print(f"Saved confusion matrix for {model_name} to {cm_path}")

    return metrics


# ------------------------------
# Main Pipeline
# ------------------------------

if __name__ == "__main__":
    debug_print("Starting main pipeline.")
    seizure_df = load_seizure_data("clean_seizures_data.csv")
    common_channels = get_common_channels("siena-scalp-eeg-database-1.0.0")
    merged_df = process_and_save_edf("siena-scalp-eeg-database-1.0.0", seizure_df, common_channels, output_dir=RES_DIR)
    if merged_df is None or merged_df.empty:
        debug_print("No processed data available. Exiting main pipeline.")
        exit()

    # Apply ICA to clean EEG channels before window extraction
    merged_df = apply_ica_to_df(merged_df)

    # Print sample rows for seizure and non-seizure data
    seizure_samples = merged_df[merged_df["Seizure"] == 1].sample(
        n=min(30, merged_df[merged_df["Seizure"] == 1].shape[0]), random_state=42)
    non_seizure_samples = merged_df[merged_df["Seizure"] == 0].sample(
        n=min(30, merged_df[merged_df["Seizure"] == 0].shape[0]), random_state=42)
    debug_print("Sample seizure rows from merged dataframe:")
    debug_print("Sample non-seizure rows from merged dataframe:")

    debug_print("Extracting raw EEG windows from merged data.")
    raw_windows, window_labels = extract_raw_windows_from_df(merged_df, window_size=WINDOW_SIZE, overlap=OVERLAP,
                                                             seizure_threshold=SEIZURE_THRESHOLD)
    debug_print(f"Raw windows shape: {raw_windows.shape}")
    debug_print(f"Window labels shape: {window_labels.shape}")

    unique, counts = np.unique(window_labels, return_counts=True)
    debug_print(f"Window label distribution before balancing: {dict(zip(unique, counts))}")

    balanced_windows, balanced_labels = balance_windows(raw_windows, window_labels, ratio=3)
    unique_bal, counts_bal = np.unique(balanced_labels, return_counts=True)
    debug_print(f"Window label distribution after balancing: {dict(zip(unique_bal, counts_bal))}")

    # Train and evaluate models using only raw EEG windows.
    metrics_cnn = train_and_evaluate_model(build_cnn_model, "CNN Model", balanced_windows, balanced_labels)
    metrics_gru = train_and_evaluate_model(build_gru_model, "GRU Model", balanced_windows, balanced_labels)
    metrics_cnn_lstm = train_and_evaluate_model(build_cnn_lstm_model, "CNN+LSTM Model", balanced_windows,
                                                balanced_labels)

    # Gather results into a DataFrame and plot as a table
    results = [metrics_cnn, metrics_gru, metrics_cnn_lstm]
    results_df = pd.DataFrame(results)
    print("\nResults Table:")
    print(results_df)

    # Plot results table using matplotlib table and save it
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=results_df.values, colLabels=results_df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    fig.tight_layout()
    table_path = os.path.join(PLOTS_DIR, "model_results_table.png")
    plt.title("Evaluation Metrics for EEG Seizure Detection Models")
    plt.savefig(table_path)
    plt.close()
    debug_print(f"Saved results table to {table_path}")

    debug_print("Training complete.")
