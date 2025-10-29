import os
import pandas as pd
import librosa
import numpy as np

# ---------------------------
# 1. OPTIONAL CLEANING HELPERS
# ---------------------------

def remove_silence(y, top_db=30):
    """Removes silent sections from the waveform."""
    intervals = librosa.effects.split(y, top_db=top_db)
    y_trimmed = np.concatenate([y[start:end] for start, end in intervals])
    return y_trimmed

def add_noise(y, noise_factor=0.005):
    """Adds light Gaussian noise to the waveform."""
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

# ---------------------------
# 2. METADATA LOADER
# ---------------------------

def load_metadata(base_path="/content/UrbanSound8K-PR-Project/data"):
    """
    Loads the UrbanSound8K metadata CSV file.
    CSV path: /content/UrbanSound8K-PR-Project/data/UrbanSound8K.csv
    """
    csv_path = os.path.join(base_path, "UrbanSound8K.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found at {csv_path}")
    meta = pd.read_csv(csv_path)
    print(f"✅ Metadata loaded successfully: {len(meta)} entries")
    return meta

# ---------------------------
# 3. AUDIO LOADER
# ---------------------------

def load_audio(file_path, sr=22050, clean=True, augment=False):
    """
    Loads a single audio file, resamples, normalizes, and optionally cleans or augments.
    """
    try:
        y, sr = librosa.load(file_path, sr=sr)
        y = librosa.util.normalize(y)
        if clean:
            y = remove_silence(y)
        if augment:
            y = add_noise(y)
        return y, sr
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None, None

# ---------------------------
# 4. MAIN DATASET LOADER
# ---------------------------

def load_dataset(base_path="/content/UrbanSound8K-PR-Project/data", sr=22050, 
                 max_files=None, clean=True, augment=False):
    """
    Loads the entire UrbanSound8K dataset.
    base_path should contain fold1..fold10 and UrbanSound8K.csv
    Returns:
        X: list of audio waveforms
        y: numpy array of integer class labels
    """
    meta = load_metadata(base_path)
    X, y = [], []

    for idx, row in meta.iterrows():
        fold = f"fold{row.fold}"
        filename = row.slice_file_name
        file_path = os.path.join(base_path, fold, filename)

        if not os.path.exists(file_path):
            print(f"⚠️ File missing: {file_path}")
            continue

        y_audio, sr = load_audio(file_path, sr=sr, clean=clean, augment=augment)
        if y_audio is not None:
            X.append(y_audio)
            y.append(row.classID)

        if max_files and len(X) >= max_files:
            break

    print(f"\n✅ Loaded {len(X)} audio files successfully across {len(set(meta['fold']))} folds.")
    return X, np.array(y)

    def get_data_subset(base_path, n_samples=100):
      X, y = load_dataset(base_path=base_path, max_files=n_samples, clean=True)
      return X, y

