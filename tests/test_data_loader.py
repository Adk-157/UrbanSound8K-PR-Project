"""
Test script for verifying UrbanSound8K dataset loading and quality.
Author: Member A (Data Engineer)
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import numpy as np
import pandas as pd
import librosa
from src.data_loader import load_dataset, load_metadata

BASE_PATH = "/content/UrbanSound8K-PR-Project/data"   # update if different

def check_missing_files(meta, base_path=BASE_PATH):
    """Check if any files referenced in CSV are missing."""
    missing = []
    for _, row in meta.iterrows():
        fold = f"fold{row.fold}"
        file_path = os.path.join(base_path, fold, row.slice_file_name)
        if not os.path.exists(file_path):
            missing.append(file_path)
    return missing


def analyze_waveforms(X, y):
    """Compute average duration and sampling uniformity."""
    lengths = [len(x) for x in X]
    avg_len = np.mean(lengths)
    min_len, max_len = np.min(lengths), np.max(lengths)
    print(f"\n📊 Average waveform length: {avg_len:.1f} samples")
    print(f"📉 Shortest: {min_len} | 📈 Longest: {max_len}")
    print(f"Total loaded: {len(X)} clips | Labels: {len(set(y))} unique classes")


def class_distribution(meta):
    """Print class distribution summary."""
    print("\n🎯 Class Distribution:")
    counts = meta['class'].value_counts().sort_index()
    for label, count in counts.items():
        print(f"{label:<20} → {count} samples")


if __name__ == "__main__":
    print("🔍 Running dataset verification tests...")

    # 1️⃣ Load metadata
    meta = load_metadata(BASE_PATH)
    print(f"✅ Metadata loaded with {len(meta)} entries")

    # 2️⃣ Check for missing files
    missing = check_missing_files(meta, BASE_PATH)
    if missing:
        print(f"⚠️ Missing {len(missing)} files out of {len(meta)}")
        print("Examples:", missing[:3])
    else:
        print("✅ No missing audio files found across all folds!")

    # 3️⃣ Load a sample subset to test waveform lengths
    print("\n🔈 Loading small sample (50 files) to analyze waveform stats...")
    X, y = load_dataset(BASE_PATH, max_files=50, clean=True, augment=False)
    analyze_waveforms(X, y)

    # 4️⃣ Show class balance
    class_distribution(meta)

    print("\n✅ Dataset verification completed successfully.")
