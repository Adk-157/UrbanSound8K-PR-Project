import os
import pandas as pd
import numpy as np
import librosa

AUGMENT_TYPES = ['noise', 'shift', 'pitch', 'stretch']

def generate_augmented_metadata(base_csv="data/UrbanSound8K.csv", output_csv="data/UrbanSound8K_augmented.csv"):
    """
    Creates a new metadata CSV that includes augmented versions for each file.
    Each augmentation gets a new row with a unique file_id and augment_type.
    """
    meta = pd.read_csv(base_csv)
    new_entries = []

    for idx, row in meta.iterrows():
        for aug in AUGMENT_TYPES:
            new_row = row.copy()
            new_row['augment_type'] = aug
            new_row['augmented_file_name'] = f"{row.slice_file_name.split('.')[0]}_{aug}.wav"
            new_entries.append(new_row)

    # Append all new rows
    aug_meta = pd.concat([meta, pd.DataFrame(new_entries)], ignore_index=True)
    aug_meta['augment_type'] = aug_meta.get('augment_type', 'none')
    aug_meta['augmented_file_name'] = aug_meta.get('augmented_file_name', aug_meta['slice_file_name'])

    aug_meta.to_csv(output_csv, index=False)
    print(f"✅ Augmented metadata created and saved to {output_csv}")
    print(f"Total entries: {len(aug_meta)} (original: {len(meta)}, new: {len(aug_meta) - len(meta)})")

    return aug_meta
