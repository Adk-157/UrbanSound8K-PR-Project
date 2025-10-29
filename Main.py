from src.data_loader import load_dataset

X, y = load_dataset("/content/UrbanSound8K-PR-Project/data", clean=True, augment=False, max_files=20)
print("Loaded:", len(X), "samples")
print("Example waveform length:", len(X[0]))
print("Example label:", y[0])

