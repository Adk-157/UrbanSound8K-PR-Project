# UrbanSound8K Pattern Recognition Project  
### IIIT Sri City — Department of ECE  
### AY 2025–26

## 📌 Overview
This project implements a complete Pattern Recognition pipeline on the UrbanSound8K dataset.  
We compare **classical PR algorithms** (SVM, k-NN, Random Forest) against a **Deep Learning baseline (MLP)**.

**Final Best Accuracy:**  
- **Euclidean k-NN — 91.07%**  
- **Mahalanobis k-NN — 90.67%**  
- Random Forest — 87.77%  
- SVM — 86.57%  
- MLP — 83.85%

---

## 📁 Repository Structure

```
UrbanSound8K-PR-Project/
├── data/                      # Dataset directory (download separately)
├── figures/                   # All confusion matrices and plots
├── notebooks/                 # Jupyter notebooks for exploration
├── reports/                   # Project reports and documentation
├── src/                       # Source code modules
│   ├── classifiers.py         # All model implementations (SVM, k-NN, RF, MLP)
│   ├── data_loader.py         # Dataset loading and preprocessing
│   ├── evaluation.py          # Evaluation metrics and confusion matrices
│   ├── feature_extractor.py   # Audio feature extraction (65+ features)
│   ├── metadata_generator.py  # Augmented metadata and dataset CSV generation
│   └── utils.py               # Utility functions and helpers
├── tests/                     # Unit tests and validation scripts
├── Main.py                    # Main execution pipeline
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
└── REPORT.md                  # Complete technical report
```

---

## 🔥 How to Use This Project

### **1. Clone the repository**
```bash
git clone https://github.com/Adk-157/UrbanSound8K-PR-Project.git
cd UrbanSound8K-PR-Project
```

### **2. Download the UrbanSound8K dataset**

The dataset is **not included in the repo** (too large for GitHub).  
Download from Google Drive and place in the `data/` folder:

👉 **[UrbanSound8K Dataset](https://drive.google.com/drive/folders/16W5iUjgl0DY2rL4_neORzwq18EY-nZpY?usp=sharing)**

After download, your structure should look like:
```
UrbanSound8K-PR-Project/
├── data/ # <-- extracted dataset here
```

### **3. Install dependencies**

```bash
pip install -r requirements.txt
```

Required packages:
- librosa
- numpy
- pandas
- scikit-learn
- tensorflow/keras
- matplotlib
- seaborn

### **4. Run the full pipeline**

```bash
python Main.py
```

This will:
- Load dataset using `data_loader.py`
- Generate augmented metadata via `metadata_generator.py`
- Extract 65+ audio features using `feature_extractor.py`
- Train all models via `classifiers.py` (SVM, k-NN, RF, MLP)
- Evaluate and generate confusion matrices using `evaluation.py`
- Save results in `figures/`

---

## 📂 Source Code Structure

### **`src/classifiers.py`**
Contains all model implementations:
- **SVM Classifier** (RBF kernel with grid search)
- **Euclidean k-NN** (L2 distance with LDA)
- **Mahalanobis k-NN** (Covariance-weighted distance)
- **Random Forest** (100 estimators with feature importance)
- **MLP** (4-layer deep learning baseline)

### **`src/data_loader.py`**
Handles:
- Dataset loading and preprocessing
- Audio file reading and validation
- Data augmentation (noise, pitch shift, time stretch)

### **`src/feature_extractor.py`**
Extracts 65+ audio features:
- MFCC (20 coefficients)
- Chroma features (12-bin)
- Spectral features (centroid, bandwidth, rolloff, contrast)
- Temporal features (zero-crossing rate, RMS energy)

### **`src/metadata_generator.py`**
- Generates augmented metadata CSV
- Manages dataset splits and folds
- Creates training/testing metadata

### **`src/evaluation.py`**
- Computes accuracy, precision, recall, F1-score
- Generates confusion matrices
- Produces performance visualizations

### **`src/utils.py`**
- Helper functions and utilities
- Logging and configuration management

---

## 👥 Team Members & Contributions

| Member             | Contribution                                        |
| ------------------ | --------------------------------------------------- |
| **Adithya Ram S**        | Data Loader, Augmentation, Random Forest Classifier |
| **Mari Venkatesh** | Feature Extraction, SVM Classifier                  |
| **Dharun SA**      | Euclidean and Mahalanobis k-NN                      |
| **E Mano Ranjan**  | Evaluation, Deep Learning MLP                       |

Full details in `REPORT.md` → Appendix Section.

---

## 📊 Results Summary

| Model                | Accuracy | F1-Score |
| -------------------- | -------- | -------- |
| **Euclidean k-NN**   | 91.07%   | 0.9107   |
| **Mahalanobis k-NN** | 90.67%   | 0.9067   |
| **Random Forest**    | 87.77%   | 0.8777   |
| **SVM**              | 86.57%   | 0.8676   |
| **MLP**              | 83.85%   | 0.8385   |

**Key Findings:**
- Classical PR methods outperformed deep learning baseline
- LDA dimensionality reduction was critical for k-NN success
- Feature engineering proved more effective than raw deep learning on this dataset size

---

## 📈 Confusion Matrices

All confusion matrices are stored in `/figures/`:
- `euclidian.jpeg`
- `maha.jpeg`
- `RF.jpeg`
- `SVM.jpeg`
- `dlmlp.jpeg`

---

## 🔗 Links

- **GitHub Repository:** [https://github.com/Adk-157/UrbanSound8K-PR-Project](https://github.com/Adk-157/UrbanSound8K-PR-Project)
- **Dataset (Google Drive):** [https://drive.google.com/drive/folders/16W5iUjgl0DY2rL4_neORzwq18EY-nZpY?usp=sharing](https://drive.google.com/drive/folders/16W5iUjgl0DY2rL4_neORzwq18EY-nZpY?usp=sharing)

---

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

Tests cover:
- Data loader validation
- Feature extraction correctness
- Model input/output shapes

---

## 📝 Documentation

Complete technical report with methodology, results, and analysis:  
👉 **[REPORT.md](reports/REPORT.md)**

Additional documentation in `/reports/` folder.

---

## 🏫 Institute

**Indian Institute of Information Technology, Sri City**  
Electronics and Communication Engineering  
Academic Year 2025–26

---

## 📄 License

This project is for academic purposes only.  
Dataset: UrbanSound8K by J. Salamon, C. Jacoby and J. P. Bello.
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
