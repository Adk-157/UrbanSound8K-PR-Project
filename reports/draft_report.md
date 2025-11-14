# UrbanSound8K Pattern Recognition Project  
### Comparative Analysis of Classical and Deep Learning Models  
**IIIT Sri City — ECE Department (2025–26)**

---

## 1. Abstract

This project implements and compares classical Pattern Recognition models (SVM, k-NN, Random Forest) and a Deep Learning baseline (MLP) on the UrbanSound8K dataset. After comprehensive preprocessing, feature extraction, augmentation, and evaluation, **Euclidean k-NN achieved the highest accuracy at 91.07%**. Our findings demonstrate that well-engineered classical PR approaches with proper feature extraction and dimensionality reduction can outperform deep learning baselines on structured audio classification tasks.

---

## 2. Dataset

- **Dataset:** UrbanSound8K  
- **Total Samples:** 8,732 audio clips  
- **Classes:** 10 environmental sound categories
  - Air Conditioner, Car Horn, Children Playing, Dog Bark, Drilling, Engine Idling, Gun Shot, Jackhammer, Siren, Street Music
- **Audio Format:** WAV files with varying durations (≤4 seconds)
- **Preprocessing:** 
  - Normalization and standardization
  - Data augmentation (noise injection, pitch shift, time stretch, time shift)
  - Spectrogram-based feature computation
  - Train-test split: 80%-20%

---

## 3. Methodology

### 3.1 Data Loading & Augmentation (Adk 157)

**Responsibilities:**
- Implemented complete dataset ingestion pipeline with metadata parsing
- Created dynamic augmentation system with configurable parameters
- Developed Random Forest classifier with hyperparameter tuning

**Key Contributions:**
- **Augmentation Techniques:**
  - Noise injection: Added Gaussian white noise for robustness
  - Pitch shifting: ±2 semitones to simulate recording variations
  - Time stretching: 0.8x–1.2x speed without pitch change
  - Time shifting: Temporal offset to capture position invariance
  
- **Random Forest Implementation:**
  - Ensemble of 100 decision trees with max depth optimization
  - Feature importance analysis for interpretability
  - Cross-validation for hyperparameter selection

**Results:**
- **Accuracy:** 87.77%
- **F1-Score:** 0.8777
- Strong performance across most classes with balanced precision-recall

---

### 3.2 Feature Extraction & SVM (Mari Venkatesh)

**Responsibilities:**
- Designed and implemented comprehensive feature extraction pipeline
- Extracted 65+ audio features per clip using Librosa
- Implemented SVM classifier with kernel selection and grid search

**Feature Categories:**
1. **MFCC (Mel-Frequency Cepstral Coefficients):** 20 coefficients capturing timbral texture
2. **Chroma Features:** 12-bin chroma vector for harmonic content
3. **Spectral Features:** Centroid, bandwidth, rolloff, contrast
4. **Temporal Features:** Zero-crossing rate, RMS energy
5. **Statistical Aggregations:** Mean, variance, min, max for robustness

**SVM Configuration:**
- Kernel: RBF (Radial Basis Function)
- Grid search over C and gamma parameters
- Feature normalization using MinMaxScaler

**Results:**
- **Accuracy:** 86.57%
- **F1-Score:** 0.8676
- Effective at separating distinct sound classes with clear spectral signatures

---

### 3.3 k-NN: Euclidean & Mahalanobis Distance (Dharun SA)

**Responsibilities:**
- Implemented both Euclidean and Mahalanobis distance-based k-NN classifiers
- Applied Linear Discriminant Analysis (LDA) for dimensionality reduction
- Optimized k value through cross-validation

**Key Contributions:**
- **Euclidean k-NN:**
  - Standard L2 distance metric
  - LDA reduced feature space to 9 dimensions
  - k=5 neighbors selected via grid search
  
- **Mahalanobis k-NN:**
  - Covariance-weighted distance metric accounting for feature correlations
  - Improved handling of correlated audio features
  - Same LDA preprocessing and k=5 configuration

**Results:**
- **Euclidean k-NN Accuracy:** 91.07% ⭐ (Best Overall)
- **Mahalanobis k-NN Accuracy:** 90.67%
- Minimal performance gap suggests features are well-normalized
- LDA dimensionality reduction was critical to both models' success

---

### 3.4 Deep Learning MLP & Evaluation (E Mano Ranjan)

**Responsibilities:**
- Designed and implemented 4-layer Multi-Layer Perceptron (MLP)
- Conducted unified evaluation across all models
- Generated all confusion matrices and performance metrics

**MLP Architecture:**
- **Input Layer:** 65 features
- **Hidden Layers:** 
  - Layer 1: 128 neurons, ReLU activation, Dropout 0.3
  - Layer 2: 64 neurons, ReLU activation, Dropout 0.3
  - Layer 3: 32 neurons, ReLU activation
- **Output Layer:** 10 neurons, Softmax activation
- **Optimizer:** Adam with learning rate 0.001
- **Training:** 100 epochs with early stopping

**Evaluation Framework:**
- Accuracy, Precision, Recall, F1-Score
- Per-class confusion matrices
- Cross-validation results

**Results:**
- **MLP Accuracy:** 83.85%
- **Observation:** Deep learning baseline underperformed classical methods
- **Analysis:** Limited dataset size and strong feature engineering favored classical PR

---

## 4. Results Summary

| Model                | Accuracy | F1-Score | Key Strength                          |
| -------------------- | -------- | -------- | ------------------------------------- |
| **Euclidean k-NN**   | 91.07%   | 0.9107   | Best overall, effective LDA reduction |
| **Mahalanobis k-NN** | 90.67%   | 0.9067   | Handles feature correlations          |
| **Random Forest**    | 87.77%   | 0.8777   | Interpretable, robust ensemble        |
| **SVM**              | 86.57%   | 0.8676   | Strong with RBF kernel                |
| **MLP**              | 83.85%   | 0.8385   | Baseline deep learning                |

---

## 5. Confusion Matrices

All confusion matrices are located in:
```
/figures/
```

Matrices include:
- `euclidian.jpeg`
- `maha.jpeg`
- `RF.jpeg`
- `Svm.jpeg`
- `dlmlp.jpeg`

**Key Observations:**
- Car Horn and Gun Shot: High precision across all models (distinct spectral signatures)
- Children Playing and Street Music: Most challenging (overlapping spectral characteristics)
- k-NN models showed more balanced confusion patterns

---

## 6. Conclusion

This project demonstrates that **classical Pattern Recognition methods with proper feature engineering can outperform deep learning baselines** on structured audio classification tasks, particularly when:

1. **Dataset size is limited** (8,732 samples insufficient for deep learning to reach full potential)
2. **Features are well-engineered** (65+ hand-crafted audio features captured discriminative information)
3. **Dimensionality reduction is applied** (LDA reduced overfitting and improved k-NN performance)

**Key Takeaway:** Euclidean k-NN with LDA achieved 91.07% accuracy, demonstrating that distance-based methods excel when feature spaces are properly reduced and normalized.

**Future Work:**
- Explore deeper neural architectures (CNNs on spectrograms)
- Implement transfer learning with pre-trained audio models
- Expand dataset through synthetic augmentation
- Ensemble methods combining classical and deep learning approaches

---

## 7. Repository & Dataset Links

### 🔗 GitHub Repository
[https://github.com/Adk-157/UrbanSound8K-PR-Project](https://github.com/Adk-157/UrbanSound8K-PR-Project)

### 🔗 Dataset (Google Drive)
[https://drive.google.com/drive/folders/16W5iUjgl0DY2rL4_neORzwq18EY-nZpY?usp=sharing](https://drive.google.com/drive/folders/16W5iUjgl0DY2rL4_neORzwq18EY-nZpY?usp=sharing)

---

## 8. Appendix — Detailed Member Contributions

### **Adk 157 — Data Loading, Augmentation, Random Forest**

Implemented the foundational data pipeline that enabled all subsequent modeling work:

- **Data Loader (`data_loader.py`):**
  - Parsed UrbanSound8K metadata CSV files
  - Created robust file path handling for fold-based organization
  - Implemented batch loading with memory optimization
  - Added advanced data augmentation (noise, shift, pitch, stretch)
  
- **Augmentation System:**
  - Noise injection with controllable SNR
  - Pitch shifting using librosa pitch_shift
  - Time stretching with librosa time_stretch
  - Time shifting with circular padding
  - Configurable augmentation probability per technique

- **Random Forest Classifier (`classifiers.py`):**
  - Scikit-learn RandomForestClassifier with 100 estimators
  - Hyperparameter tuning: max_depth, min_samples_split, min_samples_leaf
  - Feature importance extraction for interpretability
  - Achieved 87.77% accuracy with balanced performance

**Code Highlights:**
- Modular augmentation functions allowing flexible pipeline composition
- Efficient feature caching to avoid redundant computation
- Comprehensive logging for debugging and performance tracking

---

### **Mari Venkatesh — Feature Extraction, SVM**

Designed the comprehensive feature extraction system that formed the basis for all classical PR models:

- **Feature Extraction (`feature_extractor.py`):**
  - **MFCC:** 20 coefficients with delta and delta-delta features
  - **Chroma STFT:** 12-bin chromagram for pitch class profiles
  - **Spectral Features:**
    - Spectral centroid (brightness)
    - Spectral bandwidth (spread)
    - Spectral rolloff (high-frequency cutoff)
    - Spectral contrast (peak-valley differences)
  - **Temporal Features:**
    - Zero-crossing rate (noisiness/percussiveness)
    - RMS energy (loudness)
  - **Statistical Aggregation:** Mean, std, min, max for each feature

- **SVM Implementation (`classifiers.py`):**
  - Explored linear, polynomial, and RBF kernels
  - Grid search optimization over C ∈ [0.1, 1, 10, 100] and gamma ∈ [0.001, 0.01, 0.1, 1]
  - Selected RBF kernel with C=10, gamma=0.01
  - MinMaxScaler normalization for kernel stability

**Results:**
- 86.57% accuracy demonstrates effectiveness of feature engineering
- Feature extraction pipeline reused by all other models

---

### **Dharun SA — Euclidean & Mahalanobis k-NN**

Implemented both distance-based k-NN variants and achieved the project's best results:

- **k-NN Implementation (`classifiers.py`):**
  - Euclidean distance: Standard L2 metric
  - Mahalanobis distance: Covariance-weighted metric
  - K-value optimization via cross-validation (k=5 optimal)
  - Efficient distance computation with NumPy broadcasting

- **Dimensionality Reduction:**
  - Applied Linear Discriminant Analysis (LDA)
  - Reduced 65-dimensional feature space to 9 dimensions
  - Preserved class discriminability while reducing noise
  - Critical to achieving 91%+ accuracy

**Key Insight:**
The minimal gap between Euclidean (91.07%) and Mahalanobis (90.67%) suggests that feature normalization and LDA effectively decorrelated features, reducing the advantage of covariance-weighted distances.

**Implementation Details:**
- Handled singular covariance matrices in Mahalanobis computation
- Optimized nearest neighbor search for large test sets
- Implemented tie-breaking for equal distances

---

### **E Mano Ranjan — Evaluation & Deep Learning**

Created the unified evaluation framework and implemented the deep learning baseline:

- **MLP Architecture (`classifiers.py`):**
  ```
  Input(65) → Dense(128, ReLU) → Dropout(0.3) 
           → Dense(64, ReLU) → Dropout(0.3)
           → Dense(32, ReLU)
           → Dense(10, Softmax)
  ```
  - Trained with Adam optimizer, categorical cross-entropy loss
  - Early stopping with patience=10 to prevent overfitting
  - Achieved 83.85% accuracy

- **Evaluation Suite (`evaluation.py`):**
  - Confusion matrix generation for all models
  - Per-class precision, recall, F1-score
  - ROC curves and AUC metrics
  - Statistical significance tests between models
  - Visualization utilities for all plots

**Analysis:**
The MLP underperformed compared to classical methods due to:
1. Limited training data (8,732 samples insufficient for deep learning)
2. Strong hand-crafted features that captured discriminative patterns
3. No convolutional architecture to exploit spatial structure in spectrograms

**Evaluation Insights:**
- Gun Shot and Car Horn: Easiest classes (distinct signatures)
- Children Playing and Street Music: Hardest classes (variable patterns)
- k-NN models showed most balanced confusion matrices

---

## 9. References

1. Salamon, J., Jacoby, C., & Bello, J. P. (2014). A dataset and taxonomy for urban sound research. *ACM Multimedia*.
2. McFee, B., et al. (2015). librosa: Audio and music signal analysis in Python. *SciPy*.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*.
4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*.

---

**Project Completed:** November 2025  
**Institution:** Indian Institute of Information Technology, Sri City  
**Department:** Electronics and Communication Engineering

