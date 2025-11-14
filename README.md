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
│── src/
│   ├── data_loader.py
│   ├── feature_extractor.py
│   ├── svm_model.py
│   ├── knn_model.py
│   ├── rf_model.py
│   ├── mlp_model.py
│   └── evaluation.py
│── figures/
│── notebooks/
│── models/
│── Main.py
│── README.md
│── REPORT.md
```

---

## 🔥 How to Use This Project

### **1. Clone the repository**
```bash
git clone https://github.com/Adk-157/UrbanSound8K-PR-Project.git
cd UrbanSound8K-PR-Project
```

### **2. Extract the project zip**

Inside the repo, extract:
```
project.zip
```

This creates all necessary folder structures and configs.

### **3. Download the UrbanSound8K dataset**

The dataset is too large for GitHub. Download it from Google Drive:

👉 **[UrbanSound8K Dataset](https://drive.google.com/drive/folders/16W5iUjgl0DY2rL4_neORzwq18EY-nZpY?usp=sharing)**

Place it next to the repo like this:

```
/UrbanSound8K-PR-Project
/Data   <-- downloaded dataset
```

### **4. Install dependencies**

```bash
pip install -r requirements.txt
```

### **5. Run the full pipeline**

```bash
python Main.py
```

---

## 👥 Team Members & Contributions

| Member             | Contribution                                        |
| ------------------ | --------------------------------------------------- |
| **Adk 157**        | Data Loader, Augmentation, Random Forest Classifier |
| **Mari Venkatesh** | Feature Extraction, SVM Classifier                  |
| **Dharun SA**      | Euclidean and Mahalanobis k-NN                      |
| **E Mano Ranjan**  | Evaluation, Deep Learning MLP                       |

Full details are in `REPORT.md` and Appendix.

---

## 📊 Results (Confusion Matrices)

All confusion matrices are in `/figures`.

Key findings:
- k-NN models showed superior performance with proper distance metrics
- Classical PR methods outperformed the MLP baseline
- Feature engineering and dimensionality reduction were critical to success

---

## 🔗 Links

- **GitHub Repository:** [https://github.com/Adk-157/UrbanSound8K-PR-Project](https://github.com/Adk-157/UrbanSound8K-PR-Project)
- **Dataset (Google Drive):** [https://drive.google.com/drive/folders/16W5iUjgl0DY2rL4_neORzwq18EY-nZpY?usp=sharing](https://drive.google.com/drive/folders/16W5iUjgl0DY2rL4_neORzwq18EY-nZpY?usp=sharing)

---

## 🏫 Institute

**Indian Institute of Information Technology, Sri City**  
Electronics and Communication Engineering  
Academic Year 2025–26
