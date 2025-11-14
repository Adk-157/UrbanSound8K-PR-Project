import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
#                 MODEL EVALUATION FUNCTION
# ============================================================
def evaluate_model(model, X_test, y_test, class_names):
    """
    Evaluates a model and prints:
    - Accuracy
    - Precision (macro)
    - Recall (macro)
    - F1-score (macro)
    - Per-class classification report
    - Returns predictions + summary DataFrame
    """

    preds = model.predict(X_test)

    # ------------------ Metrics -----------------------
    acc = accuracy_score(y_test, preds)
    precision_macro = precision_score(y_test, preds, average="macro")
    recall_macro = recall_score(y_test, preds, average="macro")
    f1_macro = f1_score(y_test, preds, average="macro")

    # ------------------ Summary Table -----------------
    summary_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)"],
        "Value": [acc, precision_macro, recall_macro, f1_macro]
    })

    print("\n==================== MODEL SUMMARY ====================")
    print(summary_df.to_string(index=False))

    # ------------------ Per-Class Report ----------------
    print("\n================ CLASSIFICATION REPORT ================")
    print(classification_report(y_test, preds, target_names=class_names))

    return preds, summary_df



# ============================================================
#            CONFUSION MATRIX VISUALIZATION
# ============================================================
def plot_confusion(y_test, preds, class_names):
    """
    Plots the confusion matrix using seaborn heatmap.
    Automatically closes the plot window to avoid blocking execution.
    """
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # 🟢 FIX: Non-blocking plot, auto-close window
    plt.show(block=False)
    plt.pause(10)
    plt.close()
