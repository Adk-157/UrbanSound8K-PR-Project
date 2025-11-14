import os
import warnings
import numpy as np
import pandas as pd

from src import classifiers
from src.evaluation import evaluate_model, plot_confusion

# =============================
# CONFIG
# =============================
DATA_PATH = 'data/UrbanSound8K_Features_Extract.csv'

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
warnings.filterwarnings('ignore')

# Sync seeds
np.random.seed(classifiers.RANDOM_SEED)
classifiers.tf.random.set_seed(classifiers.RANDOM_SEED)


# ============================================================
#                    MAIN EXECUTION PIPELINE
# ============================================================
def main():

    print("\n==================================================")
    print("       STARTING PROJECT EXECUTION       ")
    print("==================================================")

    # -------------------------------------------------------
    # 1. CHECK DATA EXISTS
    # -------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        print("\n FATAL ERROR: Feature CSV file not found.")
        print(f"Expected file at: {DATA_PATH}\n")
        print("Please ensure that:")
        print("✓ 'UrbanSound8K_Features_Extract.csv' is inside the 'data/' folder")
        return

    print("\n CSV File Found. Proceeding...\n")

    try:
        # -------------------------------------------------------
        # 2. RUN CLASSICAL + DL PIPELINE (Member C's code)
        # -------------------------------------------------------
        print(" Running full classical + DL pipeline.... ")
        comparison_df, all_models, preds_dict, X_test, y_test = classifiers.run_full_pipeline(DATA_PATH)

        print("\n==================================================")
        print("                 MODEL COMPARISON")
        print("==================================================")
        print(comparison_df)
        print("==================================================\n")
        

        # -------------------------------------------------------
        # 3. RETRIEVE BEST MODEL + TEST SPLIT
        # -------------------------------------------------------
        print("🔍 Retrieving best model and test split...")
        best_model = classifiers.get_best_trained_model()
        X_test, y_test = classifiers.get_test_split()
        class_names = classifiers.CLASS_NAMES

        print("✅ Best model loaded successfully:", comparison_df.index[0])

        # -------------------------------------------------------
        # 4. EVALUATION (Your responsibility)
        # -------------------------------------------------------
        print("\n📊 Evaluating best model on test set...")

        preds, summary_df = evaluate_model(
            best_model,
            X_test,
            y_test,
            class_names
        )

        print("\n==================== FINAL SUMMARY TABLE ====================")
        print(summary_df.to_string(index=False))

        # -------------------------------------------------------
        # 5. PLOT CONFUSION MATRIX
        # -------------------------------------------------------
        print("\n📈 Generating Confusion Matrix...")
        plot_confusion(y_test, preds, class_names)

        print("\n\n==================================================")
        print("            PIPELINE EXECUTED SUCCESSFULLY")
        print("==================================================")
        print(f"🏆 Best Model: {comparison_df.index[0]}")
        print("==================================================\n")

        # -------------------------------------------------------
        # 5. PLOT CONFUSION MATRIX FOR ALL MODELS
        # -------------------------------------------------------
        print("\n📈 Generating confusion matrices for ALL models...\n")

        for model_name, preds in preds_dict.items():
            print(f"\n================ Confusion Matrix: {model_name} ================")
            plot_confusion(y_test, preds, class_names)


    except Exception as e:
        print("\n FATAL ERROR during pipeline execution.")
        print("--------------------------------------------------")
        print(f"Error: {str(e)}")
        print("--------------------------------------------------")
        raise e  # for debugging


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
