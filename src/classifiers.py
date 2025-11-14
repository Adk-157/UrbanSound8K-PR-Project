# classifier.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from scipy.spatial import distance
import tensorflow as tf
from tensorflow.keras import layers, models
import warnings
import os

#  Configuration 
RANDOM_SEED = 42
TEST_SIZE = 0.2
N_FOLDS_TUNING = 5 
NOISE_STD_DEV = 0.02
SAMPLE_LIMIT = 20000 

tf.random.set_seed(RANDOM_SEED)
warnings.filterwarnings('ignore')

# PREPROCESSING FUNCTIONS 

def load_and_preprocess_data(data_path):
    df = pd.read_csv(data_path)
    df_filtered = df.iloc[:SAMPLE_LIMIT, :]

    X = df_filtered.drop("label", axis=1).values
    y = df_filtered["label"].values
    n_classes = len(np.unique(y))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    noise = np.random.normal(0, NOISE_STD_DEV, X_train_scaled.shape)
    X_train_noisy = X_train_scaled + noise 

    lda = LDA(n_components=n_classes - 1)
    X_train_reduced = lda.fit_transform(X_train_noisy, y_train)
    X_test_reduced = lda.transform(X_test_scaled)
    
    return X_train_reduced, X_test_reduced, y_train, y_test, n_classes


def get_mahalanobis_vi(X_train_reduced):
    try:
        covariance_matrix_train = np.cov(X_train_reduced, rowvar=False)
        VI_train = np.linalg.inv(covariance_matrix_train)
    except np.linalg.LinAlgError:
        VI_train = np.linalg.pinv(covariance_matrix_train)
    return VI_train


#  MODEL TRAINING AND EVALUATION FUNCTIONS 

def tune_and_train_knn(X_train, y_train, VI_train, n_classes):
    param_grid_knn = {'n_neighbors': np.arange(1, 16, 2), 'weights': ['uniform', 'distance']}
    knn_mah_grid = KNeighborsClassifier(metric='mahalanobis', metric_params={'V': VI_train})
    cv_splitter = StratifiedKFold(n_splits=N_FOLDS_TUNING, shuffle=True, random_state=RANDOM_SEED)

    grid_search_knn = GridSearchCV(
        knn_mah_grid, param_grid_knn, cv=cv_splitter, scoring='accuracy', n_jobs=-1
    )
    grid_search_knn.fit(X_train, y_train)

    best_k = grid_search_knn.best_params_['n_neighbors']
    best_weights = grid_search_knn.best_params_['weights']

    knn_euc_final = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean', weights=best_weights)
    knn_mah_final = KNeighborsClassifier(
        n_neighbors=best_k, 
        metric='mahalanobis', 
        metric_params={'V': VI_train}, 
        weights=best_weights
    )
    
    knn_euc_final.fit(X_train, y_train)
    knn_mah_final.fit(X_train, y_train)
    
    return {
        'Euclidean': knn_euc_final, 
        'Mahalanobis': knn_mah_final
    }, {'k': best_k, 'weights': best_weights}


def train_baseline_models(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    svm_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=RANDOM_SEED)

    rf_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)
    
    return {'RF': rf_model, 'SVM': svm_model}


def build_and_train_mlp(X_train, y_train, X_test, y_test, n_classes, epochs=50):
    input_dim = X_train.shape[1]

    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=32,
        verbose=0 
    )
    return model


def evaluate_model_performance(model, X_test, y_test, model_name="Model"):
    if isinstance(model, tf.keras.Model):
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
    else:
        y_pred = model.predict(X_test)
        
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    return acc, f1_macro, y_pred


# MAIN EXECUTION FUNCTION 

def run_full_pipeline(DATA_PATH):
    X_train_reduced, X_test_reduced, y_train, y_test, n_classes = load_and_preprocess_data(DATA_PATH)
    VI_train = get_mahalanobis_vi(X_train_reduced)
    
    knn_models, knn_params = tune_and_train_knn(X_train_reduced, y_train, VI_train, n_classes)
    baseline_models = train_baseline_models(X_train_reduced, y_train)

    all_models = {**knn_models, **baseline_models}

    mlp_model = build_and_train_mlp(X_train_reduced, y_train, X_test_reduced, y_test, n_classes, epochs=50)
    all_models['DL MLP'] = mlp_model
    
    results = {}
    for name, model in all_models.items():
        acc, f1_macro, y_pred = evaluate_model_performance(model, X_test_reduced, y_test, name)
        results[name] = {'Accuracy': acc, 'F1-Macro': f1_macro}

    comparison_df = pd.DataFrame({
        'Model': results.keys(),
        'Accuracy': [r['Accuracy'] for r in results.values()],
        'F1-Macro': [r['F1-Macro'] for r in results.values()]
    }).set_index('Model').sort_values(by='F1-Macro', ascending=False)
    
    return comparison_df
