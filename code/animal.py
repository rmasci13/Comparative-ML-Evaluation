from utils import load_animal_data, plot_confusion_matrix, plot_results_table
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from a_decision_tree import dt_run
from a_knn import knn_run
from a_randomforest import rf_run
from a_ann import ann_run

def run():
    # Load data and run the models files to produce the optimal models
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_train, X_scaled_train, y_train, X_test, X_scaled_test, y_test = load_animal_data()
    labels = ['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']

    # DT
    best_dt, dt_train_mac_f1, dt_val_mac_f1 = dt_run(X_train, y_train, cv)
    plot_confusion_matrix(best_dt, X_test, y_test, 'Animal Test Set DT Optimal Confusion Matrix', '../plots/a_CM_dt_optimal.png', labels=labels)

    # KNN
    best_knn, knn_train_mac_f1, knn_val_mac_f1 = knn_run(X_scaled_train, y_train, cv)
    plot_confusion_matrix(best_knn, X_scaled_test, y_test, 'Animal Test Set KNN Optimal Confusion Matrix', '../plots/a_CM_knn_optimal.png', labels=labels)

    # RandomForest
    best_rf, rf_train_mac_f1, rf_val_mac_f1 = rf_run(X_train, y_train, cv)
    plot_confusion_matrix(best_rf, X_test, y_test, 'Animal Test Set RandomForest Optimal Confusion Matrix', '../plots/a_CM_rf_optimal.png', labels=labels)

    # ANN
    best_ann, ann_train_mac_f1, ann_val_mac_f1 = ann_run(X_scaled_train, y_train, cv=cv)
    plot_confusion_matrix(best_ann, X_scaled_test, y_test, 'Animal Test Set ANN Optimal Confusion Matrix', '../plots/a_CM_ANN_optimal.png', labels=labels)

    # Save models and data as list to pass
    models = [best_dt, best_knn, best_rf, best_ann]
    data = [X_train, X_scaled_train, y_train, X_test, X_scaled_test, y_test]

    # F1 scores
    dt_test_macro_f1, knn_test_macro_f1, rf_test_macro_f1, ann_test_macro_f1 = get_test_f1(models, data, 'macro')
    dt_test_weighted_f1, knn_test_weighted_f1, rf_test_weighted_f1, ann_test_weighted_f1 = get_test_f1(models, data, 'weighted')

    # Create Table
    results = [
        ['Decision Tree', f"{dt_train_mac_f1:.4f}", f"{dt_val_mac_f1:.4f}", f"{dt_test_macro_f1:.4f}", f"{dt_test_weighted_f1:.4f}"],
        ['KNN', f"{knn_train_mac_f1:.4f}", f"{knn_val_mac_f1:.4f}", f"{knn_test_macro_f1:.4f}", f"{knn_test_weighted_f1:.4f}"],
        ['RandomForest', f"{rf_train_mac_f1:.4f}", f"{rf_val_mac_f1:.4f}", f"{rf_test_macro_f1:.4f}", f"{rf_test_weighted_f1:.4f}"],
        ['ANN', f"{ann_train_mac_f1:.4f}", f"{ann_val_mac_f1:.4f}", f"{ann_test_macro_f1:.4f}", f"{ann_test_weighted_f1:.4f}"],
    ]
    cols = ['Model', 'Train Macro F1', 'Validation Macro F1', 'Test Macro F1', 'Test Weighted F1']
    plot_results_table(results, cols, 'Animal Dataset Result', '../plots/animal_results.png')
    print("Done Animal")


def get_test_f1(models, data, f1_version):
    # Extract info
    best_dt, best_knn, best_rf, best_ann = models
    X_train, X_scaled_train, y_train, X_test, X_scaled_test, y_test = data

    # Get scores
    dt_test_f1 = f1_score(y_test, best_dt.predict(X_test), average=f1_version)
    knn_test_f1 = f1_score(y_test, best_knn.predict(X_scaled_test), average=f1_version)
    rf_test_f1 = f1_score(y_test, best_rf.predict(X_test), average=f1_version)
    ann_test_f1 = f1_score(y_test, best_ann.predict(X_scaled_test), average=f1_version)

    return dt_test_f1, knn_test_f1, rf_test_f1, ann_test_f1