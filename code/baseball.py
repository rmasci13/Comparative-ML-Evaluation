from b_decision_tree import dt_run
from b_knn import knn_run
from b_log_reg import logreg_run
from b_svm import svm_run
from utils import load_baseball_data, plot_confusion_matrix, plot_results_table
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import PrecisionRecallDisplay, f1_score
import matplotlib.pyplot as plt

def run():
    # Load data and run the models files to produce the optimal models
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_train, X_scaled_train, y_train, X_test, X_scaled_test, y_test = load_baseball_data()
    labels = [0, 1]

    # Get all best models and training and validation f1 scores
    best_dt, dt_train_f1, dt_val_f1 = dt_run(X_train, y_train, cv)
    plot_confusion_matrix(best_dt, X_test, y_test, 'Baseball Test Set DT Optimal Confusion Matrix', '../plots/b_CM_dt_optimal.png', labels=labels)

    best_knn, knn_train_f1, knn_val_f1 = knn_run(X_scaled_train, y_train, cv)
    plot_confusion_matrix(best_knn, X_scaled_test, y_test, 'Baseball Test Set KNN Optimal Confusion Matrix','../plots/b_CM_KNN_optimal.png', labels=labels)

    best_logreg, logreg_train_f1, logreg_val_f1 = logreg_run(X_scaled_train, y_train, cv)
    plot_confusion_matrix(best_logreg, X_scaled_test, y_test, 'Baseball Test Set LogReg Optimal Confusion Matrix','../plots/b_CM_logreg_optimal.png', labels=labels)

    best_svm, svm_train_f1, svm_val_f1 = svm_run(X_scaled_train, y_train, cv)
    plot_confusion_matrix(best_svm, X_scaled_test, y_test, 'Baseball Test Set SVM Optimal Confusion Matrix','../plots/b_CM_svm_optimal.png', labels=labels)

    # Save models and data as list to pass
    models = [best_dt, best_knn, best_logreg, best_svm]
    data = [X_train, X_scaled_train, y_train, X_test, X_scaled_test, y_test]

    # Plot curves and get the scores
    dt_auc, knn_auc, logreg_auc, svm_auc = get_pr_auc(models, data)

    # Get F1 scores on test set
    dt_test_f1, knn_test_f1, logreg_test_f1, svm_test_f1 = get_test_f1(models, data)

    # Create Table
    results = [
        ['Decision Tree', f"{dt_train_f1:.4f}", f"{dt_val_f1:.4f}", f"{dt_test_f1:.4f}", f"{dt_auc:.4f}"],
        ['KNN', f"{knn_train_f1:.4f}", f"{knn_val_f1:.4f}", f"{knn_test_f1:.4f}", f"{knn_auc:.4f}"],
        ['Logistic Regression', f"{logreg_train_f1:.4f}", f"{logreg_val_f1:.4f}", f"{logreg_test_f1:.4f}", f"{logreg_auc:.4f}"],
        ['SVM', f"{svm_train_f1:.4f}", f"{svm_val_f1:.4f}", f"{svm_test_f1:.4f}", f"{svm_auc:.4f}"],
    ]
    cols = ['Model', 'Train F1', 'Val F1', 'Test F1', 'Test PR-AUC']
    plot_results_table(results, cols, 'Baseball Dataset Result', '../plots/baseball_results.png')

    print("Done Baseball")


def get_test_f1(models, data):
    # Extract info
    best_dt, best_knn, best_logreg, best_svm = models
    X_train, X_scaled_train, y_train, X_test, X_scaled_test, y_test = data

    # Get scores
    dt_test_f1 = f1_score(y_test, best_dt.predict(X_test))
    knn_test_f1 = f1_score(y_test, best_knn.predict(X_scaled_test))
    logreg_test_f1 = f1_score(y_test, best_logreg.predict(X_scaled_test))
    svm_test_f1 = f1_score(y_test, best_svm.predict(X_scaled_test))

    return dt_test_f1, knn_test_f1, logreg_test_f1, svm_test_f1


def get_pr_auc(models, data):
    # Extract info
    best_dt, best_knn, best_logreg, best_svm = models
    X_train, X_scaled_train, y_train, X_test, X_scaled_test, y_test = data

    # Precision-Recall AUC Curves
    figDT, axDT = plt.subplots()
    figKNN, axKNN = plt.subplots()
    figLR, axLR = plt.subplots()
    figSVM, axSVM = plt.subplots()

    dt_prauc = PrecisionRecallDisplay.from_estimator(best_dt, X_test, y_test, ax=axDT)
    axDT.set_title('DT Precision-Recall AUC')
    dt_auc = dt_prauc.average_precision

    knn_prauc = PrecisionRecallDisplay.from_estimator(best_knn, X_scaled_test, y_test, ax=axKNN)
    axKNN.set_title('KNN Precision-Recall AUC')
    knn_auc = knn_prauc.average_precision

    logreg_prauc = PrecisionRecallDisplay.from_estimator(best_logreg, X_scaled_test, y_test, ax=axLR)
    axLR.set_title('LogReg Precision-Recall AUC')
    logreg_auc = logreg_prauc.average_precision

    svm_prauc = PrecisionRecallDisplay.from_estimator(best_svm, X_scaled_test, y_test, ax=axSVM)
    axSVM.set_title('SVM Precision-Recall AUC')
    svm_auc = svm_prauc.average_precision

    figDT.savefig('../plots/DT_Precision-Recall_AUC.png')
    figKNN.savefig('../plots/KNN_Precision-Recall_AUC.png')
    figLR.savefig('../plots/LogReg_Precision-Recall_AUC.png')
    figSVM.savefig('../plots/SVM_Precision-Recall_AUC.png')

    return dt_auc, knn_auc, logreg_auc, svm_auc