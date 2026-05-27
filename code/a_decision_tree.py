from utils import create_learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


def tuning(X_train, y_train, cv):
    # Grid search for optimal model on the peaks of max depth and splitting criterion
    depths = range(1, 21)
    param_grid = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': depths
    }

    base_model = DecisionTreeClassifier(class_weight="balanced", random_state=42)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='f1_macro',
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)

    print(f"Best Splitting Criterion: {grid_search.best_params_['criterion']}")
    print(f"Best Depth: {grid_search.best_params_['max_depth']}")

    # Hyperparameter tuning for max_depth
    train_scores = []
    val_scores = []

    for depth in depths:
        model = DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=42, criterion=grid_search.best_params_['criterion'])

        # Cross validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
        val_scores.append(cv_scores.mean())

        # Training score
        model.fit(X_train, y_train)
        train_scores.append(f1_score(y_train, model.predict(X_train), average="macro"))

    # Plot
    plt.figure()
    plt.plot(depths, train_scores, label='Training F1')
    plt.plot(depths, val_scores, label='Validation F1')
    plt.xlabel('Max Depth')
    plt.ylabel('Macro F1 Score')
    plt.title('Decision Tree Max Depth Tuning Curve - Animal')
    plt.legend()
    plt.savefig('../plots/a_dt_max_depth_tuning.png')
    plt.close()

    best_model = grid_search.best_estimator_
    train_score = f1_score(y_train, best_model.predict(X_train), average="macro")
    val_score = grid_search.best_score_

    return best_model, train_score, val_score


def dt_run(X_train, y_train, cv):
    # Create learning curve for the best model found by grid search
    print("Start DT Animal")
    best_model, train_f1_score, val_f1_score = tuning(X_train, y_train, cv)
    create_learning_curve(best_model, X_train, y_train, cv, 'DT Optimal Model Learning Curve  - Animal', '../plots/a_dt_learning_curve_optimal.png', 'f1_macro')
    print(f'Done DT Animal - Training F1: {train_f1_score}, Validation F1: {val_f1_score}')
    return best_model, train_f1_score, val_f1_score