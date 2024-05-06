# libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
# dataset
file_path = r"C:\Users\irani\Downloads\ML Dr. Lars\winequality-red.csv"
df = pd.read_csv(file_path, sep=';')
X = df.drop('quality', axis=1)
y = df['quality']
# preprocessing
def log_transform(X):
    return np.log1p(X)

features_to_transform = X.columns
preprocessor = ColumnTransformer(
    transformers=[
        ('log_transform', FunctionTransformer(log_transform), features_to_transform),
        ('std_scaler', StandardScaler(), features_to_transform)
    ]
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

#classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(random_state=24),
    'Support Vector Machine': SVC(random_state=24),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=24)
}

#default hyperparameters
default_results = {}
for name, clf in classifiers.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    default_results[name] = {
        'Mean CV Accuracy': scores.mean(),
        'Test Accuracy': accuracy_score(y_test, y_pred)
    }

# hyperparameter search spaces
hyperparameter_spaces = {
    'Random Forest': {
        'classifier__n_estimators': Integer(100, 1000),
        'classifier__max_depth': Integer(10, 100),
        'classifier__min_samples_split': Integer(2, 10),
    },
    'Support Vector Machine': {
        'classifier__C': Real(0.1, 10, prior='log-uniform'),
        'classifier__gamma': Categorical(['scale', 'auto']),
    },
    'Logistic Regression': {
        'classifier__C': Real(0.1, 10, prior='log-uniform'),
    }
}

# tuning and evaluation
tuned_results = {}
for name, clf in classifiers.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    bayes_search = BayesSearchCV(pipeline, search_spaces=hyperparameter_spaces[name], n_iter=10, cv=5,
                                 scoring='accuracy', random_state=24)
    bayes_search.fit(X_train, y_train)
    y_pred = bayes_search.predict(X_test)
    tuned_results[name] = {
        'Mean CV Accuracy': bayes_search.best_score_,
        'Test Accuracy': accuracy_score(y_test, y_pred),
        'Best Params': bayes_search.best_params_
    }
def print_formatted_comparison(default_results, tuned_results):
    # Header
    print(
        f"{'Classifier':<25} {'Default Train':>15} {'Tuned Train':>15} {'Default Test':>15} {'Tuned Test':>15} {'Improvement':>15}")
    print("-" * 90)

    for clf in default_results:
        default_train = default_results[clf]['Mean CV Accuracy']
        tuned_train = tuned_results[clf]['Mean CV Accuracy']
        default_test = default_results[clf]['Test Accuracy']
        tuned_test = tuned_results[clf]['Test Accuracy']
        improvement = tuned_test - default_test
        print(
            f"{clf:<25} {default_train:>15.4f} {tuned_train:>15.4f} {default_test:>15.4f} {tuned_test:>15.4f} {improvement:>15.4f}")


print_formatted_comparison(default_results, tuned_results)


def plot_accuracy_comparisons(default_results, tuned_results):
    # Prepare data
    classifier_names = list(default_results.keys())
    default_train_accuracies = [default_results[clf]['Mean CV Accuracy'] for clf in classifier_names]
    tuned_train_accuracies = [tuned_results[clf]['Mean CV Accuracy'] for clf in classifier_names]
    default_test_accuracies = [default_results[clf]['Test Accuracy'] for clf in classifier_names]
    tuned_test_accuracies = [tuned_results[clf]['Test Accuracy'] for clf in classifier_names]

    index = np.arange(len(classifier_names))
    bar_width = 0.30
    plt.figure(figsize=(10, 6))
    plt.bar(index - bar_width/2, default_train_accuracies, bar_width, label='Default Training', color='peru')
    plt.bar(index + bar_width/2, tuned_train_accuracies, bar_width, label='Tuned Training', color='olive')
    plt.xlabel('Classifiers')
    plt.ylabel('Mean CV Accuracy')
    plt.title('Training Accuracy: Default vs Tuned Models')
    plt.xticks(index, classifier_names)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.bar(index - bar_width/2, default_test_accuracies, bar_width, label='Default Test', color='peru')
    plt.bar(index + bar_width/2, tuned_test_accuracies, bar_width, label='Tuned Test', color='olive')
    plt.xlabel('Classifiers')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy: Default vs Tuned Models')
    plt.xticks(index, classifier_names)
    plt.legend()
    plt.tight_layout()
    plt.show()
plot_accuracy_comparisons(default_results, tuned_results)


''''''''''
Results
Classifier                  Default Train     Tuned Train    Default Test      Tuned Test     Improvement
------------------------------------------------------------------------------------------
for random state 42 and itr=10
Random Forest                      0.6779          0.6873          0.6469          0.6656          0.0188
Support Vector Machine             0.5943          0.6216          0.5750          0.6094          0.0344
Logistic Regression                0.6044          0.6076          0.5719          0.5625         -0.0094
'''