# the three sklearn classifiers we compare
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_classifiers(random_state: int = 42) -> dict:
    return {
        # rbf-kernel svm
        "SVM": SVC(kernel="rbf", random_state=random_state),
        # small mlp with two hidden layers
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            max_iter=300,
            random_state=random_state,
        ),
        # plain logistic regression
        "LogReg": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
    }


def fit_scaler(X_train):
    # subtract mean and divide by std per feature, learned only from train
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def train_and_evaluate(classifiers, X_train, y_train, X_test, y_test):
  
    results = {}
    for name, clf in classifiers.items():
        # fit on train, predict on test
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # store the fitted model + its test predictions and metrics
        results[name] = {
            "model": clf,
            "y_pred": y_pred,
            "accuracy": accuracy_score(y_test, y_pred),
            "report": classification_report(
                y_test, y_pred, target_names=["fake", "real"], digits=4,
                zero_division=0,
            ),
        }
    return results
