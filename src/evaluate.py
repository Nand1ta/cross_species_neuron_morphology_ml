from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist()
    }
