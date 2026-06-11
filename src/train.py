from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def train_rf(X, y):
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )
    model.fit(X, y)
    return model


def cross_validate(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv)
    return scores
