from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from src.preprocess import load_data, preprocess

def train_model():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    joblib.dump(model, "model/model.pkl")
    return acc

if __name__ == "__main__":
    acc = train_model()
    print(f"Accuracy: {acc}")
