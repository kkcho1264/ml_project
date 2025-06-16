from sklearn.metrics import accuracy_score, classification_report
import pickle

def load_model(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return acc, report

if __name__ == "__main__":
    from preprocessing import load_data, split_data

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    model = load_model("models/random_forest.pkl")
    acc, report = evaluate_model(model, X_test, y_test)
    
    print(f"Precisión: {acc:.4f}")
    print(f"Reporte de clasificación:\n{report}")

