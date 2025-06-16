import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def train_model(X_train, y_train, model_type="rf"):
    models = {
        "rf": RandomForestClassifier(n_estimators=100),
        "svm": SVC(kernel="rbf"),
        "knn": KNeighborsClassifier(n_neighbors=3),
    }
    model = models[model_type]
    model.fit(X_train, y_train)
    return model

def save_model(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    from preprocessing import load_data, split_data

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    
    model_rf = train_model(X_train, y_train, "rf")
    save_model(model_rf, "models/random_forest.pkl")
    
    print("Modelos entrenados y guardados.")
