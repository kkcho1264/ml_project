import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
    df["species"] = df["species"].astype("category").cat.codes  # Convertir etiquetas a números
    return df

def split_data(df, test_size=0.2):
    X = df.drop(columns=["species"])
    y = df["species"]
    return train_test_split(X, y, test_size=test_size, random_state=42)

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print("Datos preprocesados y divididos correctamente.")

