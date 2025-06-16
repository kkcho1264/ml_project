import unittest
import sys
import os

# Agregar el directorio padre al path para importar src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import train_model
from src.preprocessing import load_data, split_data

class TestModels(unittest.TestCase):
    def setUp(self):
        """Configura los datos de prueba antes de cada test."""
        self.df = load_data()
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(self.df)

    def test_random_forest(self):
        """Prueba si el modelo Random Forest se entrena correctamente."""
        model = train_model(self.X_train, self.y_train, model_type="rf")
        self.assertIsNotNone(model, "El modelo Random Forest no debería ser None")

    def test_svm(self):
        """Prueba si el modelo SVM se entrena correctamente."""
        model = train_model(self.X_train, self.y_train, model_type="svm")
        self.assertIsNotNone(model, "El modelo SVM no debería ser None")

    def test_knn(self):
        """Prueba si el modelo KNN se entrena correctamente."""
        model = train_model(self.X_train, self.y_train, model_type="knn")
        self.assertIsNotNone(model, "El modelo KNN no debería ser None")

if __name__ == "__main__":
    unittest.main()

