import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Generar datos de prueba
def generate_sample_data():
    data = {
        "edad": np.random.randint(18, 70, 100),
        "ingresos_mensuales": np.random.randint(1000, 10000, 100),
        "compró": np.random.choice([0, 1], size=100)
    }
    df = pd.DataFrame(data)
    df.to_csv("datos_prueba.csv", index=False)
    print("Datos de prueba generados y guardados como 'datos_prueba.csv'.")
    return df

# Cargar los datos
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Datos cargados exitosamente.")
        return data
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return None

# Análisis exploratorio de datos (EDA)
def exploratory_data_analysis(data):
    print("\n--- Resumen del DataFrame ---")
    print(data.info())

    print("\n--- Estadísticas descriptivas ---")
    print(data.describe())

    print("\n--- Valores nulos ---")
    print(data.isnull().sum())

    print("\n--- Visualización de variables numéricas ---")
    data.hist(bins=20, figsize=(15, 10))
    plt.tight_layout()
    plt.show()

    print("\n--- Correlación entre variables ---")
    correlation = data.corr()
    sns.heatmap(correlation, annot=True, cmap="coolwarm")
    plt.show()

# Preprocesamiento de datos
def preprocess_data(data, target_column):
    # Manejo de valores nulos
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Escalado de características
    scaler = StandardScaler()
    features = data_imputed.drop(target_column, axis=1)
    features_scaled = scaler.fit_transform(features)

    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, data_imputed[target_column], test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

# Modelo de aprendizaje automático
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Modelo entrenado exitosamente.")
    return model

# Evaluación del modelo
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\n--- Reporte de clasificación ---")
    print(classification_report(y_test, y_pred))

    print("\n--- Matriz de confusión ---")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.show()

# Reducción de dimensionalidad (opcional)
def reduce_dimensionality(data, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    print(f"Datos reducidos a {n_components} componentes principales.")
    return reduced_data

# Flujo principal
def main():
    print("\n--- Generando datos de prueba ---")
    generate_sample_data()

    file_path = input("Introduce la ruta del archivo CSV (o presiona Enter para usar 'datos_prueba.csv'): ")
    if not file_path:
        file_path = "datos_prueba.csv"
    target_column = input("Introduce el nombre de la columna objetivo (por defecto 'compró'): ")
    if not target_column:
        target_column = "compró"

    data = load_data(file_path)
    if data is not None:
        exploratory_data_analysis(data)
        
        X_train, X_test, y_train, y_test = preprocess_data(data, target_column)
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()