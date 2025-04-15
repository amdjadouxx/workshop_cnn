import numpy as np
import kagglehub
import csv

# Exercice 1: Introduction et Préparation des Données
# Objectif: Comprendre comment charger et préparer les données pour l'entraînement d'un modèle.

def load_and_prepare_data():
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    # Téléchargez le dataset en utilisant kagglehub
    path = kagglehub.dataset_download("datamunge/sign-language-mnist")

    # Lisez les données d'entraînement
    # Indice: Utilisez le module `csv` pour lire les fichiers CSV et ignorez la première ligne (en-tête).

    # Lisez les données de test
    # Indice: Répétez le processus de lecture pour les données de test.

    # Convertissez en tableaux numpy et séparez les caractéristiques/étiquettes
    # Indice: Utilisez `np.array` pour convertir les données en tableaux numpy.

    # Reshapez les images en 28x28
    # Indice: Utilisez la méthode `reshape` pour ajuster la forme des données.

    # Normalisez les valeurs des pixels
    # Indice: Divisez les valeurs des pixels par 255.0 pour les normaliser.

    return X_train, y_train, X_test, y_test
