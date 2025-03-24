import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import data_preparation as dp

# Exercice 2: Prétraitement des Données
# Objectif: Apprendre à utiliser des techniques de prétraitement pour améliorer la qualité des données d'entrée.

def preprocess_data(X_train, y_train, X_test, y_test):
    # Convertissez les étiquettes en encodage one-hot
    # Indice: Utilisez `tf.keras.utils.to_categorical` pour encoder les étiquettes.

    # Configurez l'augmentation des données pour l'entraînement
    # Indice: Utilisez `ImageDataGenerator` pour appliquer des transformations comme la rotation et le décalage.
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Créez les générateurs de données d'entraînement
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=32
    )

    # Configurez le générateur de données de validation/test (aucune augmentation nécessaire)
    # Indice: Utilisez `ImageDataGenerator` pour créer un générateur de données de test sans augmentation donc sans aucun paramètre.

    # Créez le générateur de données de validation

    return train_generator  # À compléter avec le générateur de validation si nécessaire

preprocess_data(dp.load_and_prepare_data())