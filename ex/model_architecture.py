import tensorflow as tf

# Exercice 3: Construction d'un Modèle de Réseau de Neurones Convolutif (CNN)
# Objectif: Construire et comprendre l'architecture d'un modèle CNN.

def create_model():
    # Pseudocode pour définir le modèle CNN
    # 1. Créez un modèle séquentiel
    # 2. Ajoutez une couche de convolution avec un certain nombre de filtres, taille de noyau, et fonction d'activation
    # 3. Ajoutez une couche de pooling pour réduire la dimensionnalité
    # 4. Répétez les étapes 2 et 3 pour ajouter plus de couches
    # 5. Aplatissez les sorties des couches précédentes
    # 6. Ajoutez une ou plusieurs couches denses avec une fonction d'activation
    # 7. Ajoutez une couche de sortie avec le nombre de classes et une fonction d'activation appropriée

    num_class = 25

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='sigmoid'),
        tf.keras.layers.Dense(25, activation='softmax')
    ])

    return model