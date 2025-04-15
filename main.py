import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import kagglehub
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image, ImageTk
import os

# Exercice 1: Introduction et Préparation des Données
def load_dataset():
    path = kagglehub.dataset_download("datamunge/sign-language-mnist")
    print("Path to dataset files:", path)

    with open(os.path.join(path, 'sign_mnist_train.csv'), 'r') as file:
        train_data = [row for row in csv.reader(file)][1:]

    with open(os.path.join(path, 'sign_mnist_test.csv'), 'r') as file:
        test_data = [row for row in csv.reader(file)][1:]

    X_train = np.array([row[1:] for row in train_data], dtype='float32').reshape(-1, 28, 28, 1) / 255.0
    y_train = np.array([row[0] for row in train_data], dtype='int32')

    X_test = np.array([row[1:] for row in test_data], dtype='float32').reshape(-1, 28, 28, 1) / 255.0
    y_test = np.array([row[0] for row in test_data], dtype='int32')

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    return X_train, y_train, X_test, y_test

# Exercice 2: Prétraitement des Données
def preprocess_data(y_train, y_test, num_classes=25):
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator()

    return train_datagen, test_datagen

# Exercice 3: Construction d'un Modèle de Réseau de Neurones Convolutif (CNN)
def build_or_load_model(model_path, input_shape, num_classes):
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print("Model loaded from disk.")
    else:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    return model

# Exercice 4: Entraînement et Évaluation du Modèle
def train_and_evaluate_model(model, train_generator, test_generator, X_train, X_test, model_path, epochs=15):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // 32,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=len(X_test) // 32
    )

    model.save(model_path)
    print("Model trained and saved to disk.")

    test_loss, test_accuracy = model.evaluate(test_generator, steps=len(X_test) // 32)
    print(f"Test accuracy: {test_accuracy:.4f}")
    return history

# Exercice 5: Visualisation des Résultats
def plot_training_results(history):
    epochs = range(len(history.history['accuracy']))

    plt.plot(epochs, history.history['accuracy'], 'r', label='Training accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, history.history['loss'], 'r', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

# Exercice 6: Discussion et Conclusion
def load_image(index, X_test, y_test, model, result_label, image_label):
    img_array = X_test[index].reshape(1, 28, 28, 1)
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)
    letter = chr(predicted_label + 65)

    result_label.config(text=f"Predicted Label: {letter}")
    true_letter = chr(np.argmax(y_test[index]) + 65)
    print(f"True Label: {true_letter}")

    img_tk = ImageTk.PhotoImage(Image.fromarray((img_array[0] * 255).astype(np.uint8).reshape(28, 28)))
    image_label.config(image=img_tk)
    image_label.image = img_tk

def main():
    X_train, y_train, X_test, y_test = load_dataset()
    train_datagen, test_datagen = preprocess_data(y_train, y_test)
    train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
    test_generator = test_datagen.flow(X_test, y_test, batch_size=32)

    model_path = 'asl_model.h5'
    model = build_or_load_model(model_path, input_shape=(28, 28, 1), num_classes=25)

    if not os.path.exists(model_path):
        history = train_and_evaluate_model(model, train_generator, test_generator, X_train, X_test, model_path)
        plot_training_results(history)

    root = tk.Tk()
    root.title("Image Classification")

    image_frame = tk.Frame(root)
    image_frame.pack()

    image_label = tk.Label(root)
    image_label.pack()

    result_label = tk.Label(root, text="Predicted Label: ")
    result_label.pack()

    for i in range(100):
        img_array = (X_test[i].reshape(28, 28) * 255).astype(np.uint8)
        img_tk = ImageTk.PhotoImage(Image.fromarray(img_array))
        img_label = tk.Label(image_frame, image=img_tk)
        img_label.image = img_tk
        img_label.grid(row=i // 10, column=i % 10)
        img_label.bind("Load", lambda event, index=i: load_image(index, X_test, y_test, model, result_label, image_label))

    root.mainloop()

if __name__ == "__main__":
    main()
