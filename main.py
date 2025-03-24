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
# Objectif: Comprendre comment charger et préparer les données pour l'entraînement d'un modèle.
path = kagglehub.dataset_download("datamunge/sign-language-mnist")
print("Path to dataset files:", path)

# Read training data
with open(path + '/sign_mnist_train.csv', 'r') as file:
    train_data = list(csv.reader(file))[1:]  # Skip header row
    
# Read test data 
with open(path + '/sign_mnist_test.csv', 'r') as file:
    test_data = list(csv.reader(file))[1:]  # Skip header row

# Convert to numpy arrays and separate features/labels
X_train = np.array([row[1:] for row in train_data], dtype='float32')
y_train = np.array([row[0] for row in train_data], dtype='int32')

X_test = np.array([row[1:] for row in test_data], dtype='float32')
y_test = np.array([row[0] for row in test_data], dtype='int32')

# Reshape images to 28x28
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

# Exercice 2: Prétraitement des Données
# Objectif: Apprendre à utiliser des techniques de prétraitement pour améliorer la qualité des données d'entrée.
# Convert labels to one-hot encoding
num_classes = 25  # ASL alphabet has 25 classes (excluding J and Z which require motion)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Setup data augmentation for training
train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Setup validation/test data generator (no augmentation needed)
test_datagen = ImageDataGenerator()

# Create data generators
train_generator = train_datagen.flow(
    X_train, y_train,
    batch_size=32
)

test_generator = test_datagen.flow(
    X_test, y_test,
    batch_size=32
)

# Exercice 3: Construction d'un Modèle de Réseau de Neurones Convolutif (CNN)
# Objectif: Construire et comprendre l'architecture d'un modèle CNN.
model_path = 'asl_model.h5'

# Check if the model already exists
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Model loaded from disk.")
else:
    # Define the CNN model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Exercice 4: Entraînement et Évaluation du Modèle
    # Objectif: Entraîner le modèle et évaluer ses performances.
    # Compile the model
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

    # Train the model using the data generators
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // 32,
        epochs=15,
        validation_data=test_generator,
        validation_steps=len(X_test) // 32
    )

    # Save the model
    model.save(model_path)
    print("Model trained and saved to disk.")

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_generator, steps=len(X_test) // 32)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Exercice 5: Visualisation des Résultats
    # Objectif: Visualiser les performances du modèle pour mieux comprendre son comportement.
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

# Exercice 6: Discussion et Conclusion
# Objectif: Réfléchir sur les apprentissages et discuter des prochaines étapes possibles.
# (Discussion points can be added here in the workshop)

def load_image(index):
    img_array = X_test[index]  # Use the index directly from the event
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)

    # Map the predicted label to the corresponding letter
    label_mapping = {i: chr(i + 65) for i in range(num_classes)}  # Assuming labels are 0-24 for A-Y
    letter = label_mapping[predicted_label]

    result_label.config(text=f"Predicted Label: {letter}")

    # Display the true label in the terminal
    true_label = np.argmax(y_test[index])  # Get the true label from the one-hot encoded array
    true_letter = chr(true_label + 65)  # Map the true label to the corresponding letter
    print(f"True Label: {true_letter}")  # Print the true label to the terminal

    img_tk = ImageTk.PhotoImage(Image.fromarray((img_array[0] * 255).astype(np.uint8).reshape(28, 28)))
    image_label.config(image=img_tk)
    image_label.image = img_tk

# Create the main window
root = tk.Tk()
root.title("Image Classification")

# Create a frame to hold the images
image_frame = tk.Frame(root)
image_frame.pack()

# Create a label to display the image
image_label = tk.Label(root)
image_label.pack()

# Display 100 images in a grid
for i in range(100):
    img_array = X_test[i].reshape(28, 28) * 255
    img = Image.fromarray(img_array.astype(np.uint8))
    img_tk = ImageTk.PhotoImage(img)
    img_label = tk.Label(image_frame, image=img_tk)
    img_label.image = img_tk  # Keep a reference to avoid garbage collection
    img_label.grid(row=i // 10, column=i % 10)  # 10 images per row
    img_label.bind("<Button-1>", lambda event, index=i: load_image(index))  # Bind click event to load_image

# Create a label to display the result
result_label = tk.Label(root, text="Predicted Label: ")
result_label.pack()
root.mainloop()

def display_result(index):
    img_array = X_test[index]  # Use the index directly from the event
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)

    # Map the predicted label to the corresponding letter
    letter = chr(predicted_label + 65)  # Assuming labels are 0-24 for A-Y
    true_label = np.argmax(y_test[index])  # Get the true label from the one-hot encoded array
    true_letter = chr(true_label + 65)  # Map the true label to the corresponding letter
    result_label.config(text=f"Predicted Label: {letter}, True Label: {true_letter}")  # Display both predicted and true labels

    # Create a label to display the true label
    true_label_display = tk.Label(root, text="True Label: ")
    true_label_display.pack()

    # Update the display_result function to show the true label
    def display_result(index):
        img_array = X_test[index]  # Use the index directly from the event
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = model.predict(img_array)
        predicted_label = np.argmax(prediction)

        # Map the predicted label to the corresponding letter
        letter = chr(predicted_label + 65)  # Assuming labels are 0-24 for A-Y
        true_label = np.argmax(y_test[index])  # Get the true label from the one-hot encoded array
        true_letter = chr(true_label + 65)  # Map the true label to the corresponding letter
        result_label.config(text=f"Predicted Label: {letter}, True Label: {true_letter}")  # Display both predicted and true labels
        true_label_display.config(text=f"True Label: {true_letter}")  # Update the true label display
        true_label_display.pack()  # Ensure the true label display is packed
