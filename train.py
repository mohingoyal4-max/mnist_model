import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    """
    Main function to train the MNIST digit classification model,
    evaluate it, and save the model and evaluation artifacts.
    """
    print("Loading and preprocessing data...")
    # 1. Data Processing & Splitting
    # Load the standard MNIST dataset
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Flatten the 28x28 images into 1D arrays (784 features)
    x_train_full = x_train_full.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    
    # Normalize the pixel values (0 to 1)
    x_train_full = x_train_full.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Split specifically: 50,000 samples for training
    # Standard MNIST has 60,000 so we take the first 50,000 for training.
    # The testing/validation set is strictly the 10,000 test samples.
    x_train = x_train_full[:50000]
    y_train = y_train_full[:50000]
    
    print("Building model architecture...")
    # 2. Model Architecture
    # Construct an ANN with exactly 3 hidden layers
    model = keras.Sequential([
        layers.InputLayer(input_shape=(784,), name='input_layer'),
        layers.Dense(64, activation='relu', name='hidden_1'),  # Hidden Layer 1
        layers.Dense(64, activation='relu', name='hidden_2'),  # Hidden Layer 2
        layers.Dense(32, activation='relu', name='hidden_3'),  # Hidden Layer 3
        layers.Dense(10, activation='softmax', name='output')  # Output Layer
    ])
    
    # 3. Hyperparameters & Training
    # Optimizer: Adam with exactly 1e-4 learning rate
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    
    # We use sparse categorical crossentropy because labels are integers (0-9)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("Training the model...")
    # Train the model with a batch size of 32
    history = model.fit(x_train, y_train, 
                        epochs=20, 
                        batch_size=32, 
                        validation_data=(x_test, y_test), 
                        verbose=1)
    
    # Save the trained model
    model_filename = 'mnist_model.h5'
    model.save(model_filename)
    print(f"Model saved successfully to {model_filename}")
    
    # 4. Evaluation & Visualizations
    print("Generating evaluation artifacts...")
    
    # Generate line plot for Loss vs. Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png', bbox_inches='tight')
    plt.close()
    
    # Evaluate on the 10,000 test set
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate Precision, Recall, F1-Score, and Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n--- Evaluation Metrics on Test Set ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("--------------------------------------\n")
    
    # Generate graphical Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Digit')
    plt.ylabel('True Digit')
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    plt.close()
    print("Visualizations saved: 'loss_plot.png', 'confusion_matrix.png'")

if __name__ == '__main__':
    main()
