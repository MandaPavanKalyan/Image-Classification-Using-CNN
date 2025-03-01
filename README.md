# Convolutional Neural Network (CNN) for Image Classification

## Features
- **Data Loading**: Load training and testing data from CSV files.
- **Data Preprocessing**: Reshape and normalize the data for model training.
- **Model Building**: Create a CNN model using TensorFlow and Keras.
- **Model Training**: Train the model on the training data.
- **Model Evaluation**: Evaluate the model on the testing data.
- **Prediction**: Make predictions on new data samples.

---

## Installation

Ensure you have Python installed (>=3.6). Install the required dependencies:

```sh
pip install numpy tensorflow matplotlib
```

## Usage

1. **Prepare Data**: Ensure you have the following CSV files with your data:
    - `input.csv`: Training input data.
    - `labels.csv`: Training labels.
    - `input_test.csv`: Testing input data.
    - `labels_test.csv`: Testing labels.

2. **Load and Preprocess Data**: Load the data from the CSV files, reshape, and normalize it.

3. **Build and Compile the Model**: Create a Sequential model with convolutional, pooling, and dense layers. Compile the model with a loss function, optimizer, and metrics.

4. **Train the Model**: Train the model on the training data for a specified number of epochs.

5. **Evaluate the Model**: Evaluate the model on the testing data to check its performance.

6. **Make Predictions**: Use the trained model to make predictions on new data samples.

### Example Code

Here is an example of the complete code:

```python
import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import matplotlib.pyplot as plt
import random

# Load Data
x_train = np.loadtxt('input.csv', delimiter=',')
y_train = np.loadtxt('labels.csv', delimiter=',')
x_test = np.loadtxt('input_test.csv', delimiter=',')
y_test = np.loadtxt('labels_test.csv', delimiter=',')

# Reshape and Normalize Data
x_train = x_train.reshape(len(x_train), 100, 100, 3)
y_train = y_train.reshape(len(y_train), 1)
x_test = x_test.reshape(len(x_test), 100, 100, 3)
y_test = y_test.reshape(len(y_test), 1)
x_train = x_train / 255
x_test = x_test / 255

# Print Data Shapes
print("shape of x_train: ", x_train.shape)
print("shape of y_train: ", y_train.shape)
print("shape of x_test: ", x_test.shape)
print("shape of y_test: ", y_test.shape)

# Display Random Training Image
idx = random.randint(0, len(x_train))
plt.imshow(x_train[idx, :])
plt.show()

# Build and Compile Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train Model
model.fit(x_train, y_train, epochs=5, batch_size=64)

# Evaluate Model
model.evaluate(x_test, y_test)

# Make Predictions
idx2 = random.randint(0, len(y_test))
plt.imshow(x_test[idx2, :])
plt.show()

y_pred = model.predict(x_test[idx2, :].reshape(1, 100, 100, 3))
print(y_pred)

y_pred = y_pred > 0.5
if y_pred == 0:
    pred = "dog"
else:
    pred = "cat"
print("Our model says it is a:", pred)
```

## Future Enhancements

- Improve model accuracy with more complex architectures.
- Add support for more image classes.
- Implement data augmentation for better generalization.
- Integrate visualization tools for model performance.

## Author

Developed by Pavan Kalyan Manda

Website Developer | IoT & Embedded Systems Enthusiast | AI/ML Developer
