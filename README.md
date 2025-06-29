#  Deep Learning with TensorFlow/Keras

### Topic: Implementing Neural Networks with Keras

---

##  Summary

* Introduction to **TensorFlow** and its high-level API **Keras**
* Building and training a neural network using Keras
* Understanding **model compilation**, **fitting**, and **hyperparameters**
* Exploring **epochs**, **batch size**, and **learning rate**
* Evaluating models and making predictions

---

## 1. What is TensorFlow and Keras?

###  TensorFlow:

An **open-source deep learning library** developed by Google. It powers systems like:

* Google Translate
* Smart compose in Gmail
* YouTube video recommendations

###  Keras:

Keras is a **high-level API** that runs on top of TensorFlow. It makes model building easy, readable, and modular.

---

###  Analogy: TensorFlow vs Keras

> Think of **TensorFlow** as a car engine and **Keras** as the steering wheel and dashboard.
> Keras lets you **drive the power of TensorFlow** without touching the internals of the engine.

---

## 2. Building a Neural Network in Keras

We’ll use the **Fashion MNIST** dataset — images of shoes, shirts, and bags (28x28 grayscale pixels).

---

###  Load Data

```python
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

# Load dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize pixel values (0–255) → (0–1)
x_train, x_test = x_train / 255.0, x_test / 255.0
```

---

###  Define Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

model = Sequential([
    Flatten(input_shape=(28, 28)),           # Converts image to 1D
    Dense(128, activation='relu'),           # Hidden layer
    Dense(10, activation='softmax')          # Output layer (10 classes)
])
```

---

###  Analogy: Making a Sandwich

> * `Flatten` = Flatten the ingredients
> * `Dense` = Stack each layer of the sandwich (hidden layer)
> * `Softmax` = Final slice, assigns scores (which ingredient is best)

---

## 3. Compiling the Model

Before training, we **compile** the model by specifying:

* **Loss function** — how the model learns
* **Optimizer** — how weights are updated
* **Metrics** — how we evaluate performance

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

###  Analogy: Cooking Instructions

> * Loss = How bad the meal tastes (used to adjust the recipe)
> * Optimizer = The chef who adjusts seasoning
> * Metrics = The food critic’s final rating

---

## 4. Fitting the Model (Training)

```python
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
```

###  Key Terms:

| Term                 | Meaning                                             |
| -------------------- | --------------------------------------------------- |
| **Epoch**            | One complete pass through the training data         |
| **Batch Size**       | Number of samples processed before updating weights |
| **Validation Split** | Percentage of training data used for validation     |

---

###  Analogy: Studying for a Test

> * **Epoch** = How many times you review all your notes
> * **Batch size** = How many pages you study at a time
> * **Validation split** = Small quiz after each round of studying to check understanding

---

## 5. Evaluating the Model

```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)
```

---

###  Analogy: Final Exam

> After studying (training), evaluation is the **final exam** to test how much the model really learned — using **data it hasn’t seen before**.

---

## 6. Making Predictions

```python
predictions = model.predict(x_test)

# Predict the first image's label
import numpy as np
np.argmax(predictions[0])
```

---

###  Analogy: Guessing from Experience

> After training, the model can **guess what it sees** based on the patterns it’s learned — like a student identifying animal pictures after studying biology.

---

##  Summary Table

| Concept    | Explanation           | Analogy                 |
| ---------- | --------------------- | ----------------------- |
| TensorFlow | Deep learning library | Car engine              |
| Keras      | User-friendly API     | Steering wheel          |
| Compile    | Set up model config   | Cooking recipe          |
| Fit        | Train model           | Study sessions          |
| Epoch      | One full data pass    | Reading your notes once |
| Batch size | Samples per update    | Pages read per session  |
| Evaluate   | Final test            | Exam                    |
| Predict    | Use the model         | Real-world application  |

---

##  Final Code (All-in-One)

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.datasets import fashion_mnist

# Load and normalize data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)

# Predict
predictions = model.predict(x_test)
```


