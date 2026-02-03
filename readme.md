\# Iris Flower Classification with PyTorch

\## 📊 Project Overview

This project implements a \*\*machine learning classification model\*\* that predicts the \*\*species of Iris flowers\*\* based on their physical characteristics. Using PyTorch, we build a neural network that analyzes four botanical measurements to determine whether a flower belongs to one of three Iris species.

\## 🌸 What Does This Model Predict?

The model predicts the \*\*Iris flower species\*\* from these three categories:

| Species | Characteristics | Sample Count |

|---------|----------------|--------------|

| \*\*Iris-setosa\*\* | Distinctive, easily separable | 50 samples |

| \*\*Iris-versicolor\*\* | Intermediate characteristics | 50 samples |

| \*\*Iris-virginica\*\* | Similar to versicolor | 50 samples |

\## 🔍 Input Features

The model uses \*\*four numerical measurements\*\* (in centimeters) from each flower:

1\. \*\*Sepal Length\*\* - Length of the sepal (leaf-like structure)

2\. \*\*Sepal Width\*\* - Width of the sepal

3\. \*\*Petal Length\*\* - Length of the petal

4\. \*\*Petal Width\*\* - Width of the petal

Example input: \`\[5.1, 3.5, 1.4, 0.2\]\` → \*\*Prediction: Iris-setosa\*\*

\## 🧠 How It Works

\### 1. \*\*Data Processing\*\*

\- Loads the classic Iris dataset (150 samples)

\- Normalizes features for better training

\- Encodes species names into numerical labels

\- Splits data into training, validation, and test sets

\### 2. \*\*Neural Network Architecture\*\*

\`\`\`

Input (4 features) → Hidden Layer (10 neurons) → Hidden Layer (10 neurons) → Output (3 classes)

\`\`\`

\- \*\*Activation\*\*: ReLU (introduces non-linearity)

\- \*\*Parameters\*\*: 193 trainable weights and biases

\### 3. \*\*Training Process\*\*

\- \*\*Loss Function\*\*: CrossEntropyLoss (for multi-class classification)

\- \*\*Optimizer\*\*: Adam (adaptive learning rate)

\- \*\*Training\*\*: 100 epochs with batch size of 8

\- \*\*Validation\*\*: Monitored to prevent overfitting

\### 4. \*\*Prediction Pipeline\*\*

\`\`\`

Raw Measurements → Normalization → Neural Network → Class Probabilities → Species Prediction

\`\`\`

\## 📈 Model Performance

| Metric | Performance |

|--------|-------------|

| \*\*Training Accuracy\*\* | ~95-100% |

| \*\*Validation Accuracy\*\* | ~90-95% |

| \*\*Test Accuracy\*\* | ~90-95% |

| \*\*Prediction Time\*\* | < 1ms per sample |

\## 💡 Real-World Applications

This model demonstrates concepts that apply to:

\- \*\*Plant species identification\*\* in botanical research

\- \*\*Medical diagnosis\*\* from patient measurements

\- \*\*Quality control\*\* in manufacturing based on product dimensions

\- \*\*Customer classification\*\* based on behavioral features

\## 🔄 Model Capabilities

\### ✅ What It Can Do:

\- Accurately classify Iris flowers with ~95% accuracy

\- Handle measurements with decimal precision

\- Provide confidence scores for predictions

\- Run efficiently on both CPU and GPU

\- Save and reload trained models for later use

\### ⚠️ Limitations:

\- Only works with the three Iris species in the dataset

\- Requires all four measurements as input

\- Trained on a specific dataset (may not generalize to all Iris variations)

\- Simple architecture suitable for educational purposes

\## 🎯 Educational Value

This project serves as a \*\*practical introduction to PyTorch\*\* and covers:

1\. \*\*Tensors & Operations\*\*: Data manipulation fundamentals

2\. \*\*Automatic Differentiation\*\*: How neural networks learn

3\. \*\*Model Building\*\*: Creating custom neural architectures

4\. \*\*Training Loops\*\*: Manual control over the learning process

5\. \*\*Model Persistence\*\*: Saving and loading trained models

6\. \*\*GPU Acceleration\*\*: Utilizing hardware for faster computation

\## 🚀 Getting Predictions

\`\`\`python

\# Example: Making a prediction

sample\_flower = \[6.7, 3.0, 5.2, 2.3\] # Features in cm

prediction = model.predict(sample\_flower) # Returns: "Iris-virginica"

\`\`\`

The model outputs both the predicted species and confidence scores for each possible class, allowing you to see how certain the model is about its prediction.

\## 📋 Summary

This project demonstrates a \*\*complete machine learning pipeline\*\* from data loading to prediction, using a classic dataset to predict flower species with high accuracy. It's designed as both a working classification system and an educational resource for learning PyTorch fundamentals.