# ResNet Hyperparameter Tuning with KerasTuner

This project demonstrates how to build and tune a ResNet-inspired deep learning model for the Fashion MNIST dataset using TensorFlow and KerasTuner. The code leverages hyperparameter tuning to optimize the model architecture and training process.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Code Overview](#code-overview)
- [Running the Code](#running-the-code)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

---

## Features
- Implements a ResNet-inspired architecture with residual blocks.
- Uses KerasTuner for hyperparameter optimization, including:
  - Number of filters in the initial convolutional layer.
  - Number of residual blocks.
  - Filters in each residual block.
  - Learning rate for the Adam optimizer.
- Performs hyperparameter tuning to maximize validation accuracy.
- Evaluates the best model on the test dataset.

## Requirements

To run this project, you need the following libraries installed:

- TensorFlow 2.0+
- KerasTuner

You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

## Dataset
The project uses the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist), which contains 70,000 grayscale images of 28x28 pixels, categorized into 10 classes. The dataset is split into:
- 60,000 training samples
- 10,000 test samples

The dataset is automatically loaded using TensorFlow's `keras.datasets` module.

## Code Overview

1. **Data Preprocessing:**
   - Load the Fashion MNIST dataset.
   - Normalize the image pixel values to the range [0, 1].
   - Convert labels to one-hot encoded format.

2. **Residual Block Function:**
   - Implements a residual block with two convolutional layers and a shortcut connection.

3. **HyperModel Class:**
   - Defines a custom `ResNetModel` class that builds the model with hyperparameter tuning for the number of filters, residual blocks, and learning rate.

4. **KerasTuner:**
   - Uses the `RandomSearch` tuner to explore hyperparameter combinations.
   - Optimizes the model for validation accuracy.

5. **Training and Evaluation:**
   - Trains the model with the best hyperparameters.
   - Evaluates the final model on the test dataset.

## Running the Code

1. Clone the repository or copy the code to your local environment.
2. Ensure all dependencies are installed (see [Requirements](#requirements)).
3. Run the script:

   ```bash
   python resnet.py
   ```

4. The tuning process will output the best hyperparameters and save the results in the specified directory.

## Results
After running the hyperparameter tuning, the best model's performance on the test dataset is evaluated and displayed. The test loss and accuracy are printed as:

```
test loss: 0.3063, test accuracy: 0.8876 # Your results may vary
```

## Acknowledgments
- [TensorFlow](https://www.tensorflow.org/): For providing the framework to build and train deep learning models.
- [KerasTuner](https://keras.io/keras_tuner/): For facilitating hyperparameter optimization.
- [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist): For the dataset used in this project.
