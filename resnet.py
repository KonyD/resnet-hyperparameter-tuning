import tensorflow as tf

# Importing required modules from TensorFlow and Keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, Flatten, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Importing HyperModel and RandomSearch for hyperparameter tuning from KerasTuner
from kerastuner import HyperModel, RandomSearch

# Loading the Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Reshaping and normalizing the image data
train_images = train_images.reshape(-1, 28, 28, 1).astype("float32") / 255.0
test_images = test_images.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Converting labels to one-hot encoded format
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Function to create a residual block
def residual_block(x, filters, kernel_size=3, strides=1):
    shortcut = x  # Save the input tensor to add later as a shortcut connection
    
    # First convolutional layer
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding="same")(x)
    x = BatchNormalization()(x)  # Normalize the output
    x = Activation("relu")(x)  # Apply ReLU activation
    
    # Second convolutional layer
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding="same")(x)
    x = BatchNormalization()(x)  # Normalize the output again
    
    # Adjust the shortcut if the dimensions differ
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, kernel_size=1, strides=strides, padding="same")(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    # Add the shortcut connection and apply ReLU activation
    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    
    return x

# Custom HyperModel class for building the ResNet model
class ResNetModel(HyperModel):
    
    def build(self, hp):
        inputs = Input(shape=(28, 28, 1))  # Input layer with shape for Fashion MNIST data
        
        # Initial convolutional layer with hyperparameter tuning for the number of filters
        x = Conv2D(filters=hp.Int("initial_filters", min_value=32, max_value=128, step=32),
                   kernel_size=3, padding="same", activation="relu")(inputs)
        x = BatchNormalization()(x)  # Normalize the output
        
        # Adding residual blocks based on the number specified by the hyperparameter
        for i in range(hp.Int("num_blocks", min_value=1, max_value=3, step=1)):
            x = residual_block(x, hp.Int("res_filters" + str(i), min_value=32, max_value=128, step=32))
        
        # Flatten the output for the dense layers
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)  # Dense layer with 128 units
        outputs = Dense(10, activation="softmax")(x)  # Output layer for 10 classes
        
        # Define the model
        model = Model(inputs, outputs)
        
        # Compile the model with Adam optimizer and hyperparameter tuning for learning rate
        model.compile(optimizer=Adam(hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG")),
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        return model

# Initializing the RandomSearch tuner
tuner = RandomSearch(
    ResNetModel(),
    objective="val_accuracy",  # Optimize for validation accuracy
    max_trials=2,  # Number of trials to perform
    executions_per_trial=1,  # Number of executions per trial
    directory="resnet_hyperparameter_tuning_directory",  # Directory for saving tuning results
    project_name="resnet_model_tuning"  # Name of the tuning project
)

# Start the hyperparameter tuning process
tuner.search(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))

# Retrieve the best model after tuning
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the best model on the test set
test_loss, test_accuracy = best_model.evaluate(test_images, test_labels)

# Print the test loss and accuracy
print(f"test loss: {test_loss:.4f}, test accuracy: {test_accuracy:.4f}")
