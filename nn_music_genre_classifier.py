import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt


# If this message occurs at the start of running the code, it is due to the use of oneDNN custom operations in TensorFlow: 
#  I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. 
#  You may see slightly different numerical results due to floating-point 
#  round-off errors from different computation orders. 
# To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# write this command on Powershell: $env:TF_ENABLE_ONEDNN_OPTS="0"

DATASET_PATH = "data.json"


def load_data(dataset_path):
    # load the dataset
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
    # convert lists to numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])
    return inputs, targets

def plot_history(history):
    '''Plot accuracy and loss over epochs'''
    fig, axs = plt.subplots(2)
    # accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")
    # loss subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error evaluation")
    plt.tight_layout()
    plt.savefig("training_history.png")  # Save the figure as an image
    plt.show()
        

if __name__ == "__main__":
    # load the data
    inputs, targets = load_data(DATASET_PATH)
    print(inputs.shape)
    print(targets.shape)

    # split the data into 70% training and 30% testing datasets
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs, targets, test_size=0.3)
    
    inputs_shape = (inputs.shape[1], inputs.shape[2])
    
    # build the neural network architecture - Multi Layer Perceptron (MLP) with 3 hidden layers 
    # -> Dropout and Regularization have been incorporated to address overfitting
    model = keras.Sequential([
        # input layer  
        keras.layers.Input(shape=inputs_shape),        
        # flatten the inputs from a 3D array into a 1D array (index 0 represents different audio segments)
        keras.layers.Flatten(),
        # 1st hidden layer (simple dense layer with 512 neurons and relu activation function) 
        keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        # 2nd hidden layer (simple dense layer with 256 neurons and relu activation function)
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        # 3rd hidden layer (simple dense layer with 64 neurons and relu activation function)
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        # output layer (10 neurons for each genre and softmax activation function for probability distribution of each genre)
        keras.layers.Dense(10, activation='softmax')
    ])

    # compile the neural network
    # Adam optimizer with learning rate of 0.0001
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    # sparse_categorical_crossentropy loss function for multi-class classification
    # accuracy metric to evaluate the model
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # train the neural network using mini-batch to calculate the gradient descent, and store the loss and accuracy of the model
    history = model.fit(inputs_train, targets_train, validation_data=(
        inputs_test, targets_test), epochs=100, batch_size=32)
    
    # plot accuracy and loss over epochs
    plot = plot_history(history)

    # Though the results vary every time the model is trained, we identify that when initially running the model:
    # Test accuracy is around 60% and Test loss is around 1.5 while training accuracy is around 80% and training loss is around 0.5
    # This big difference is clearly indicative of overfitting, as the model is performing well on the training data but not on the test data
    # Ways to address overfitting:
    # Simpler architecture: Reduce the number of layers or neurons in the network
    # Data augmentation: Increase the size of the training dataset by applying transformations to the input data
    # Early stopping: Monitor the validation loss during training and stop when it starts to increase
    # Batch normalization: Normalize the activations of the network to make training more stable
    # Dropout: Randomly set a fraction of input units to 0 at each update during training time, which helps prevent overfitting
    # Regularization: Add a penalty term to the loss function to prevent the weights from becoming too large (L1/L2 regularization)
    
    
    # Solving Overfitting:
    # To address overfitting, improving test accuracy and reducing test loss, we will tune Hyperparameters:
    # Dropout probability = 0.1 - 0.5
    # Regularization parameter = 0.001 - 0.01 (L2 regularization for audio data) 
    # This will be incoroporated in the model architecture and training process
    # After 100 epochs and applying regularization and dropout we have:
    # test accuracy 0.56 and test loss 1.63
    # After 200 epochs and applying regularization and dropout we have:
    
    #Though the model is trained, we can save the model to a file for later use
    model.save("music_genre_classifier_MLP.keras")
    print("Model saved to disk")
    
    # evaluate the model on the test dataset
    test_loss, test_accuracy = model.evaluate(inputs_test, targets_test)
    print(f"Test accuracy: {test_accuracy}")
    print(f"Test loss: {test_loss}")
    
    # make predictions on a sample
    sample = inputs_test[0]
    sample = np.expand_dims(sample, axis=0)
    prediction = model.predict(sample)
    print(f"Model prediction: {np.argmax(prediction)}")
    print(f"Actual target: {targets_test[0]}")