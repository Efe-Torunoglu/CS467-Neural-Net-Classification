import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt


DATA_PATH = "data.json"


def load_data(data_path):
    '''
    loads a dataset from its path and returns the inputs X  and targets y
    '''
    with open(data_path, "r") as fp:
        data = json.load(fp)
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y


def prepare_datasets(test_size, validation_size):
    # load data
    X, y = load_data(DATA_PATH)
    # create the train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)
    # create the train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=validation_size)  # current shape (130, 13)
    # CNN expects a 3D array for each sample, so we need to add a new dimension to the dataset (130, 13, 1)
    X_train = X_train[..., np.newaxis]  # 4D array (num_samples, 130, 13, 1)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    '''CNN with 3 convolutional layers, max pooling, batch normalization, dropout, and dense layers'''
    '''
    # create the model
    model = keras.Sequential()
    # 1st convolutional layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)) # 32 filters or kernels, 3x3 kernel size
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')) # 3x3 pooling window, stride of 2 vertically and horizontally, 0-padding to keep the same dimensions
    model.add(keras.layers.BatchNormalization()) # normalize the activations of the previous layer at each batch (aids in reliability and speed of training)
    # 2nd convolutional layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)) 
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')) 
    model.add(keras.layers.BatchNormalization()) 
    # 3rd convolutional layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    # flatten the output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu')) # 64 neurons in the dense layer
    model.add(keras.layers.Dropout(0.3)) # 30% dropout rate
    # output layer
    model.add(keras.layers.Dense(10, activation='softmax')) # 10 neurons for 10 genres, softmax activation function
    return model
    '''
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(
        64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(64, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D(
        (2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model


def predict(model, X, y):
    # X is a 3D array (130, 13, 1). The model expects a 4D array (1, 130, 13, 1).
    # add a new dimension to the array so it becomes 4D (1, 130, 13, 1)
    X = X[np.newaxis, ...]
    # prediction will be a 2D array (1, 10) with the probabilities of each genre
    prediction = model.predict(X)
    # extract index with max value (1D array with 1 element with the index of the genre with the highest probability)
    predicted_index = np.argmax(prediction, axis=1)
    print(f"Expected index: {y}, Predicted index: {predicted_index}")


def plot_history(history, test_accuracy, test_loss):
    '''Plot accuracy and loss over epochs, including test set accuracy and loss'''
    plt.figure(figsize=(12, 6))
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.hlines(test_accuracy, 0, len(
        history.history['accuracy'])-1, colors='r', linestyles='dashed')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation', 'Test Final'], loc='upper left')
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.hlines(test_loss, 0, len(
        history.history['loss'])-1, colors='r', linestyles='dashed')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation', 'Test Final'], loc='upper left')
    # Save plots to file
    plt.savefig('training_history_with_test.png')
    # Display the plots
    plt.show()


if __name__ == "__main__":
    # create train, validation, and test sets
    # custom function to split the data (%training set, %validation set, %test set)
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(
        0.25, 0.2)
    # build the CNN model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)
    # compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    '''
    # train the CNN
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=16 , epochs=50) # Hyperparameters: batch size, epochs (can be later tuned)
    '''
    # Early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)
    # Learning rate scheduler
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5)

    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=100,
                        callbacks=[early_stopping, lr_scheduler])

    # evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Set Accuracy: {test_accuracy}")
    # make a prediction on a sample
    X = X_test[100]
    y = y_test[100]
    predict(model, X, y)
    # save the model
    model.save("CNN_music_genre_classifier.keras")
    # plot accuracy and loss over epochs
    plot_history(history, test_accuracy, test_error)

# Run # 1
# Results with Hyperparameters: batch size=32, epochs=30
# -> Adam optimizer - learning rate=0.0001
# -> Batch size=32
# -> Epochs=30
# accuracy: 0.7461 - loss: 0.7137 - val_accuracy: 0.7223 - val_loss: 0.8002
# Test Set Accuracy: 0.7084501385688782

# Run # 2
# Results with Hyperparameters: batch size=16, epochs=50
# -> Adam optimizer - learning rate=0.0001
# -> Batch size=16
# -> Epochs=50
# accuracy: 0.8325 - loss: 0.4835 - val_accuracy: 0.7330 - val_loss: 0.8204
# Test Set Accuracy: 0.731277525424957

# Run # 3
# Increased Convolutional Layer Filters: Increased the number of filters from 32 to 64.
# Dense Layer Units: Increased the number of units in the dense layer from 64 to 128.
# Dropout Rate: Increased the dropout rate to 0.5 to reduce overfitting.
# Learning Rate: Increased the initial learning rate to 0.001.
# Early Stopping: Added early stopping to stop training when validation loss stops improving.
# Learning Rate Scheduler: Added a learning rate scheduler to reduce the learning rate when validation loss stops improving.
# Results with Hyperparameters: batch size=32, epochs=100

# TENGO PENDIENTE CORRER EL MODELO, REVISAR SI EL CODIGO PARA PLOT ESTÁ CORRECTO, SI EL MODELO SE GUARDA CORRECTAMENTE, Y SI LA PREDICCIÓN FUNCIONA CORRECTAMENTE
# COMPLETAR EL CODIGO DE PREDICCIÓN PARA QUE EN VEZ DEL INDICE DIGA EL NOMBRE DEL GÉNERO
