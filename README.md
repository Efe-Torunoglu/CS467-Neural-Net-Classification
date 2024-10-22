# CS467 Neural Net Classification
 
* The dataset used for audio processing corresponds to the folder "GTZAN_dataset"
* The processed dataset (of extracted MFFCs) corresponds to the file "data.json". The code runs a function that looks for corrupt files, reports their file names in "corrupt_files_report.txt", and deletes them, before starting to process the dataset. 
* The first neural network trained with the data on the file "data.json" was a Multilayer Perceptron, which gave a test accuracy of around 0.58 after hyperparemeter tuning. Its code is in the file "nn_music_genre_classifier.py", and its trained weights are stored in the file "music_genre_classifier.keras". The accuracy metrics are stored in the file "training_history.png".
* The second neural network trained with the data on the file "data.json" was a Convolutional Neural Network, which gave a test accuracy of around 0.79 after hyperparemeter tuning. Its code is in the file "cnn_music_genre.py", and its trained weights are stored in the file "CNN_music_genre.keras". The accuracy metrics are stored in the file "training_history_with_test.png".
* The program currently processes an audio file selected by the user (with formats: *.mp3, *.flac, *.wav) from its local disk, processes the audio file, saves it in a json file, and makes a prediction by giving 3 the highest music genre matches with their corresponding softmax values. 
* The program has a desktop UI (using tkinter) that allows users to select an audio clip from their local disk, play the file, stop the file from being played, and make a prediction about its music genre.
* The program also has a web UI (using flask) that allows users to select an audio clip from their local disk and make a prediction about its music genre

# ToDos:
* Upload the web app to a free web hosting service to make it publicly available 
* Improve neural network accuracy of the model used in the app 
* Clean code of code smells and refactor where needed
