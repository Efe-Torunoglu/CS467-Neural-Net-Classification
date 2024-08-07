import math
from flask import Flask, render_template, request
import numpy as np
import keras
import librosa
import os


SAMPLE_RATE = 22050
GENRE_MODEL_PATH = 'CNN_music_genre_classifier.keras'
GENRE_MAPPINGS = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    GET request renders default home page
    POST request processes audio file and predicts top 3 genres
    Returns: 
        render_template() function to render index.html web page
    """
    if request.method == 'POST':
        file_path = upload_file(request.files['file'])
        if file_path:
            data_array = process_audio_file(file_path)
            top_3_genres = predict_genres(data_array)
            filename = os.path.basename(file_path)
            delete_after_prediction(file_path)
            return render_template('index.html', genres=top_3_genres, filename=filename)
    return render_template('index.html')

def upload_file(uploaded_file):
    """
    Handles file upload and saves the file to the server.
    Args:
        uploaded_file: File object from the request
    Returns:
        file_path: Path to the saved file
    """
    if uploaded_file.filename != '':
        file_path = uploaded_file.filename
        uploaded_file.save(file_path)
        return file_path
    return None

def delete_after_prediction(file_path):
    """
    Deletes the file after prediction is done
    Args:
        file_path: File path to the audio file
    """
    if os.path.exists(file_path):
        os.remove(file_path)

def process_audio_file(file_path, n_mfcc=13, n_fft=2048, hop_length=512, duration_of_clip=3):
    """
    Modified audio processing file function from desktop_predictor.py   
    Args:
        file_path: File path to the audio file       
    Returns: 
        data_array: MFCC information in a data array
    """
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    samples_per_clip = sr * duration_of_clip
    data = {"mfcc": []}
    total_samples = len(signal)
    num_clips = math.ceil(total_samples / samples_per_clip)
    expected_num_mfcc_vectors_per_segment = math.ceil(samples_per_clip / hop_length)

    for clip in range(num_clips):
        start_sample = samples_per_clip * clip
        end_sample = start_sample + samples_per_clip
        if end_sample > total_samples:
            end_sample = total_samples
        current_clip = signal[start_sample:end_sample]
        if len(current_clip) < n_fft:
            pad_width = n_fft - len(current_clip)
            current_clip = np.pad(current_clip, (0, pad_width), mode='constant')
        mfcc = librosa.feature.mfcc(y=current_clip, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T
        if len(mfcc) == expected_num_mfcc_vectors_per_segment:
            data["mfcc"].append(mfcc.tolist())

    data_array = np.array(data["mfcc"])
    return data_array

def predict_genres(data_array):
    """
    Modified predict genres function from desktop_predictor.py
    
    Args:
        data_array: MFCC information in a data array
        
    Returns: 
        top_3_genres: List of 3 lists containing genre and accuracy metric
    """
    model = keras.models.load_model(GENRE_MODEL_PATH)
    mfccs = data_array
    mfccs = mfccs.reshape(mfccs.shape[0], mfccs.shape[1], mfccs.shape[2], 1)
    predictions = model.predict(mfccs)
    mean_softmax_values = np.mean(predictions, axis=0)
    top_3_indices = np.argsort(mean_softmax_values)[-3:][::-1]
    genre_mappings = GENRE_MAPPINGS
    top_3_genres = [[genre_mappings[index], mean_softmax_values[index]] for index in top_3_indices]
    data_array = None  # Reset the data array after prediction
    return top_3_genres


if __name__ == '__main__':
    app.run(debug=True)


# To write this code, we followed the tutorial by Miguel Grinberg:
#    blog.miguelgrinberg.com/post/handling-file-uploads-with-flask
# TO DO: Create audio music player for local web UI
