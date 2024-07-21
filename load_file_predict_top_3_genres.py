import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
import sounddevice as sd
import numpy as np
import librosa
import math
import json
import os


# Load the pre-trained model
model = tf.keras.models.load_model('CNN_music_genre_classifier.keras')


def select_audio_file():
    # Initialize Tkinter root
    root = tk.Tk()
    root.withdraw()  # We don't want a full GUI, so keep the root window from appearing

    # Show an "Open" dialog box and return the path to the selected file
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav *.flac")])
    print(f"Selected file: {file_path}")
    return file_path


def play_audio_file(file_path):
    if file_path:
        print(f"Loading and playing: {file_path}")
        # Load the audio file
        data, sr = librosa.load(file_path, sr=None)  # sr=None to preserve the original sampling rate
        # Play the audio
        sd.play(data, sr)
        # Wait for the audio to finish playing
        sd.wait()
    else:
        print("No file selected or provided.")


def process_audio_file(file_path, n_mfcc=13, n_fft=2048, hop_length=512, duration_of_clip=3):
    signal, sr = librosa.load(file_path, sr=22050)
    samples_per_clip = sr * duration_of_clip
   
    data = {"mfcc": []}
    total_samples = len(signal)
    num_clips = math.ceil(total_samples / samples_per_clip)
    # print(f"debugger -> len(signal): {total_samples}")
    # print(f"debugger -> samples per clip: {samples_per_clip}")
    # print(f"debugger -> sampling rate: {sr}")
    # print(f"debugger -> duration of track: {len(signal) / sr}")
    # Calculate the expected number of MFCC vectors per segment
    # This calculation assumes the audio is evenly divided into segments of `duration_of_clip` seconds
    expected_num_mfcc_vectors_per_segment = math.ceil(samples_per_clip / hop_length) 
    
    for clip in range(num_clips):
        start_sample = samples_per_clip * clip
        end_sample = start_sample + samples_per_clip
        
        # Verify the end sample is within the signal
        if end_sample > total_samples:
            end_sample = total_samples 
            
        # Extract the current clip from the signal
        current_clip = signal[start_sample:end_sample]
        
        # Check if the current clip needs padding
        if len(current_clip) < n_fft:
            # Pad the current clip
            pad_width = n_fft - len(current_clip)
            current_clip = np.pad(current_clip, (0, pad_width), mode='constant')       
            
        # Compute the MFCCs for the current clip
        mfcc = librosa.feature.mfcc(y=current_clip, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T
        
        # Store the MFCCs after validation
        if len(mfcc) == expected_num_mfcc_vectors_per_segment:
            data["mfcc"].append(mfcc.tolist())
            
    # Create the path for the JSON file in the same directory as this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(current_dir, "clip_data.json")
        
    # Save the data to the JSON file
    try:
        with open(json_file_path, "w") as fp:
            json.dump(data, fp, indent=4)
        print(f"Data successfully saved to {json_file_path}")
    except IOError as e:
        print(f"Error saving data to {json_file_path}: {e}")
    
    return np.array(data["mfcc"])


def predict_genre(model, data_array):
    # select the json file
    mfccs = data_array
    mfccs = mfccs.reshape(mfccs.shape[0], mfccs.shape[1], mfccs.shape[2], 1)  # Add channel dimension for Conv2D
    print(f"Dimensions of 'mfccs' after reshaping: {mfccs.shape}")  # This line prints the dimensions
    
    # Make predictions for each segment
    predictions = model.predict(mfccs)
    
    # Calculate the mean softmax values for each genre across all predictions
    mean_softmax_values = np.mean(predictions, axis=0)
  
    # Find the indices of the 3 highest mean softmax values
    # get the last 3 values sorted in ascending order and then reverse the order (to descending order)
    top_3_indices = np.argsort(mean_softmax_values)[-3:][::-1]  
    
    # Load genre mappings
    genre_mappings = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}

    # Prepare the top 3 genres and their softmax values in a 2D array (list of lists)
    top_3_genres = [[genre_mappings[index], mean_softmax_values[index]] for index in top_3_indices]
    
    # # Print the top 3 genres with the highest mean softmax values
    # print("Top 3 Predicted Genres with highest mean softmax values:")
    # for index in top_3_indices:
    #     genre_name = genre_mappings[index]
    #     softmax_value = mean_softmax_values[index]
    #     print(f"{genre_name}: {softmax_value}")    
    
    return top_3_genres


if __name__ == "__main__":
    file_path = select_audio_file()
    # Call the function to process the audio file and save the MFCCs to the JSON file
    data_array = process_audio_file(file_path)
    print(f"Dimensions of 'mfccs' before reshaping: {data_array.shape}")  # This line prints the dimensions
    top_n_array = predict_genre(model, data_array)
    # Print the top 3 genres and their softmax values
    print("Top 3 Predicted Genres with their softmax values:")
    for genre, softmax_value in top_n_array:
        print(f"{genre}: {softmax_value}")
