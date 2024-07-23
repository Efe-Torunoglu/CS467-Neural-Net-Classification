import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import sounddevice as sd
import numpy as np
import tensorflow as tf
import math
import librosa


AUDIO_FILE_TYPES = [("Audio Files", "*.mp3 *.wav *.flac")]
DEFAULT_GEOMETRY = "400x200"
SAMPLE_RATE = 22050
GENRE_MODEL_PATH = 'CNN_music_genre_classifier.keras'
GENRE_MAPPINGS = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}


class MusicGenrePredictorApp:
    def __init__(self, master):
        '''Constructor for the MusicGenrePredictorApp class'''
        self.master = master
        self._setup_ui()
        self.file_path = None
        self.data_array = None

    def _setup_ui(self):
        '''Sets up the user interface for the application'''
        # Main window
        self.master.title("Music Genre Predictor")
        self.master.geometry(DEFAULT_GEOMETRY)
        # Label
        self.label = tk.Label(self.master, text="Select an audio file to predict its top 3 music genres.")
        self.label.pack()
        # Select file button
        self.select_file_button = tk.Button(self.master, text="Select File", command=self.select_file)
        self.select_file_button.pack()
        # File name label
        self.file_name_label = tk.Label(self.master, text="No file selected")
        self.file_name_label.pack()
        # Play and stop buttons
        self.play_button = tk.Button(self.master, text="Play Audio", command=self.play_audio)
        self.play_button.pack()
        self.stop_button = tk.Button(self.master, text="Stop Audio", command=self.stop_audio)
        self.stop_button.pack()  
        # Predict button
        self.predict_button = tk.Button(self.master, text="Predict Genres", command=self.display_genres)
        self.predict_button.pack()
        # Quit button
        self.quit_button = tk.Button(self.master, text="Quit", command=self.master.quit)
        self.quit_button.pack()

    def select_file(self):
        '''Selects an audio file using a file dialog and returns the file path'''
        self.file_path = filedialog.askopenfilename(filetypes=AUDIO_FILE_TYPES)
        if self.file_path:
            file_name = self.file_path.split("/")[-1]
            self.file_name_label.config(text=f"Selected file: {file_name}")
            print(f"Selected file: {self.file_path}")
        return self.file_path
            
    def play_audio(self):
        '''Plays the audio file that is currently selected'''
        if self.file_path:
            print(f"Loading and playing: {self.file_path}")
            data, sr = librosa.load(self.file_path, sr=SAMPLE_RATE) 
            sd.play(data, sr)
        else:
            messagebox.showerror("Error", "Please select a file")
            return
            
    def quit(self):
        '''Closes the application'''
        self.master.destroy()
        
    def stop_audio(self):
        '''Stops the audio that is currently playing'''
        sd.stop()
        print("Audio stopped.")    
        
    def process_audio_file(self, n_mfcc=13, n_fft=2048, hop_length=512, duration_of_clip=3):
        '''Processes the audio file to extract the MFCCs and returns them in the data array'''
        if not self.file_path:
            self.file_path = self.select_file()
        signal, sr = librosa.load(self.file_path, sr=SAMPLE_RATE)
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
        
        self.data_array = np.array(data["mfcc"])
        return self.data_array

    def predict_genres(self):
        '''Predicts the top 3 music genres for the audio file'''
        if not self.file_path:
            messagebox.showerror("Error", "Please select a file")
            return
        model = tf.keras.models.load_model(GENRE_MODEL_PATH)
        mfccs = self.data_array if self.data_array is not None else self.process_audio_file()
        mfccs = mfccs.reshape(mfccs.shape[0], mfccs.shape[1], mfccs.shape[2], 1)
        predictions = model.predict(mfccs)
        mean_softmax_values = np.mean(predictions, axis=0)
        top_3_indices = np.argsort(mean_softmax_values)[-3:][::-1]
        genre_mappings = GENRE_MAPPINGS
        top_3_genres = [[genre_mappings[index], mean_softmax_values[index]] for index in top_3_indices]
        self.data_array = None  # Reset the data array after prediction
        return top_3_genres
    
    def display_genres(self):
        '''Displays the top 3 music genres for the audio file in a message box and returns the top-3 genres'''
        top_3_genres = self.predict_genres()
        if not self.file_path:
            return
        file_name = self.file_path.split("/")[-1]
        self.file_name_label.config(text=f"Selected file: {file_name}")
        message = "Top 3 Genres for: " + file_name + "\n" 
        for genre, softmax_value in top_3_genres:
            message += f"{genre}: {softmax_value:.2f}\n"
        messagebox.showinfo("Predicted Genres", message)
        self.file_name_label.config(text=f"No file selected")
        self.file_path = None
        return top_3_genres


def main():
    root = tk.Tk()
    app = MusicGenrePredictorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()