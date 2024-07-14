import os
import librosa
import math
import json

DATASET_PATH = "GTZAN_dataset"
JSON_PATH = "data.json"
SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# Preparing the data - doing preprocessing on the dataset
# extract the inputs and the targets - basically the labels and the MFCCs from the music dataset
# and then store that in a JSON file so that we can use it when we train the neural network
# The dataset we will use to train our model is the GTZAN dataset

# define function that will save the MFCCs and the labels in a JSON file
# we will use for the number of MFCCs 13, the number of FFTs 2048, the hop length 512 and the number of segments 5
# segmenting each audio file will allow us to have more data to train our model on

# Before start processing the dataset, we need to remove any corrupt files that may exist in the dataset


def delete_corrupt_files(dataset_path, report_path="corrupt_files_report.txt"):
    with open(report_path, "w") as report_file:
        for dirpath, dirnames, filenames in os.walk(dataset_path):
            if dirpath != dataset_path:
                for f in filenames:
                    file_path = os.path.join(dirpath, f)
                    try:
                        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                    except Exception as e:
                        report_file.write(f'{file_path}\n')
                        os.remove(file_path)
                        print(f'{file_path} removed')
                        continue
                    print(f'Processed: {file_path}')
    print('All corrupt files removed and reported')


def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    # dictionary to store the data
    data = {"mapping": [], "mfcc": [], "labels": []}
    num_samples_per_segment = int(SAMPLES_PER_TRACK // num_segments)
    # To know the overall number of mfcc vectors per segment (we are calculating the mfccs at each hop length):
    expected_num_mfcc_vectors_per_segment = math.ceil(
        num_samples_per_segment / hop_length)  # rounded to the nearest greater int
    # loop through all the genres in the dataset
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # ensure we are not at the root level
        if dirpath is not dataset_path:
            # Extract genre label from the directory path
            # This line is modified to extract only the genre name
            genre_label = os.path.basename(dirpath)
            data["mapping"].append(genre_label)
            print(f'Processing {genre_label}')
            # process files for a specific genre
            for f in filenames:
                # load the audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                # process segments extracting MFCC and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    mfcc = librosa.feature.mfcc(
                        y=signal[start_sample:finish_sample], sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T
                    # store mfcc for segment if it has the expected length
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        # we convert the numpy array to a list so we can save it as JSON
                        data["mfcc"].append(mfcc.tolist())
                        # the first iteration was the root folder, so we subtract 1 to get the correct index of the genre
                        data["labels"].append(i-1)
                        print(f'{file_path}, segment: {s}')

    # save the data in a JSON file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
    print(f'\nData saved to {json_path}')


if __name__ == '__main__':
    # delete any corrupt files in the dataset and report them in a text file
    delete_corrupt_files(DATASET_PATH)

    # This code will save the MFCCs and the labels in a JSON file
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)


''' DOCUMENTATION:
 
This code preprocesses the GTZAN dataset for music genre classification and saves the extracted features (MFCCs) and labels (genres) into a JSON file. The file performs the following tasks:

Delete Corrupt Files: The delete_corrupt_files function walks through the dataset directory, attempts to load each file with librosa, and removes any files that cause an exception, logging their paths to a report file. This helps ensure data integrity.

Save MFCCs and Labels to JSON: The save_mfcc function is designed to process the audio files, extract MFCCs (Mel Frequency Cepstral Coefficients) as features, and save these features along with their corresponding labels (genres) into a JSON file. This process includes:

Creating a data dictionary with keys for "mapping" (genre labels), "mfcc" (features), and "labels" (numeric indices corresponding to genres).
Walking through the dataset directory, skipping the root, and processing files within each genre subdirectory.
For each audio file, it segments the file into smaller portions (based on num_segments), extracts MFCCs for each segment, and appends this data along with the genre label index to the data dictionary.
After processing all files, it saves the data dictionary to a JSON file specified by json_path.
Parameters for MFCC Extraction: The function commonly used values for parameters like n_mfcc=13, n_fft=2048, and hop_length=512 for MFCC extraction. The number of segments (num_segments) can be adjusted to increase or decrease the amount of data generated from each track.

Genre Label Extraction and Mapping: The genre label for each track is extracted from the directory name and added to the "mapping" list in the data dictionary. This mapping is used to convert genre labels into numeric indices, which are stored in the "labels" list alongside the corresponding MFCCs in the "mfcc" list.

Error Handling and Reporting: The script includes basic error handling for file processing and reports corrupt files. It does not handle errors during the MFCC extraction process or JSON file writing, which could be improved in a future release.

Execution: The script is executed by first removing corrupt files and then saving the MFCCs and labels to a JSON file. The num_segments parameter is set to 10 in the main block, which overrides the default value of 5 specified in the function definition. This means each track will be divided into 10 segments for MFCC extraction, providing more data for training.

'''


# To write this code, we followed the youtube tutorial by Valerio Velardo: https://www.youtube.com/watch?v=szyGiObZymo
