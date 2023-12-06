import os
import librosa
import numpy as np
import csv

def convert_to_mfcc(file_path, max_length):
    # Load the audio file
    y, sr = librosa.load(file_path)
    
    # Compute MFCCs from the audio
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    
    # If the number of frames is less than the max length, pad with zeros
    if mfcc.shape[1] < max_length:
        pad_width = max_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # If the number of frames is more than the max length, truncate the excess
    elif mfcc.shape[1] > max_length:
        mfcc = mfcc[:, :max_length] 
    
    return mfcc

def process_directory(directory_path, max_length):
    mfccs = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(directory_path, filename)
            mfcc = convert_to_mfcc(file_path, max_length)
            #print(mfcc.shape)
            mfccs[filename] = mfcc
    return mfccs

def write_to_csv(data_list, output_csv):
    with open(output_csv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        for data in data_list:
         
            csvwriter.writerow(data)

# Directory containing the .wav files
def main(filePath, label, filename):
    directory_path = filePath

    # Define the constant size for the MFCC (number of frames)
    max_length = 130

    # Process the directory
    mfccs = process_directory(directory_path, max_length)
    flattened_v = []

    counter = 0

    for index in mfccs:
        vector = mfccs[index].flatten()
        flattened_v.append(vector)
        if counter > 500:
            break
        counter += 1
    X = np.vstack(flattened_v)
    (n, d) = X.shape
    column_vector = np.ones((n,1))
    print(X.shape)
    print(column_vector.shape)
    X = np.concatenate((column_vector*label, X), axis=1)
    print(X.shape)

    write_to_csv(X, filename)

def create_test_point(filename):

    # Define the constant size for the MFCC (number of frames)
    max_length = 130

    # Process the directory
    mfcc = convert_to_mfcc(filename, max_length)
    X = mfcc.flatten()
    X = X.reshape(1, -1)
  
    with open('test.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
 
        csvwriter.writerow(X)

    
# main('Data/data/KillerWhale', 1, 'output.csv')
# main('Data/data/SpermWhale', 2,'output.csv')
# main('Data/data/BottlenoseDolphin', 3,'output.csv')
# main('Data/data/AtlanticSpottedDolphin', 4,'output.csv')    
# main('Data/data/SpinnerDolphin', 5,'output.csv')  
# main('Data/data/BowheadWhale', 6,'output.csv') 