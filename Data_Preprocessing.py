#Author: Dan Griffith 101111722
#Date: November 6, 2023
#Version: 1.0 

import librosa
import os
import matplotlib.pyplot as plt 
import librosa.display


#Converts the audio file into MFCCs
def gen_MFCCS(filename, show):
    

    x, sr = librosa.load(filename)
    mfcc = librosa.feature.mfcc(y=x, sr=sr)
    mfcc = mfcc.T
    print(mfcc.shape)
 

    #Input a value of true for the MFCCs to be shown each time.
    if  show:
        librosa.display.specshow(mfcc, sr=sr,x_axis='time')
        plt.show()
    print(mfcc)
    return mfcc

#stores the MFCCs with their corresponding labels into an object and then returns it.
#The argument data_folder is used as "raw-data/animal_name"

def main(data_folder):
    
    label = data_folder.split('/')[1]
    file_list = os.listdir(data_folder)

    data = {
        "MFCCs":[],
        "Label":[]
    }
    
    for filename in file_list:
       
        MFCC = gen_MFCCS(data_folder+ '/'+filename, False)

        data['Label'].append(label)
        data['MFCCs'].append(MFCC)

    return data



data_path = 'raw-data/KillerWhale'
main(data_path)