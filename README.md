# 🐟 Project Dory


<br>

# 🎯 Background & Objective
<p>
For decades, researchers have been collecting sound data from the Earth's oceans without an efficient way to classify it. 
NOAA (National Oceanic and Atmospheric Administration) and the U.S NAVY have collected over 300 terabyts of oceanic sound data over a 4 year period through the SanctSound project 
which aims to allow people to easily explore and access as much of this data as possible. Classifying this data is already happening thanks to the NOAA Big Data Program. 
Project Dory hopes to build on this effort by providing a platform that is able to listen to and identify sounds created by marine mammals. 
Inspired by Dr. Roger Payne’s multi-platinum album ‘Songs of The Humpback Whale’, we hope this project will be used in the effort to preserve and study marine life.
    
The objective of Project Dory is to develop a platform to aid researchers in classifying marine mammals from sound data. 
We aim to develop an AI model that is able to classify the presence of one or multiple marine mammals inside a sound clip. 
The platform will be able to take an audio file (.wav) and output what animals are present with a time stamp corresponding to the original audio file.
</p>
<br>

# 🛠️ Programming & Tech
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
<br>

## Methods
- Project Dory will utilize deep learning techniques, such as convolutional neural networks (CNNs), and classify data using The Mel-frequency cepstral coefficients (MFCCs) to classify the features of the audio data based on time and frequency. 
The AI model will be trained using a supervised learning approach, where it will be trained on labeled marine mammal sound data.
<br>

## Data Set
- Project Dory plans on using the Watkins Marine Mammal Sound Database. 
This database contains the labeled recordings of over 60 different marine mammals collected over 7 decades.
<br>

## Validation & Analysis
- Project Dory will validate its findings through the introduction of new data not seen by the model. 
We will perform loss validation calculations to see how well our model is able to predict the marine mammals present in the audio clip. 
Once our model archives reasonable performance levels, we will continue with the validation of labeled data by comparing our ground truth labels with the outputs from our model.
<br>

    os
    librosa 
    matplotlib
    
## Data_Preprocessing

This file contains all the functions needed to process the sound data. The goal is to take a .WAV file and output a corresponding MFCC picture. 

## Functions
    
    gen_MFCCS(filename, show) - This function takes the .WAV file and converts it to a MFCC

        - filename: the name of the file
        - show: a boolean value indicating whether or not you want to display the MFCC

    main(data_folder) - This is the main function that iterates over the data folder and returns the processed data.

        - data object: has two attributes:
            - label: A list of labels(what animal, should be the same things per folder).
            - MFCC: A list of MFCCs
        - data_folder: Subfolder in raw data folder. String should look like this: 'raw-data/animal_name'

## Input: 
    .WAV sound file or folder. 

## Processing:
    1. Pre-emphasis filter to boost higher frequencies to reduce noise and improve detection.
    2. Windowing allows for us to slice the audio clips into smaller parts without introducing additional noise.
       - Will use the Hamming or Hanning method to slice depending on the performance.
       - x[n] is the sliced frame, s[n] is the original frame, w[n] is the weight, and L is the window width. 
       - w[n] = (1-a) - a cos (2pi*n/L-1) Hamming a = 0.46164, Hanning a = 0.5
       - x[n] = w[n] * s[n]
    3. Discrete Fourier Transformation (DFT) is used to extract the frequency data. (Add equation)
    4. Mel filterbank maps the measured frequency to that we perceived in the context of frequency resolution.
        - We square the DFT output. This is (x[k]^2) and is called the DFT power spectrum. 
        - k is the DFT binary number
        - m is the mel-filter bank number. 
    5. Taking the log of filterbank output to reduce acoustic variants that are not significant for sound recognition.
    6. Cepstrum - IDFT is used to separate the glottal source and the filter.
    7. DCT is used to remove the features that do not correlate. This makes our model more efficient. 
 
        - This will output 12 Cepstrum coefficients. 
    8. Dynamic Features (delta) are the characterizing features over time by calculating the energy (13th feature) and 
       d(t) which is the change in features from one frame to the next. This is the first derivative of the features.
    9. Cepstral mean and variance normalization is used to countermeasure the variants in each recording.
      

## Considerations: 
    - MFCC is influenced by noise. We may want to use PLP instead. This uses cube-root compression instead of log compression.
      It also uses linear regression to finalize cepstral coefficients. 
    - PLP is said to be more robust against noise and a better performance than MFCC depending on the task. 
    - We may want to look into using PLP if MFCC is not working as expected.


## Output:
    The extracted features from the MFCC as an object. See main() function description for details.
