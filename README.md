# Detecting and Classifying Animal Calls
Steps to run the Project:
1) Run driver.py to generate features, labels, and train the model. Provide the below parameters:<br>
   a) Base directory containing the dataset (Example: "/cache/rmishra"). The application looks for the audio files in the folder <base-   dir>/cc16_ML (Example: '/cache/rmishra/cc16_ML'). The audio, accelerometer and label data against 5 hyenas can be found in sub-folders 'cc16_352a', 'cc16_352b', 'cc16_354a', 'cc16_360a', 'cc16_366a' respectively under the <base-dir>/cc16_ML directory<br>
    b) Spectrogram Window Size<br>
    c) Spectrogram slide<br>
    d) only_z_axis (boolean) : Denoting if the model has to be run using the combination of "hyena audio + z-axis accelerometer data", or
    "hyena audio + 3-channel accelerometer data"<br>
    
2) Run evaluation/evaluate.py to evaluate the trained model. Provide the below parameters:<br>
  a) The saved model path which has to be evaluated<br>
  b) Dataset path (Example: <base-dir>/dataset)<br>
  c) only_z_axis (boolean) : Denoting if the model has to be evaluated using the combination of "hyena audio + z-axis accelerometer data",   or "hyena audio + 3-channel accelerometer data"<br>
  
