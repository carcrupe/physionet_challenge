# Physionet challenge

Classification of 12-lead ECGs: the PhysioNet/Computing in Cardiology Challenge 2020
https://physionetchallenges.github.io/2020/

The electrocardiogram (ECG) is a non-invasive representation of the electrical activity of the heart from electrodes placed on the surface of the torso. The standard 12-lead ECG has been widely used to diagnose a variety of cardiac abnormalities such as cardiac arrhythmias, and predicts cardiovascular morbidity and mortality. The early and correct diagnosis of cardiac abnormalities can increase the chances of successful treatments. However, manual interpretation of the electrocardiogram is time-consuming, and requires skilled personnel with a high degree of training.

Automatic detection and classification of cardiac abnormalities can assist physicians in the diagnosis of the growing number of ECGs recorded. Over the last decade, there have been increasing numbers of attempts to stimulate 12-lead ECG classification. Many of these algorithms seem to have the potential for accurate identification of cardiac abnormalities. However, most of these methods have only been tested or developed in single, small, or relatively homogeneous datasets. The PhysioNet/Computing in Cardiology Challenge 2020 provides an opportunity to address this problem by providing data from a wide set of sources.

The goal of the 2020 Challenge is to identify clinical diagnoses from 12-lead ECG recordings.

# EDA, feature extraction and machine learning modeling

In the uploaded notebook physionet_EDA&models.ipynb, I have followed several steps to train the ML classification model:

1. Load the ECG signals and header with information such as sampling rate, age of the subject and diagnostic.
2. Extract a few paremeters from the header and with respect to the amplitude and time between R peaks in the ECG signal. I have saved this information as features in a dataframe and the diagnostic as the target for the model. Also, applying the FFT, I have stored as features the four highest peaks of the spectrum, to add information about the main frequencies of the signal.
3. I have trained and optimized several models, obtaining in the best case an accuracy for the classification of around 0.51.
4. The model is saved as classifier.model
5. To make predictions using the saved model, run the driver.py as follows: <br /><br />
      python driver.py input_directory output_directory <br /><br />
  where "input_directory" is a directory for input data files and output_directory is a directory for output of the predictions.
6. Evaluation metrics provided at the Physionet website are run to compare the predictions with the original data.
  
  # Conclusions & things to do
  
This is only a first approach to get familiar with the data, do some exploratory analyisis and predictions. ECG signals are really complicated, hard to interpret and, as expected, a simple feature extracion of the R peaks or FFT characteristics is not sufficient to deploy an accurate classification model.

As I mentioned in the notebook, some of the arrythmias are not really related to the R peak of the ECG. Therefore, after optimizing and training the models, I could not achieve an accuracy of more than 0.5. 

As the next step, I will work on extracting more meaningful features from the ECG events, such as position and duration of the P wave, QRS, T wave, ST segment, PR segment, etc. Using these paramenters as features for the training of the model, should give much better results in the prediction of each of the targeted arrythmias.
