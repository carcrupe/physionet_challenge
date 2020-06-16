# physionet_challenge

Classification of 12-lead ECGs: the PhysioNet/Computing in Cardiology Challenge 2020
https://physionetchallenges.github.io/2020/

The electrocardiogram (ECG) is a non-invasive representation of the electrical activity of the heart from electrodes placed on the surface of the torso. The standard 12-lead ECG has been widely used to diagnose a variety of cardiac abnormalities such as cardiac arrhythmias, and predicts cardiovascular morbidity and mortality. The early and correct diagnosis of cardiac abnormalities can increase the chances of successful treatments. However, manual interpretation of the electrocardiogram is time-consuming, and requires skilled personnel with a high degree of training.

Automatic detection and classification of cardiac abnormalities can assist physicians in the diagnosis of the growing number of ECGs recorded. Over the last decade, there have been increasing numbers of attempts to stimulate 12-lead ECG classification. Many of these algorithms seem to have the potential for accurate identification of cardiac abnormalities. However, most of these methods have only been tested or developed in single, small, or relatively homogeneous datasets. The PhysioNet/Computing in Cardiology Challenge 2020 provides an opportunity to address this problem by providing data from a wide set of sources.

The goal of the 2020 Challenge is to identify clinical diagnoses from 12-lead ECG recordings.

# EDA and machine learning modeling

To run this project I have used the notebooks from Google Cloud Platform. A Virtual Machine with 4CPUs and 15GB or RAM.

In the uploaded notebook physionet_EDA&models.ipynb, I have followed several steps to train and save the ML model:

- Load the ECG signals and header with information such as sampling rate, age of the subject and diagnostic.
- Extract a few paremeters from the header and with respect to the amplitude and time between R peaks in the ECG signal. I have saved this information as features in a dataframe and the diagnostic as the target for the model.
- I have trained and optimized several models, obtaining in the best case an accuracy for the classification of around 0.4.
- The model is saved as classifier.model
- To make predictions, run the driver.py as follows:
      [python driver.py input_directory output_directory]
  where "input_directory" is a directory for input data files and output_directory is a directory for output of the predictions.
