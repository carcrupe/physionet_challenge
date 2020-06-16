#!/usr/bin/env python

import numpy as np
from scipy.signal import butter, lfilter
from scipy import stats

def detect_peaks(ecg_measurements,signal_frequency,gain):

        """
        Method responsible for extracting peaks from loaded ECG measurements data through measurements processing.

        This implementation of a QRS Complex Detector is by no means a certified medical tool and should not be used in health monitoring. 
        It was created and used for experimental purposes in psychophysiology and psychology.
        You can find more information in module documentation:
        https://github.com/c-labpl/qrs_detector
        If you use these modules in a research project, please consider citing it:
        https://zenodo.org/record/583770
        If you use these modules in any other project, please refer to MIT open-source license.

        If you have any question on the implementation, please refer to:

        Michal Sznajder (Jagiellonian University) - technical contact (msznajder@gmail.com)
        Marta lukowska (Jagiellonian University)
        Janko Slavic peak detection algorithm and implementation.
        https://github.com/c-labpl/qrs_detector
        https://github.com/jankoslavic/py-tools/tree/master/findpeaks
        
        MIT License
        Copyright (c) 2017 Michal Sznajder, Marta Lukowska
    
        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:
        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.
        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.

        """


        filter_lowcut = 0.001
        filter_highcut = 15.0
        filter_order = 1
        integration_window = 30  # Change proportionally when adjusting frequency (in samples).
        findpeaks_limit = 0.35
        findpeaks_spacing = 100  # Change proportionally when adjusting frequency (in samples).
        refractory_period = 240  # Change proportionally when adjusting frequency (in samples).
        qrs_peak_filtering_factor = 0.125
        noise_peak_filtering_factor = 0.125
        qrs_noise_diff_weight = 0.25


        # Detection results.
        qrs_peaks_indices = np.array([], dtype=int)
        noise_peaks_indices = np.array([], dtype=int)


        # Measurements filtering - 0-15 Hz band pass filter.
        filtered_ecg_measurements = bandpass_filter(ecg_measurements, lowcut=filter_lowcut, highcut=filter_highcut, signal_freq=signal_frequency, filter_order=filter_order)

        filtered_ecg_measurements[:5] = filtered_ecg_measurements[5]

        # Derivative - provides QRS slope information.
        differentiated_ecg_measurements = np.ediff1d(filtered_ecg_measurements)

        # Squaring - intensifies values received in derivative.
        squared_ecg_measurements = differentiated_ecg_measurements ** 2

        # Moving-window integration.
        integrated_ecg_measurements = np.convolve(squared_ecg_measurements, np.ones(integration_window)/integration_window)

        # Fiducial mark - peak detection on integrated measurements.
        detected_peaks_indices = findpeaks(data=integrated_ecg_measurements,
                                                     limit=findpeaks_limit,
                                                     spacing=findpeaks_spacing)

        detected_peaks_values = integrated_ecg_measurements[detected_peaks_indices]

        return detected_peaks_values,detected_peaks_indices

 
def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
        """
        Method responsible for creating and applying Butterworth filter.
        :param deque data: raw data
        :param float lowcut: filter lowcut frequency value
        :param float highcut: filter highcut frequency value
        :param int signal_freq: signal frequency in samples per second (Hz)
        :param int filter_order: filter order
        :return array: filtered data
        """
        nyquist_freq = 0.5 * signal_freq
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq
        b, a = butter(filter_order, [low, high], btype="band")
        y = lfilter(b, a, data)
        return y

def findpeaks(data, spacing=1, limit=None):
        """
        Janko Slavic peak detection algorithm and implementation.
        https://github.com/jankoslavic/py-tools/tree/master/findpeaks
        Finds peaks in `data` which are of `spacing` width and >=`limit`.
        :param ndarray data: data
        :param float spacing: minimum spacing to the next peak (should be 1 or more)
        :param float limit: peaks should have value greater or equal
        :return array: detected peaks indexes array
        """
        len = data.size
        x = np.zeros(len + 2 * spacing)
        x[:spacing] = data[0] - 1.e-6
        x[-spacing:] = data[-1] - 1.e-6
        x[spacing:spacing + len] = data
        peak_candidate = np.zeros(len)
        peak_candidate[:] = True
        for s in range(spacing):
            start = spacing - s - 1
            h_b = x[start: start + len]  # before
            start = spacing
            h_c = x[start: start + len]  # central
            start = spacing + s + 1
            h_a = x[start: start + len]  # after
            peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

        ind = np.argwhere(peak_candidate)
        ind = ind.reshape(ind.size)
        if limit is not None:
            ind = ind[data[ind] > limit]
        return ind


def get_12ECG_features(data, header_data):

    tmp_hea = header_data[0].split(' ')
    ptID = tmp_hea[0]
    num_leads = int(tmp_hea[1])
    sample_Fs= int(tmp_hea[2])
    gain_lead = np.zeros(num_leads)
    
    for ii in range(num_leads):
        tmp_hea = header_data[ii+1].split(' ')
        gain_lead[ii] = int(tmp_hea[2].split('/')[0])

    # for testing, we included the mean age of 57 if the age is a NaN
    # This value will change as more data is being released
    for iline in header_data:
        if iline.startswith('#Age'):
            tmp_age = iline.split(': ')[1].strip()
            age = int(tmp_age if tmp_age != 'NaN' else 57)
        elif iline.startswith('#Sex'):
            tmp_sex = iline.split(': ')[1]
            if tmp_sex.strip()=='Female':
                sex =1
            else:
                sex=0
        elif iline.startswith('#Dx'):
            label = iline.split(': ')[1].split(',')[0]


    
#   We are only using data from lead1
    peaks,idx = detect_peaks(data[0],sample_Fs,gain_lead[0])
   
#   mean
    mean_RR = np.mean(idx/sample_Fs*1000)
    mean_Peaks = np.mean(peaks*gain_lead[0])

#   median
    median_RR = np.median(idx/sample_Fs*1000)
    median_Peaks = np.median(peaks*gain_lead[0])

#   standard deviation
    std_RR = np.std(idx/sample_Fs*1000)
    std_Peaks = np.std(peaks*gain_lead[0])

#   variance
    var_RR = stats.tvar(idx/sample_Fs*1000)
    var_Peaks = stats.tvar(peaks*gain_lead[0])

#   Skewness
    skew_RR = stats.skew(idx/sample_Fs*1000)
    skew_Peaks = stats.skew(peaks*gain_lead[0])

#   Kurtosis
    kurt_RR = stats.kurtosis(idx/sample_Fs*1000)
    kurt_Peaks = stats.kurtosis(peaks*gain_lead[0])

    features = np.hstack([age,sex,mean_RR,mean_Peaks,median_RR,median_Peaks,std_RR,std_Peaks,var_RR,var_Peaks,skew_RR,skew_Peaks,kurt_RR,kurt_Peaks])

  
    return features



def ecg_wave_detector(ecg, rpeaks):
    """
    Returns the localization of the P, Q, T waves. This function needs massive help!
    https://neurokit.readthedocs.io/en/latest/_modules/neurokit/bio/bio_ecg_preprocessing.html

    Parameters
    ----------
    ecg : list or ndarray
        ECG signal (preferably filtered).
    rpeaks : list or ndarray
        R peaks localization.

    Returns
    ----------
    ecg_waves : dict
        Contains wave peaks location indices.

    Example
    ----------
    >>> import neurokit as nk
    >>> ecg =  nk.ecg_simulate(duration=5, sampling_rate=1000)
    >>> ecg = nk.ecg_preprocess(ecg=ecg, sampling_rate=1000)
    >>> rpeaks = ecg["ECG"]["R_Peaks"]
    >>> ecg = ecg["df"]["ECG_Filtered"]
    >>> ecg_waves = nk.ecg_wave_detector(ecg=ecg, rpeaks=rpeaks)
    >>> nk.plot_events_in_signal(ecg, [ecg_waves["P_Waves"], ecg_waves["Q_Waves_Onsets"], ecg_waves["Q_Waves"], list(rpeaks), ecg_waves["S_Waves"], ecg_waves["T_Waves_Onsets"], ecg_waves["T_Waves"], ecg_waves["T_Waves_Ends"]], color=["green", "yellow", "orange", "red", "black", "brown", "blue", "purple"])

    Notes
    ----------
    *Details*

    - **Cardiac Cycle**: A typical ECG showing a heartbeat consists of a P wave, a QRS complex and a T wave.The P wave represents the wave of depolarization that spreads from the SA-node throughout the atria. The QRS complex reflects the rapid depolarization of the right and left ventricles. Since the ventricles are the largest part of the heart, in terms of mass, the QRS complex usually has a much larger amplitude than the P-wave. The T wave represents the ventricular repolarization of the ventricles. On rare occasions, a U wave can be seen following the T wave. The U wave is believed to be related to the last remnants of ventricular repolarization.

    *Authors*

    - `Dominique Makowski <https://dominiquemakowski.github.io/>`_
    """
    q_waves = []
    p_waves = []
    q_waves_starts = []
    s_waves = []
    t_waves = []
    t_waves_starts = []
    t_waves_ends = []
    for index, rpeak in enumerate(rpeaks[:-3]):

        try:
            epoch_before = np.array(ecg)[int(rpeaks[index-1]):int(rpeak)]
            epoch_before = epoch_before[int(len(epoch_before)/2):len(epoch_before)]
            epoch_before = list(reversed(epoch_before))

            q_wave_index = np.min(find_peaks(epoch_before))
            q_wave = rpeak - q_wave_index
            p_wave_index = q_wave_index + np.argmax(epoch_before[q_wave_index:])
            p_wave = rpeak - p_wave_index

            inter_pq = epoch_before[q_wave_index:p_wave_index]
            inter_pq_derivative = np.gradient(inter_pq, 2)
            q_start_index = find_closest_in_list(len(inter_pq_derivative)/2, find_peaks(inter_pq_derivative))
            q_start = q_wave - q_start_index

            q_waves.append(q_wave)
            p_waves.append(p_wave)
            q_waves_starts.append(q_start)
        except ValueError:
            pass
        except IndexError:
            pass

        try:
            epoch_after = np.array(ecg)[int(rpeak):int(rpeaks[index+1])]
            epoch_after = epoch_after[0:int(len(epoch_after)/2)]

            s_wave_index = np.min(find_peaks(epoch_after))
            s_wave = rpeak + s_wave_index
            t_wave_index = s_wave_index + np.argmax(epoch_after[s_wave_index:])
            t_wave = rpeak + t_wave_index

            inter_st = epoch_after[s_wave_index:t_wave_index]
            inter_st_derivative = np.gradient(inter_st, 2)
            t_start_index = find_closest_in_list(len(inter_st_derivative)/2, find_peaks(inter_st_derivative))
            t_start = s_wave + t_start_index
            t_end = np.min(find_peaks(epoch_after[t_wave_index:]))
            t_end = t_wave + t_end

            s_waves.append(s_wave)
            t_waves.append(t_wave)
            t_waves_starts.append(t_start)
            t_waves_ends.append(t_end)
        except ValueError:
            pass
        except IndexError:
            pass

# pd.Series(epoch_before).plot()
#    t_waves = []
#    for index, rpeak in enumerate(rpeaks[0:-1]):
#
#        epoch = np.array(ecg)[int(rpeak):int(rpeaks[index+1])]
#        pd.Series(epoch).plot()
#
#        # T wave
#        middle = (rpeaks[index+1] - rpeak) / 2
#        quarter = middle/2
#
#        epoch = np.array(ecg)[int(rpeak+quarter):int(rpeak+middle)]
#
#        try:
#            t_wave = int(rpeak+quarter) + np.argmax(epoch)
#            t_waves.append(t_wave)
#        except ValueError:
#            pass
#
#    p_waves = []
#    for index, rpeak in enumerate(rpeaks[1:]):
#        index += 1
#        # Q wave
#        middle = (rpeak - rpeaks[index-1]) / 2
#        quarter = middle/2
#
#        epoch = np.array(ecg)[int(rpeak-middle):int(rpeak-quarter)]
#
#        try:
#            p_wave = int(rpeak-quarter) + np.argmax(epoch)
#            p_waves.append(p_wave)
#        except ValueError:
#            pass
#
#    q_waves = []
#    for index, p_wave in enumerate(p_waves):
#        epoch = np.array(ecg)[int(p_wave):int(rpeaks[rpeaks>p_wave][0])]
#
#        try:
#            q_wave = p_wave + np.argmin(epoch)
#            q_waves.append(q_wave)
#        except ValueError:
#            pass
#
#    # TODO: manage to find the begininng of the Q and the end of the T wave so we can extract the QT interval


    ecg_waves = {"T_Waves": t_waves,
                 "P_Waves": p_waves,
                 "Q_Waves": q_waves,
                 "S_Waves": s_waves,
                 "Q_Waves_Onsets": q_waves_starts,
                 "T_Waves_Onsets": t_waves_starts,
                 "T_Waves_Ends": t_waves_ends}
    return(ecg_waves)

