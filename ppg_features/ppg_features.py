from __future__ import division
import scipy.signal as sig
import numpy as np
import ecg_feature_selection as efs


def filter_ppg(ppg, baseline_order = 2, baseline_width = 501):
    """
    filter and detrend the raw ppg data
    
    Args:
        ppg (np array): The raw ppg data we wish to filter.
        baseline_order (int): The order to use for the savgol_filter in detrending. Defualt 3
        baseline_width (int): How many points the savgol filter will be using to detrend. Default 101
    Returns:
        numpy array: Ther filtered and detrended ppg data
    """
    
    trendline = sig.savgol_filter(ppg, baseline_width, baseline_order)
    detrended_ppg = ppg - trendline
    
    return detrended_ppg


def spo2(ppgR, ppgIR, peak_order = 80, pulse_threshold = -700, consecutive_points = 3):
    """
    given the red and infrared PPG data, get the spo2 level
    
    Args:
        ppgR (numpy array): The raw red ppg data.
        ppgIR (numpy array): The raw infrared ppg data
        peak_order (int): Approximate number of points the peak detect algorithm will use to search for local minima. Default 80
        pulse_threshold (int): Number below which we calssify as not part of the pulse wave. Default -700
        consectuive_points (int): How many consecutive miniumums that should be within the pulse_threshold to know where to start pulling real ppg data. Default 3
    Returns:
        float: The spo2 level (in percent)
    """
    #cut off ends where user doesn't have a finger on the sensor
    start = 0
    end = len(ppgIR)
    filtIR = filter_ppg(ppgIR)
    filtR = filter_ppg(ppgR)
    mins = sig.argrelmin(filtR, order = peak_order)[0]
    values = filtR[mins]
    for i in range(len(values)):
        if min(values[i:i+consecutive_points]) > pulse_threshold: 
            start = mins[i]
            break
    for i in range(list(mins).index(start), len(values)):
        try:
            if values[i+1] < pulse_threshold:
                end = mins[i]
                break
        except IndexError:
            break
    #get the DC portion of the signals
    print(start, end)
    dc_r = np.median(ppgR[start:end])
    dc_ir = np.median(ppgIR[start:end])
    #get the root mean square of the AC portion of the signals
    rms_r = np.sqrt(np.mean(filtR[start:end]**2))
    rms_ir = np.sqrt(np.mean(filtIR[start:end]**2))
    #calculate the absorption ratio
    r = (rms_r/dc_r)/(rms_ir/dc_ir)
    #using the ratio calculate the spo2 level
    #spo2 = 114.515 - 37.313*abs(r)
    spo2 = 110 - 25*abs(r) #standard forumla for spo2 level is 110 - 25*((ACrms of Red/DC of red)/(ACrms of IR/DC of IR))
    #if spo2 < 60:
        #raise ValueError("Make sure to put ppg Red first, then ppg Infrared")
        
    
    return spo2


def perfusion_index(ppg, peak_order = 80, pulse_threshold = 1.5, look_distance = 50): #use Infrared PPG 
    """
    given the raw ppg data, caluclate the perfusion index
    
    Args:
        ppg (numpy array): The raw ppg data.
        peak_order (int): The approximate number of points to be looking for local minima. Default 80
        pulse_threshold (float): How many times greater than the median above which we classify as non pulse wave. Default 1.5
        look_distance (int): How many points after the start of the pulse wave to look for the pulse peak. Default 50
    
    Returns:
        float: The perfusion index as a percentage
    """
    #filter out the DC offset 
    filt = filter_ppg(ppg)
    #find the mins and peaks of the ppg data
    mins = sig.argrelmin(filt, order = peak_order)[0]
    med = np.median(filt[mins])
    valleys = []
    for point in mins:
        if abs(filt[point]) < abs(pulse_threshold*med):
            valleys.append(point)
    peaks = []
    for valley in valleys:
        peaks.append(list(filt[valley:]).index(max(filt[valley:valley + look_distance]))+ valley)
    #find the differences between the peaks and the mins
    difs = []
    for i in range(len(peaks)):
        difs.append(abs(peaks[i] - valleys[i]))
    dif = np.median(difs)
    #the perfusion index is defined as (AC/DC)*100
    pi = (dif/abs(np.median(ppg)))*100
    
    return pi

def bpm(ppg, fs = 200, peak_order = 80, pulse_threshold = 1.5): # use Infrared PPG
    """ 
    given the raw ppg data, give the average beats per minute        

    Args:
        ppg (numpy array): The raw ppg data.
        fs (int): The Sampling frequency in Hz (Default 200).
        peak_order (int): The approximate number of points to be looking for local minima. Default 80
        pulse_threshold (float): How many times greater than the median above which we classify as non pulse wave. Default 1.5   
    Returns:
        float: The average heartrate in beats per minute.
    """
    filt = filter_ppg(ppg)
    mins = sig.argrelmin(filt, order = peak_order)[0]
    med = np.median(filt[mins])
    valleys = []
    for point in mins:
        if abs(filt[point]) < abs(pulse_threshold*med):
            valleys.append(point)
    deltas = []
    last = valleys[0]
    for valley in valleys[1:]:
        deltas.append(valley - last)
        last = valley
    med_delta = np.median(deltas)
    med_delta_sec = med_delta / fs
    bpm = (1/med_delta_sec)*60
    return bpm

def pulse_transit_time(ecg, ppg, fs = 200, peak_order = 80, pulse_threshold = 1.5):
    """
    given the raw ecg and ppg data, find the median pulse transit time between r peaks and begining of the ppg pulse wave
    
    Args:
        ecg (numpy array): The raw ECG data
        ppg (numpy array): The raw PPG data
        fs (int): The sampling frequency in Hz. Default 200
        peak_order (int): The approximate number of points to be looking for local minima. Default 80
        pulse_threshold (float): How many times greater than the median above which we classify as non pulse wave. Default 1.5
    Returns:
        float: The median pulse transit time
    """
    filt = filter_ppg(ppg)
    r_peaks = efs.get_r_peaks(ecg)
    mins = sig.argrelmin(filt, order = peak_order)[0]
    med = np.median(filt[mins])
    valleys = []
    for point in mins:
        if abs(filt[point]) < abs(pulse_threshold*med):
            valleys.append(point)
    #peaks = []
    #for valley in valleys:
        #peaks.append(list(filt).index(max(filt[valley:valley+50])))
    #peaks = np.array(peaks)
    #r_peaks = get_r_peaks(ecg)
    #return valleys, r_peaks
    ptt = []
    used_r_peaks = []
    for i in range(len(valleys)):
        for j in range(len(r_peaks)):
            if valleys[i] - r_peaks[j] < fs and j not in used_r_peaks and valleys[i] - r_peaks[j] > 0:
                ptt.append((valleys[i] - r_peaks[j])/fs)
                used_r_peaks.append(j)
                continue
    median_ptt = np.median(ptt)
    return median_ptt   


