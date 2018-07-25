import scipy.signal as sig
import numpy as np

def filter_ppg(ppg, baseline_order = 2, baseline_width = 501):
    """
    filter and detrend the raw ppg data
    
    Args:
        ppg (np array): The raw ppg data we wish to filter.
        baseline_order (int): The order to use for the savgol_filter in detrending. Defualt 3
        baseline_width (int): How many points the savgol filter will be using to detrend. Default 101
    Returns:
        filtered (np array): Ther filtered and detrended ppg data
    """
    
    trendline = sig.savgol_filter(ppg, baseline_width, baseline_order)
    detrended_ppg = ppg - trendline
    
    return detrended_ppg

def spo2(ppgR, ppgIR):
    """
    given the red and infrared PPG data, get the spo2 level
    
    Args:
        ppgR (numpy array): The raw red ppg data.
        ppgIR (numpy array): The raw infrared ppg data
    Returns:
        spo2 (float): The spo2 level (in percent)
    """
    #cut off ends where user doesn't have a finger on the sensor
    start = 0
    end = len(ppgIR)
    """
    for i in range(len(ppgR)):
        if ppgR[i] < -70000:
            start = i + 400
            break
    for i in range(len(ppgR[start:])):
        if ppgR[start + i] > -70000:
            end = i + start - 400
            break
    if end != 0:
        ppgR = ppgR[start:end]
        ppgIR = ppgIR[start:end]
    else: 
        ppgR = ppgR[start:]
        ppgIR = ppgIR[start:]
    """
    filtIR = filter_ppg(ppgIR)
    filtR = filter_ppg(ppgR)
    mins = sig.argrelmin(filtR, order = 80)[0]
    values = filtR[mins]
    for i in range(len(values)):
        if min(values[i:i+3]) > -700: 
            start = mins[i]
            break
    for i in range(list(mins).index(start), len(values)):
        try:
            if values[i+1] < -700:
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
    spo2 = 110 - 25*abs(r)
    #if spo2 < 60:
        #raise ValueError("Make sure to put ppg Red first, then ppg Infrared")
        
    
    return spo2


def perfusion_index(ppg): #use Infrared PPG 
    """
    given the raw ppg data, caluclate the perfusion index
    
    Args:
        ppg (numpy array): The raw ppg data.
    
    Returns:
        pi (float): The perfusion index as a percentage
    """
    filt = filter_ppg(ppg)
    mins = sig.argrelmin(filt, order = 80)[0]
    med = np.median(filt[mins])
    valleys = []
    for point in mins:
        if abs(filt[point]) < abs(1.5*med):
            valleys.append(point)
    peaks = []
    for valley in valleys:
        peaks.append(list(filt).index(max(filt[valley:valley+50])))
    difs = []
    for i in range(len(peaks)):
        difs.append(abs(peaks[i] - valleys[i]))
    dif = np.median(difs)
    pi = (dif/abs(np.median(ppg)))*100
    
    return pi

def bpm(ppg, fs = 200): # use Infrared PPG
    """ 
    given the raw ppg data, give the average beats per minute        

    Args:
        ppg (numpy array): The raw ppg data.
        fs (int): The Sampling frequency in Hz (Default 200).
    
    Returns:
        bpm (float): The average heartrate in beats per minute.
    """
    filt = filter_ppg(ppg)
    mins = sig.argrelmin(filt, order = 80)[0]
    med = np.median(filt[mins])
    valleys = []
    for point in mins:
        if abs(filt[point]) < abs(1.5*med):
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

def pulse_transit_time(ecg, ppg, fs = 200):
    filt = filter_ppg(ppg)
    r_peaks = get_r_peaks(ecg)
    mins = sig.argrelmin(filt, order = 80)[0]
    med = np.median(filt[mins])
    valleys = []
    for point in mins:
        if abs(filt[point]) < abs(1.5*med):
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
    return ptt   