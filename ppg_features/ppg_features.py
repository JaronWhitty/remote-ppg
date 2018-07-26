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

def filter_ecg(signal, normalized_frequency = .6, Q = 30, baseline_width = 301, 
               baseline_order = 3, baseline_freq_low = .01, baseline_freq_high = .1, fs = 200, butter_order = 2,
               points = 11, num_peak_points = 5, preserve_peak = False):
    """
    filter and detrend a raw ECG signal 
    
    Args:
        signal (numpy array): The raw ECG data 
        normalized_frequency (float): the normalized frequency we wish to filter out, must be between 0 and 1, with 1 being half the sampling frequency. Default .6
        Q (int): Quality factor for the notch filter. Default 30
        baseline_width (int): How wide a window to use for the baseline removal. Default 301
        baseline_order (int): Polynomial degree to use for the baseline removal. Default 3
        baseline_freq_low (float): low end of frequency to cut off to eliminate baseline drift. Default .01 Hz
        baseline_freq_high (float): high end frequency to cut off to eliminate baseline drift. Default .1 Hz
        butter_order (int): The order of the butter filter used to eliminate baseline drift. Defualt 2
        points (int): The number of points to use for the bartlett window. Default 11
        num_peak_points (int): The number of points around each r-peak to keep at their original amplitude. Default 5
        
    Returns:
         numpy array: The filtered and detrended ECG signal
    """
    #filter out some specific frequency noise
    b, a = sig.iirnotch(normalized_frequency, Q)
    filt_signal = sig.filtfilt(b, a, signal, axis = 0)
    #remove baseline wander
    #baseline = sig.savgol_filter(filt_signal, baseline_width, baseline_order, axis = 0)
    #detrended_signal = filt_signal - baseline
    #using a zero phase iir filter based off a butterworth of order 2 cutting off low frequencies
    """ Other Option (Use a much higher baseline_width, 1301 perhaps) """
    nyquist = fs / 2
    bb, ba = sig.iirfilter(butter_order, [baseline_freq_low / nyquist, baseline_freq_high / nyquist])
    trend = sig.filtfilt(bb, ba, filt_signal, axis = 0)
    #center trendline onto signal
    together = np.median(trend) - np.median(filt_signal)
    trend_center = trend - together
    baseline_removed = filt_signal - trend_center
    trend2 = sig.savgol_filter(baseline_removed, baseline_width, baseline_order)
    baseline_removed = baseline_removed - trend2 
    #wiener filter
    #filt_signal = sig.wiener(detrended_signal)
    filt_signal = sig.wiener(baseline_removed)
    #smooth signal some more for cleaner average heartbeat
    bart = np.bartlett(points)
    smooth_signal = np.convolve(bart/bart.sum(), filt_signal, mode = 'same')
    
    #preserve the r-peak amplitude (if desired)
    #When smoothing the curve with np.convolve, we destroy amplitude in the r-peaks. We use the detrended signal's 
    #peak amplitude to preserve the r-peak amplitude to stay consistent with a normal ECG waveform. 
    if preserve_peak:
        #r_peaks = get_r_peaks(detrended_signal)
        r_peaks = get_r_peaks(baseline_removed)
        for peak in r_peaks:
            for i in range(num_peak_points):
                #smooth_signal[peak + i] = detrended_signal[peak + i]
                #smooth_signal[peak - i] = detrended_signal[peak - i]
                if peak + i > len(baseline_removed)-1 or peak - i < 0:
                    continue
                else:
                    smooth_signal[peak + i] = baseline_removed[peak + i]
                    smooth_signal[peak - i] = baseline_removed[peak - i]
                
    
    
    return smooth_signal

def get_r_peaks(signal, exp = 3, peak_order = 80, high_cut_off = .8, low_cut_off = .5, med_perc = .55, too_noisy = 1.6, noise_level = 5000, noise_points = 10):
    """
    get the r peaks from raw ecg_signal 
    
    Args:
        signal (numpy array): The signal from which to find the r-peaks
        exp (int): exponent that we take the signal data to, find peaks easier. Default 2
        peak_order (int): number of data points on each side to compare when finding peaks. Default 80
        high_cut_off (float): percent above the median r-peak amplitude that constitues an invalid r-peak. Default .8
        low_cut_off (float): percent below the median r-peak amplitude that constitutes an invalid r-peak. Dfeault .5
        med_perc (float): percent of the median time one peak back and one peak forward that would surely not be an r peak. Defualt = .55
        too_noisy (float): How many times the median standard deviation around an R peak that flags noise instead of acutal heart beat. Default 1.6
        noise_level (float): Number above which we would consider noise from the original signal. Default 5000
        noise_points (int): Number of points on each side of the peaks to check for the noise level. Default 10
    Returns:
        numpy array: The indexes of the detected r-peaks
    """
    og_signal = signal
    signal = filter_ecg(signal)
    #exentuate the r peaks
    #r_finder = signal**exp
    #peaks = sig.argrelextrema(r_finder, np.greater, order = peak_order, mode = 'wrap')
    #convert peaks to 1D numpy array
    #peaks = peaks[0]
    #use derivative and find mins to find general location of r peaks
    deriv = np.gradient(signal)
    peak_areas = sig.argrelmin(deriv, order = peak_order)
    peak_areas = peak_areas[0]
    #now find the maximum around each peak area
    peaks = []
    for area in peak_areas:
        try:
            peaks.append(list(signal).index(max(signal[area-10:area])))
        except ValueError: #if the area is right at the beginning 
            peaks.append(list(signal).index(max(signal[:area])))
    peaks = np.array(peaks)   
    #when user is not touching the electrodes correctly, the sensor gives very high amplitude spikes, we ignore these
    #ocassionaly there are higher amplitude t-waves then normal. These are still shorter amplitude to the r-peaks. We ignore these as well
    median = np.median(signal[peaks])
    valid = []
    for i in range(len(peaks)):
        if abs(signal[peaks[i]]) <= abs(median + median * high_cut_off) and abs(signal[peaks[i]]) >= abs(median - median * low_cut_off):
            valid.append(i)
    peaks = peaks[valid]        
    #often times noise is filtered down to around the same level as r peaks and the standard deviation filter isn't good enough
    #To cover these cases we look at the original unfiltered signal to take out peaks that are noise
    valid = []
    for i in range(len(peaks)):
        if not any(og_signal[peaks[i] - noise_points: peaks[i] + noise_points] > noise_level):
            valid.append(i)
    peaks = peaks[valid]
    #when the signal is all noise this will get rid of all the peaks
    if len(peaks) == 0:
        return peaks
    #some t-waves are still caught in r-peak detection to filter those out look at the distance between peaks
    #we look at the distances from one peak back to one peak forward, thus to single out t peaks
    dist = []
    for i in range(1, len(peaks) - 1):
        dist.append(peaks[i+1] - peaks[i-1])
    median = np.median(dist)
    #from the way we look at the distance we skipped the first and last, so add them back in
    
    not_t = [0]
   
    for i in range(len(dist)):
        if dist[i] > median*med_perc:
            not_t.append(i + 1)

    not_t.append(len(peaks) -1)
    #occasionally there happens to be noise at a similar amplitude and similar distances as r-peaks 
    #to get rid of these we can eliminate the detected peaks that have unusally high standard deviations around them
    peaks = peaks[not_t]
    not_noise = []
    #find the distance before and after each peak to look at
    dist = []
    last = peaks[0]
    for i in range(1, len(peaks)):
        dist.append(peaks[i] - last)
        last = peaks[i]
    med_distance = np.median(dist)
    look_distance = int(med_distance / 2)
    #get the standard deviation around each peak
    stds = []
    for peak in peaks:
        if peak - look_distance < 0:
            stds.append(np.std(signal[:look_distance]))
            continue
        else:
            stds.append(np.std(signal[peak - look_distance:peak + look_distance]))
            
    med_std = np.median(stds)
    #accept only the peaks with more normal standard deviation around it
    for i in range(len(stds)):
        if stds[i] < too_noisy* med_std and stds[i] > 1/too_noisy * med_std:
            not_noise.append(i)
            
    peaks = peaks[not_noise]
    
    return peaks

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
        spo2 (float): The spo2 level (in percent)
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
        pi (float): The perfusion index as a percentage
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
        peaks.append(list(filt).index(max(filt[valley:valley + look_distance])))
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
        bpm (float): The average heartrate in beats per minute.
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
        median_ptt (float): The median pulse transit time
    """
    filt = filter_ppg(ppg)
    r_peaks = get_r_peaks(ecg)
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