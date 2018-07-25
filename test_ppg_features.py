#test_ppg_features
import pytest
from ppg_features import ppg_features as ppg
import numpy as np
import pickle

with open ('test_data/testData.pkl', 'rb') as f:
    test_data = pickle.load(f)
    
@pytest.fixture
def set_up_filter():
    nothing = test_data['nothing'] 
    chicken = test_data['chicken']
    peaker = np.array(([1,1000] + [1]*100)*50 + [1000])
    return nothing, chicken, peaker

def test_filter_ecg(set_up_filter):
    nothing, chicken, peaker = set_up_filter
    filt_nothing = ppg.filter_ecg(nothing)
    filt_chicken = ppg.filter_ecg(chicken)
    assert len(nothing) == len(filt_nothing)
    assert len(chicken) == len(filt_chicken)
    assert np.std(nothing) > 10*np.std(filt_nothing)
    assert np.std(chicken) < 10*np.std(filt_chicken)
    filt_chicken_preserve = ppg.filter_ecg(chicken, preserve_peak = True)
    assert np.std(filt_chicken_preserve) > np.std(filt_chicken)
    filt_peaker = ppg.filter_ecg(peaker)
    assert np.std(filt_peaker) < np.std(peaker)
    too_many_peak_points = ppg.filter_ecg(peaker, num_peak_points = 10000, preserve_peak = True)
    assert np.std(too_many_peak_points) < np.std(peaker)
    
    
def test_get_r_peaks(set_up_filter):
    nothing, chicken, peaker = set_up_filter
    filt_nothing = ppg.filter_ecg(nothing)
    filt_chicken = ppg.filter_ecg(chicken)
    filt_peaker = ppg.filter_ecg(peaker)
    filt_chicken_cut = ppg.filter_ecg(chicken[:4000])
    peaks_nothing = ppg.get_r_peaks(filt_nothing)
    peaks_chicken = ppg.get_r_peaks(filt_chicken)
    peaks_peaker = ppg.get_r_peaks(filt_peaker)
    peaks_cut = ppg.get_r_peaks(filt_chicken_cut)
    assert len(peaks_peaker) == 50
    assert len(peaks_nothing) > 0
    assert len(peaks_chicken) > 0
    assert len(peaks_chicken) == len(peaks_cut)