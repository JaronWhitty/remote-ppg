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
    
@pytest.fixture
def set_up_filter_ppg():
    flat = [0]*5000
    not_flat = np.arange(0, 5000)
    return flat, not_flat

def test_filter_ppg(set_up_filter_ppg):
    flat, not_flat = set_up_filter_ppg
    filt_flat = ppg.filter_ppg(flat)
    filt_not_flat = ppg.filter_ppg(not_flat)
    assert np.mean(flat) == np.mean(filt_flat)
    assert np.mean(not_flat) != np.mean(filt_not_flat)
    assert np.std(not_flat) > np.std(filt_not_flat)

@pytest.fixture
def set_up_ppg():
    x = np.arange(5000)
    nsr_IR = 350*np.sin(2*np.pi*x/200) + 85000
    nsr_R = 200*np.sin(2*np.pi*x/200) + 70000
    return nsr_IR, nsr_R

def test_spo2(set_up_ppg):
    nsr_IR, nsr_R = set_up_ppg
    assert ppg.spo2(nsr_R, nsr_IR) > 90
    assert ppg.spo2(nsr_R, nsr_R) == 85
    assert ppg.spo2(nsr_IR, nsr_IR) == 85
    
def test_perfusion_index(set_up_ppg):
    nsr_IR, nsr_R = set_up_ppg
    nsr_IR_lessDC = nsr_IR - 10000
    assert ppg.perfusion_index(nsr_IR) > ppg.perfusion_index(nsr_IR_lessDC)
    assert ppg.perfusion_index(nsr_R) > ppg.perfusion_index(nsr_IR)

@pytest.fixture
def set_up_bpm():
    x = np.arange(5000)
    nsr_IR_60 = 350*np.sin(2*np.pi*x/200) + 85000
    nsr_R_60 = 200*np.sin(2*np.pi*x/200) + 70000
    nsr_IR_120 = 350*np.sin(2*np.pi*x/100) + 85000
    nsr_R_120 = 200*np.sin(2*np.pi*x/100) + 70000
    return nsr_IR_60, nsr_R_60, nsr_IR_120, nsr_R_120
    
    
def test_bpm(set_up_bpm):
    IR_60, R_60, IR_120, R_120 = set_up_bpm
    assert ppg.bpm(IR_60) == ppg.bpm(R_60) == 60.0
    assert ppg.bpm(IR_120) == ppg.bpm(R_120) == 120.0

@pytest.fixture
def set_up_ptt():
    x = np.arange(2000)
    sin_wave = np.sin(2*np.pi*x/200)
    cos_wave = np.cos(2*np.pi*x/200)
    return sin_wave, cos_wave

def test_pulse_transit_time(set_up_ptt):
    sin_wave, cos_wave = set_up_ptt
    #set up waves are at 60 bpm so the pulse transit time difference should be half a second
    assert abs(ppg.pulse_transit_time(sin_wave, cos_wave) - ppg.pulse_transit_time(cos_wave, sin_wave)) == 0.5
    
    