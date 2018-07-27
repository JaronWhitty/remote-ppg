#test_ppg_features
import pytest
from ppg_features import ppg_features as ppg
import numpy as np


    
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
    noise = [1000, -1000]*1000
    nsr_IR_noise = np.array(noise + list(nsr_IR) + noise)
    nsr_R_noise = np.array(noise + list(nsr_R) + noise)
    assert ppg.spo2(nsr_R, nsr_IR) > 90
    assert ppg.spo2(nsr_R, nsr_R) == 85
    assert ppg.spo2(nsr_IR, nsr_IR) == 85
    #adding noise to the beginning and end of the ppg signal should not effect the spo2 much
    assert abs(ppg.spo2(nsr_R, nsr_IR) - ppg.spo2(nsr_R_noise, nsr_IR_noise)) < .1
    
def test_perfusion_index(set_up_ppg):
    nsr_IR, nsr_R = set_up_ppg
    nsr_IR_lessDC = nsr_IR - 10000
    assert ppg.perfusion_index(nsr_IR) < ppg.perfusion_index(nsr_IR_lessDC)
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
    
    