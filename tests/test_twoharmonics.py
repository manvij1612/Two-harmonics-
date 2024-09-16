import pytest
from twoharmonics.harmonics_waveforms import calculate_opening_angle, make_waveform_from_spherical, _make_waveform_from_cartesian, make_waveform
from twoharmonics.plot_fun import plot_waveform_harmonics, plot_waveform_harmonics_td, plot_waveform_harmonics_superposition_td
import numpy 
import lal
import lalsimulation as lalsim
import lalsimulation

def test_calculate_opening_angle():
	mass1 = 20.0
	mass2 = 10.0
	chi_eff = 0.3
	chi_p = 0.4
	freq = 20.0
	result = calculate_opening_angle(mass1, mass2, chi_eff, chi_p, freq)
	assert result == pytest.approx(0.139901305698753)

def test_make_waveform_from_spherical():
	approx = 'IMRPhenomPv2'
	theta_jn = 0.0
	phi_jl = 0.0
	phase = 0.0
	mass1 = 20.0
	mass2 = 10.0
	tilt_1 = 0.1
	tilt_2 = 0.2
	phi_12 = 0.3
	a_1 = 0.4
	a_2 = 0.0
	distance = 1.0
	f_ref = 20.0
	flow = 20.0
	fhigh = 1024.0
	df = 1.0 / 128.0
	flen = None
	hp, hc = make_waveform_from_spherical(approx, theta_jn, phi_jl, phase, mass1, mass2,
										  tilt_1, tilt_2, phi_12, a_1, a_2, distance, f_ref,
										  flow, fhigh, df, flen)
	assert hp is not None
	assert hc is not None


def test_make_waveform_from_cartesian():
	approximant = 'IMRPhenomPv2'
	mass1 = 30
	mass2 = 30
	spin_1x = [0.0]
	spin_1y = [0.0]
	spin_1z = [0.0]
	spin_2x = [0.0]
	spin_2y = [0.0]
	spin_2z = [0.0]
	iota = [0.0]
	phase = 0.0
	distance = 1.0
	flow = 20.0
	df = 1.0
	flen = 2048
	fhigh = 2048
	f_ref = None

	hp, hc = _make_waveform_from_cartesian(approximant, mass1, mass2, spin_1x, spin_1y, spin_1z, 
										   spin_2x, spin_2y, spin_2z, iota, phase, distance, 
										   flow, df, flen, fhigh, f_ref)

	assert isinstance(hp, lal.COMPLEX16FrequencySeries)
	assert isinstance(hc, lal.COMPLEX16FrequencySeries)
	
	assert hp.data.data.shape[0] == flen
	assert hc.data.data.shape[0] == flen