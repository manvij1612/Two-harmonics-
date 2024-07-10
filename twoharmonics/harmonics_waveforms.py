# Copyright (C) 2023  LIGO Scientific Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#

import numpy 
import lal
import lalsimulation as lalsim
import lalsimulation
from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions
from lalsimulation import SimInspiralTransformPrecessingWvf2PE
from lalsimulation import DetectorPrefixToLALDetector
from lalsimulation import FLAG_SEOBNRv4P_HAMILTONIAN_DERIVATIVE_NUMERICAL
from lalsimulation import FLAG_SEOBNRv4P_EULEREXT_QNM_SIMPLE_PRECESSION
from lalsimulation import FLAG_SEOBNRv4P_ZFRAME_L
from lalsimulation import SimNeutronStarEOS4ParameterPiecewisePolytrope
from lalsimulation import SimNeutronStarRadius
from lalsimulation import CreateSimNeutronStarFamily
from lalsimulation import SimNeutronStarLoveNumberK2
from lalsimulation import SimNeutronStarEOS4ParameterSpectralDecomposition
from lal import MSUN_SI, C_SI, MRSUN_SI
from conversions import calculate_dpsi, calculate_dphi, component_spins
from pycbc.types import FrequencySeries

#-
__author__ = "Stephen Fairhurst <stephen.fairhurst@ligo.org>, Amit Reza <amit.reza@ligo.org>, Sarah Caudill <sarah.caudill@ligo.org>"
__version__ = "FIXME"
__date__ = "FIXME"

#
# =============================================================================
#
#                           Two-harmonic waveform
#
# =============================================================================

def _make_waveform_from_cartesian(approximant, mass1, mass2, spin_1x, spin_1y, spin_1z, 
                                  spin_2x, spin_2y, spin_2z, iota, phase, distance, flow, 
                                  df, flen, fhigh, f_ref):
    
    
    if f_ref is None:
        f_ref = flow
    
    distance = 1.*1e6
    
    
    parameters = {}
    parameters['approximant'] = lalsim.GetApproximantFromString(str(approximant))
    
    
    hp, hc = lalsim.SimInspiralChooseFDWaveform(lal.MSUN_SI * mass1, lal.MSUN_SI * mass2,
                                                spin_1x[0], spin_1y[0], spin_1z[0], 
                                                spin_2x[0], spin_2y[0], spin_2z[0], distance*lal.PC_SI, 
                                                iota[0], phase, 0.0, 0.0, 0.0, df, flow, 
                                                fhigh, f_ref, None, parameters['approximant'])
    
    #- PyCBC Frequency Series is used
    hp = FrequencySeries(hp.data.data[:], delta_f = hp.deltaF, epoch = hp.epoch)

    hc = FrequencySeries(hc.data.data[:], delta_f = hc.deltaF, epoch = hc.epoch)
    
    
    if flen is not None:
        hp.resize(flen)
        hc.resize(flen)
    
    return hp, hc

def _cartesian_spins_from_spherical(theta_jn, phi_jl, tilt_1, tilt_2, phi_12, 
                                    a_1, a_2, mass1, mass2, f_ref, phase):
    
    
    data = component_spins(theta_jn, phi_jl, tilt_1, tilt_2, phi_12, 
                           a_1, a_2, mass1, mass2, f_ref, phase)
    data = numpy.array(data).T
    
    return data
    

def _make_waveform_from_spherical(approx, theta_jn, phi_jl, phase, mass1, mass2, 
                                  tilt_1, tilt_2, phi_12, a_1, a_2, distance, f_ref, 
                                  flow, fhigh, df, flen):
    
    
    if f_ref is None:
        f_ref = flow

    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = \
        _cartesian_spins_from_spherical([theta_jn], [phi_jl], [tilt_1], [tilt_2], 
                                        [phi_12], [a_1], [a_2], [mass1], [mass2], [f_ref], [phase])
    
    return _make_waveform_from_cartesian(approx, mass1, mass2, spin_1x, spin_1y, spin_1z, 
                                         spin_2x, spin_2y, spin_2z, iota, phase, distance, 
                                         flow, df, flen, fhigh, f_ref)

#-

def make_waveform(approx, theta_jn, phi_jl, phase, psi_J, mass1, mass2, tilt_1, tilt_2, phi_12, a_1, a_2, beta, distance, f_ref, flow, fhigh, df, flen):
    
    hp, hc = _make_waveform_from_spherical(approx, theta_jn, phi_jl, phase, mass1, mass2, tilt_1, tilt_2, 
                            phi_12, a_1, a_2, distance, f_ref, flow, fhigh, df, flen)

    dpsi = calculate_dpsi(theta_jn, phi_jl, beta)
    
    fp = numpy.cos(2 * (psi_J - dpsi))
    fc = -1. * numpy.sin(2 * (psi_J - dpsi))
    h = (fp * hp + fc * hc)
    h *= numpy.exp(2j * calculate_dphi(theta_jn, phi_jl, beta))
    
    return h

def calculate_precessing_harmonics(mass1, mass2, a_1, a_2, tilt_1, tilt_2, phi_12, beta, 
                                   distance, harmonics, approx, f_ref, flow, fhigh, df, flen):
    
    harm = {}
    if (0 in harmonics) or (4 in harmonics):
        h0 = make_waveform(approx, 0., 0., 0., 0., mass1, mass2, tilt_1, tilt_2, phi_12, 
                           a_1, a_2, beta, distance, f_ref, flow, fhigh, df, flen)
        hpi4 = make_waveform(approx, 0., 0., numpy.pi / 4, numpy.pi / 4, mass1, mass2, tilt_1, tilt_2,
                             phi_12, a_1, a_2, beta, distance, f_ref, flow, fhigh, df, flen)
        if (0 in harmonics):
            harm[0] = (h0 - hpi4) / 2
        
        if (4 in harmonics):
            harm[4] = (h0 + hpi4) / 2
        
    if (1 in harmonics) or (3 in harmonics):
        
        h0 = make_waveform(approx, numpy.pi / 2, 0., numpy.pi / 4, numpy.pi / 4, mass1, mass2,
                           tilt_1, tilt_2, phi_12, a_1, a_2, beta, distance, f_ref, 
                           flow, fhigh, df, flen)
        
        hpi2 = make_waveform(approx, numpy.pi / 2, numpy.pi / 2, 0., numpy.pi / 4, mass1, mass2,
                             tilt_1, tilt_2, phi_12, a_1, a_2, beta, distance, f_ref, 
                             flow, fhigh, df, flen)
        
        if (1 in harmonics):
            
            harm[1] = -1. * (h0 + hpi2) / 4
            
        if (3 in harmonics):
            
            harm[3] = -1. * (h0 - hpi2) / 4
            
    if (2 in harmonics):
        
        h0 = make_waveform(approx, numpy.pi / 2, 0., 0., 0., mass1, mass2, tilt_1, tilt_2, phi_12, 
                           a_1, a_2, beta, distance, f_ref, flow, fhigh, df, flen)
        hpi2 = make_waveform(approx, numpy.pi / 2, numpy.pi / 2, 0., 0., mass1, mass2, tilt_1, tilt_2,
                             phi_12, a_1, a_2, beta, distance, f_ref, flow, fhigh, df, flen)
        harm[2] = -(h0 + hpi2) / 6
        
    return harm

def add_waveforms(h1, h2):
    
    if h1.sample_rate != h2.sample_rate:
        raise ValueError('Sample rate must be the same')

    h_sum = h1.copy()
    h_sum_lal = h_sum.lal()
    lalsim.SimAddInjectionREAL8TimeSeries(h_sum_lal, h2.lal(), None)
    h_sum.data[:] = h_sum_lal.data.data[:]
    
    return h_sum

#-
def make_waveform_from_precessing_harmonics(
    harmonic_dict, theta_jn, phi_jl, phase, f_plus_j, f_cross_j
):
    """Generate waveform for a binary merger with given precessing harmonics and
    orientation

    Parameters
    ----------
    harmonic_dict: dict
        harmonics to include
    theta_jn: np.ndarray
        the angle between total angular momentum and line of sight
    phi_jl: np.ndarray
        the initial polarization phase
    phase: np.ndarray
        the initial orbital phase
    psi_J: np.ndarray
        the polarization angle in the J-aligned frame
    f_plus_j: np.ndarray
        The Detector plus response function as defined using the J-aligned frame
    f_cross_j: np.ndarray
        The Detector cross response function as defined using the J-aligned frame
    """
    A = _harmonic_amplitudes(theta_jn, phi_jl, f_plus_j, f_cross_j, harmonic_dict)

    h_app = 0
    for k, harm in harmonic_dict.items():
        if h_app:
            h_app += A[k] * harm
        else:
            h_app = A[k] * harm
    h_app *= np.exp(2j * phase + 2j * phi_jl)
    return h_app


def _harmonic_amplitudes(theta_jn, phi_jl, f_plus_j, f_cross_j, harmonics = [0, 1]):
    """Calculate the amplitudes of the precessing harmonics as a function of
    orientation

    Parameters
    ----------
    theta_jn: np.ndarray
        the angle between J and line of sight
    phi_jl: np.ndarray
        the precession phase
    f_plus_j: np.ndarray
        The Detector plus response function as defined using the J-aligned frame
    f_cross_j: np.ndarray
        The Detector cross response function as defined using the J-aligned frame
    harmonics: list, optional
        The list of harmonics you wish to return. Default is [0, 1]
    """
    amp = {}
    if 0 in harmonics:
        amp[0] = (
            (1 + np.cos(theta_jn)**2) / 2 * f_plus_j
            - 1j * np.cos(theta_jn) * f_cross_j
        )
    if 1 in harmonics:
        amp[1] = 2 * np.exp(-1j * phi_jl) * (
            np.sin(theta_jn) * np.cos(theta_jn) * f_plus_j
            - 1j * np.sin(theta_jn) * f_cross_j
        )
    if 2 in harmonics:
        amp[2] = 3 * np.exp(-2j * phi_jl) * (np.sin(theta_jn)**2) * f_plus_j
    if 3 in harmonics:
        amp[3] = 2 * np.exp(-3j * phi_jl) * (
            -np.sin(theta_jn) * np.cos(theta_jn) * f_plus_j
            - 1j * np.sin(theta_jn) * f_cross_j
        )
    if 4 in harmonics:
        amp[4] = np.exp(-4j * phi_jl) * (
            (1 + np.cos(theta_jn)**2) / 2 * f_plus_j
            + 1j * np.cos(theta_jn) * f_cross_j
        )
    return amp
#-

def calculate_opening_angle(mass1, mass2, chi_eff, chi_p, freq):
    
    theta1 = numpy.arctan2(chi_p, chi_eff)
    chi1 = numpy.sqrt(chi_p**2 + chi_eff**2)

    beta, s1x, s1y, s1z, s2x, s2y, s2z = SimInspiralTransformPrecessingNewInitialConditions(0., 0.,
                                theta1, 0., 0., chi1, 0., mass1*MSUN_SI, mass2*MSUN_SI, freq, 0.)
    return beta

def make_full_waveform(hp, hc, psi0):
    
    fp = numpy.cos(2 * psi0)
    fc = numpy.sin(2 * psi0)
    temp = fp * hp + fc * hc
    
    return temp


def make_single_spin_waveform(waveform, theta_JN, phi_JL, phi0, psi0, chi_eff, chi_p, 
                              mass1, mass2, flow, df, flen):
    
    hp, hc = make_single_spin_plus_cross(waveform, theta_JN, phi_JL, phi0, chi_eff, chi_p, 
                                         mass1, mass2, flow, df, flen)

    beta = calculate_opening_angle(mass1, mass2, chi_eff, chi_p, flow)
    dpsi = calculate_dpsi(theta_JN, phi_JL, beta)
    h = make_full_waveform(hp, hc, psi0 + dpsi)
    dphi = calculate_dphi(theta_JN, phi_JL, beta)
    h *= numpy.exp(2j * dphi )
    
    return h

def make_single_spin_plus_cross(approximant, theta_JN, phi_JL, phi0, chi_eff, chi_p, mass1, mass2, flow, df, flen):
   
    chi1 = numpy.sqrt(chi_p**2 + chi_eff**2)
    theta1 = numpy.arctan2(chi_p, chi_eff)
    inc, s1x, s1y, s1z, s2x, s2y, s2z = SimInspiralTransformPrecessingNewInitialConditions(theta_JN, 
                                                            phi_JL, theta1, 0., 0., chi1, 0.,
                                                            mass1*MSUN_SI, mass2*MSUN_SI, flow, phi0)
     
    fhigh = 2048
    f_ref = flow
    distance = 1.*1e6
    parameters = {}
    parameters['approximant'] = lalsim.GetApproximantFromString(str(approximant))
    
    hp, hc = lalsim.SimInspiralChooseFDWaveform(lal.MSUN_SI*mass1, lal.MSUN_SI*mass2, 
                                                s1x, s1y, s1z, s2x, s2y, s2z,
                                                distance*lal.PC_SI, inc, phi0, 
                                                0.0, 0.0, 0.0, df, flow, fhigh,
                                                f_ref, None, parameters['approximant'])
    
    hp = FrequencySeries(hp.data.data[:], delta_f = hp.deltaF, epoch = hp.epoch)
    hc = FrequencySeries(hc.data.data[:], delta_f = hc.deltaF, epoch = hc.epoch)
    
    hp.resize(flen)
    hc.resize(flen)
    
    return hp, hc
