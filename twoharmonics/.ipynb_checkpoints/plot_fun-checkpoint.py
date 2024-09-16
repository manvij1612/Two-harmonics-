import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from twoharmonics.harmonics_waveforms import calculate_opening_angle, make_waveform_from_spherical, _make_waveform_from_cartesian, make_waveform, calculate_precessing_harmonics, make_single_spin_waveform 
import lal
from lal import LIGOTimeGPS, DimensionlessUnit

#-
# set the plotting specs
plt.rcParams.update({
    'lines.markersize': 6,
    'lines.markeredgewidth': 1.5,
    'lines.linewidth': 2.0,
    'font.size': 20,
    'axes.titlesize':  20,
    'axes.labelsize':  20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 20
})
figsize = [20, 10]
#-

def plot_waveform_harmonics(waveform_dict, harmonic_dict, coeff_dict, flow, 
                            title = None, fname = None, fig_dir = 'figures'):
    
    fhigh = 2048
    
    plt.figure(figsize = figsize)
    c = ['m' , 'y']
    
    for i, (wk, wf) in enumerate(waveform_dict.items()):
        freq = np.linspace(flow,fhigh,len(wf))
        plt.loglog(freq, abs(wf), label = r"$\alpha_{0} = %s$" % wk, color = c[i], lw = 3 )
    
    for m, c in coeff_dict.items():
        freq = np.linspace(flow,fhigh,len(harmonic_dict[m]))
        plt.loglog(freq, c * abs(harmonic_dict[m]), '--', label = 'harmonic %d' % m, lw = 3 )

    print(len(wf))
	plt.xlim(flow, fhigh)
    plt.ylim(1e-23, 2e-20)
    plt.grid('False')
    plt.legend(loc = 'best', edgecolor = 'black', fontsize = 15)
    plt.xlabel('Frequency (Hz)', fontsize = 20)
    plt.ylabel('Strain', fontsize = 20)
    if title: plt.title(title)

#def convert_to_freq_series(wf_data, delta_f, epoch):
	#freq_series = lal.CreateCOMPLEX16FrequencySeries( name='waveform', epoch=epoch, f0=0.0, deltaF=delta_f, units=lal.DimensionlessUnit, #length=len(wf_data))
    #freq_series.data.data = wf_data
    #return freq_series

def freq_to_time_series(freq_series, delta_f):
    delta_t = 1/delta_f
    time_data = np.fft.ifft(freq_series.data.data) 
    return lal.CreateREAL8TimeSeries(name = 'h(t)', epoch = LIGOTimeGPS(0), f0 = 0.0, delta_t = deltaT, unit = lal.DimensionlessUnit,length=len(time_data.real), data=time_data.real)

def plot_waveform_harmonics_td(waveform_dict, harmonic_dict, coeff_dict, times = [-2, 0.0], 
                               title = None, fname = None, fig_dir = 'figures'):
    
    
    plt.figure(figsize = figsize)
    c = ['m' , 'y'] 
    
    for i, (wk, wf) in enumerate (waveform_dict.items()):
        ht = freq_to_time_series(wf, wf.deltaF)
        plt.plot(ht.epoch.gpsSeconds + np.arange(len(ht.data)) * ht.deltaT, ht.data, label = r'$\alpha_{0} = %s$' % wk, color = c[i], lw = 5)
    
    for m, c in coeff_dict.items():
        ht = freq_to_time_series(harmonic_dict[m], harmonic_dict[m].deltaF)
        plt.plot(ht.epoch.gpsSeconds + np.arange(len(ht.data)) * ht.deltaT, c * ht.data, '--', label = 'harmonic %d' % m, lw = 5)
    
    plt.xlim(times)
    plt.grid('False')
    plt.legend(loc = 'best', edgecolor = 'black', fontsize = 15)
    plt.xlabel('Time (sec)', fontsize = 20)
    plt.ylabel('Strain', fontsize = 20)
    if title: plt.title(title)

def plot_waveform_harmonics_superposition_td(waveform_dict, harmonic_dict, coeff_dict, 
                                             times = [-2, 0.0], title = None, fname = None, 
                                             fig_dir = 'figures'):
    
    
    plt.figure(figsize = figsize)
    for wk, wf in waveform_dict.items():
        htmp = freq_to_time_series(wf, wf.deltaF)
    
    h_sup = CreateREAL8TimeSeries('h_sup', htmp.epoch, 0, htmp.deltaT, DimensionlessUnit, len(htmp.data), np.zeros(len(htmp.data)))
    
    ax1 = plt.axes()  # standard axes
    ax2 = plt.axes([0.14, 0.5, 0.45, 0.35])
    
    for m, c in coeff_dict.items():
        
        ht = freq_to_time_series(harmonic_dict[m], harmonic_dict[m].deltaF)
        h_sup.data += c * ht.data
        
        ax1.plot(ht.epoch.gpsSeconds + np.arange(len(ht.data)) * ht.deltaT, c * ht.data, '--', label = 'harmonic %d' % m, lw = 3 )
        ax2.plot(ht.epoch.gpsSeconds + np.arange(len(ht.data)) * ht.deltaT, c * ht.data, '--', label = 'harmonic %d' % m, lw = 3 )
    
    sup_env = hilbert(h_sup.data)
    
    ax1.plot(h_sup.epoch.gpsSeconds + np.arange(len(h_sup.data)) * h_sup.deltaT, h_sup.data, alpha = 0.6, color = 'gray', lw = 3)
    ax1.fill_between(h_sup.epoch.gpsSeconds + np.arange(len(h_sup.data)) * h_sup.deltaT, np.abs(sup_env) , -np.abs(sup_env), alpha = 0.6, 
                     label = 'superposition', color = 'gray', lw = 3)
    
    ax2.plot(h_sup.epoch.gpsSeconds + np.arange(len(h_sup.data)) * h_sup.deltaT, h_sup.data, alpha = 0.6, color = 'gray', lw = 3)
    ax2.fill_between(h_sup.epoch.gpsSeconds + np.arange(len(h_sup.data)) * h_sup.deltaT, np.abs(sup_env) , -np.abs(sup_env), alpha = 0.6, 
                     label = 'superposition', color = 'gray', lw = 3)
    
    ax1.set_xlim(times)
    ax1.set_ylim((np.min(np.real(-sup_env)), np.max(np.real(sup_env)) + 6e-19))
    ax2.set_ylim((0.6 * np.min(np.real(-sup_env)), 0.6 * np.max(np.real(sup_env))))
    
    ax2.set_xlim((-0.35 , -0.01))
    ax2.set_yticks([])
    ax1.grid()
    plt.legend(loc = 'best', bbox_to_anchor = (1.65, 0.5), edgecolor = 'black', fontsize = 15)
    ax1.set_xlabel('Time(s)', fontsize = 20)
    ax1.set_ylabel('Strain', fontsize = 20)
    if title: plt.title(title)