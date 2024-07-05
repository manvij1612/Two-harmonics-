import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from harmonics_waveforms import * 
from pycbc.types.timeseries import TimeSeries

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
        plt.loglog(wf.get_sample_frequencies(), abs(wf), label = r"$\alpha_{0} = %s$" % wk, color = c[i], lw = 3 )
    
    for m, c in coeff_dict.items():
        plt.loglog(harmonic_dict[m].get_sample_frequencies(), c * abs(harmonic_dict[m]), '--', label = 'harmonic %d' % m, lw = 3 )

    plt.xlim(flow, fhigh)
    plt.ylim(1e-23, 2e-20)
    plt.grid('False')
    plt.legend(loc = 'best', edgecolor = 'black', fontsize = 15)
    plt.xlabel('Frequency (Hz)', fontsize = 20)
    plt.ylabel('Strain', fontsize = 20)
    if title: plt.title(title)
        
        
def plot_waveform_harmonics_td(waveform_dict, harmonic_dict, coeff_dict, times = [-2, 0.0], 
                               title = None, fname = None, fig_dir = 'figures'):
    
    
    plt.figure(figsize = figsize)
    c = ['m' , 'y']
    
    for i, (wk, wf) in enumerate (waveform_dict.items()):
        ht = wf.to_timeseries()
        plt.plot(ht.get_sample_times(), ht, label = r'$\alpha_{0} = %s$' % wk, color = c[i], lw = 5)
    
    for m, c in coeff_dict.items():
        ht = harmonic_dict[m].to_timeseries()
        plt.plot(ht.get_sample_times(), c * ht, '--', label = 'harmonic %d' % m, lw = 5)
    
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
        htmp = wf.to_timeseries()
    
    h_sup = TimeSeries(np.zeros(len(htmp)), delta_t = htmp.delta_t)
    
    ax1 = plt.axes()  # standard axes
    ax2 = plt.axes([0.14, 0.5, 0.45, 0.35])
    
    for m, c in coeff_dict.items():
        
        ht = harmonic_dict[m].to_timeseries()
        h_sup = add_waveforms(c*ht, h_sup)
        
        ax1.plot(ht.get_sample_times(), c * ht, '--', label = 'harmonic %d' % m, lw = 3 )
        ax2.plot(ht.get_sample_times(), c * ht, '--', label = 'harmonic %d' % m, lw = 3 )
    
    sup_env = hilbert(h_sup)
    
    ax1.plot(h_sup.get_sample_times(), h_sup, alpha = 0.6, color = 'gray', lw = 3)
    ax1.fill_between(h_sup.sample_times , np.abs(sup_env) , -np.abs(sup_env), alpha = 0.6, 
                     label = 'superposition', color = 'gray', lw = 3)
    
    ax2.plot(h_sup.get_sample_times(), h_sup, alpha = 0.6, color = 'gray', lw = 3)
    ax2.fill_between(h_sup.sample_times , np.abs(sup_env) , -np.abs(sup_env), alpha = 0.6, 
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

