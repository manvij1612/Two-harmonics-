import numpy 
import lal
from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions
from lal import MSUN_SI, C_SI, MRSUN_SI
#-

def component_spins(theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass1, mass2, f_ref, phase):
    
    data = []
    for i in range(len(theta_jn)):
        iota, S1x, S1y, S1z, S2x, S2y, S2z = \
        SimInspiralTransformPrecessingNewInitialConditions(theta_jn[i], phi_jl[i], tilt_1[i], 
                                                           tilt_2[i], phi_12[i], a_1[i], a_2[i], 
                                                           mass1[i] *MSUN_SI, mass2[i] * MSUN_SI,
                                                           float(f_ref[i]), float(phase[i]))
        
        data.append([iota, S1x, S1y, S1z, S2x, S2y, S2z])
        
        return data


def calculate_dpsi(theta_jn, phi_jl, beta):
    
    if theta_jn == 0:
        return -1. * phi_jl
    
    n = numpy.array([numpy.sin(theta_jn), 0, numpy.cos(theta_jn)])
    j = numpy.array([0, 0, 1])
    l = numpy.array([numpy.sin(beta) * numpy.sin(phi_jl), numpy.sin(beta) * numpy.cos(phi_jl), numpy.cos(beta)])
    p_j = numpy.cross(n, j)
    p_j /= numpy.linalg.norm(p_j)
    p_l = numpy.cross(n, l)
    p_l /= numpy.linalg.norm(p_l)
    cosine = numpy.inner(p_j, p_l)
    sine = numpy.inner(n, numpy.cross(p_j, p_l))
    dpsi = numpy.pi / 2 + numpy.sign(sine) * numpy.arccos(cosine)
    
    return dpsi

def calculate_dphi(theta_jn, phi_jl, beta):
    
    if theta_jn == 0:
        return 0.
    
    n = numpy.array([numpy.sin(theta_jn), 0, numpy.cos(theta_jn)])
    j = numpy.array([0, 0, 1])
    l = numpy.array([numpy.sin(beta) * numpy.sin(phi_jl), 
                     numpy.sin(beta) * numpy.cos(phi_jl), numpy.cos(beta)])
    e_j = numpy.cross(j,l)
    e_j /= numpy.linalg.norm(e_j)
    e_n = numpy.cross(n,l)
    e_n /= numpy.linalg.norm(e_n)
    cosine = numpy.inner(e_j, e_n)
    sine = numpy.inner(l, numpy.cross(e_j, e_n))
    dphi = numpy.sign(sine)*numpy.arccos(cosine)
    
    return dphi
