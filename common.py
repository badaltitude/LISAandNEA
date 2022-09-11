#### Common functions for the LISA glitches pipelines (versions 4+)

from h5py import File
from ipynb.fs.defs import utils
import logging
import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos, pi, sqrt, dot
import os
from scipy.spatial.transform import Rotation as R
# from scipy.signal import welch

import lisaconstants
import lisainstrument
import lisagwresponse
import lisaglitch
import pytdi

from lisaconstants import ASTRONOMICAL_UNIT, GRAVITATIONAL_CONSTANT, SPEED_OF_LIGHT, GM_SUN
from lisainstrument import Instrument
from lisagwresponse import ReadStrain
from lisaglitch import StepGlitch, RectangleGlitch, ShapeletGlitch, IntegratedShapeletGlitch
from lisaglitch import TimeSeriesGlitch
from lisaglitch import OneSidedDoubleExpGlitch, TwoSidedDoubleExpGlitch 
if lisaglitch.__version__=='1.1': from lisaglitch import IntegratedOneSidedDoubleExpGlitch, IntegratedTwoSidedDoubleExpGlitch
from pytdi import Data, michelson, ortho


def check_versions():
    # only use a known combination of packages is installed or it could get messy
    assert(
            (
            (lisaconstants.__version__=='1.2') and
            (lisaglitch.__version__=='1.0') and
            (lisagwresponse.__version__=='1.0.1') and 
            (lisainstrument.__version__=='1.0.4') and
            (pytdi.__version__=='1.2')
            ) 
        or
            (
            (lisaconstants.__version__=='1.2') and
            (lisaglitch.__version__=='1.1') and
            (lisagwresponse.__version__=='1.1') and 
            (lisainstrument.__version__=='1.0.7') and
            (pytdi.__version__=='1.2')        
            )
        )


def estimate_transformation(orbits_path, index=0, gw_beta=0., gw_lambda=0.):

    # Unit vector in the direction of GW wave propgation
    k = np.array([-cos(gw_beta)*cos(gw_lambda),
        -cos(gw_beta)*sin(gw_lambda),
        -sin(gw_beta)
        ])

    # XYZ coordinates of spacecraft 1 at the start of the orbits file
    coord_sc1_0 = utils.get_pos(orbits_path, sc=1, index=index)
    coord_sc2_0 = utils.get_pos(orbits_path, sc=2, index=index)
    coord_sc3_0 = utils.get_pos(orbits_path, sc=3, index=index)

    # calculate the simulated TCB difference between time of wavefront arriving at sun and time at spacecraft
    time_at_sc1 = dot(coord_sc1_0,k)/SPEED_OF_LIGHT
    time_at_sc2 = dot(coord_sc2_0,k)/SPEED_OF_LIGHT
    time_at_sc3 = dot(coord_sc3_0,k)/SPEED_OF_LIGHT
    
    # return the transformation which will give the earliest response at LISA constellation
    return min(time_at_sc1, time_at_sc2, time_at_sc3)


def make_glitch(glitch_type, level=3e-9, width=2, beta=1, quantum_n=1, t_rise=1, t_fall=1, displacement=1e-9, inj_point='tm_12', t_inj=10000, duration=25000, fs=16, t0=0):
    '''Creates one glitch of glitch_type in inj_point
    
    Args:
        glitch_type (str): type
        width (float): width of the rectangle (glitch duration) [s]
        level (float): amplitude [injection point unit]
        beta (float): damping time [s]
        quantum_n (int): number of shapelet components (quantum energy level)
        inj_point (str): tm_ij injection point, c.f. lisaglitch.INJECTION_POINTS [m s-1]
        t_inj (float): injection time [s]
        duration (float): simulation duration [s]
        fs (float): simulation sampling frequency [Hz]
        t0 (float): simulation initial time [s]
    
    Returns:
        the glitch
    '''
    all_inj_points = ['tm_12', 'tm_23', 'tm_31', 'tm_13', 'tm_32', 'tm_21']
    dt = 1/fs # simualtion sampling interval [s]
    size = duration*fs # simulation size [samples]
    glitch_type = glitch_type.lower()

    if inj_point=='random':
        inj_point = np.random.choice(all_inj_points)
    elif inj_point not in all_inj_points:
        raise ValueError('Not a valid injection point')

    if glitch_type=='StepGlitch'.lower():
        g = StepGlitch(level=level,
                inj_point=inj_point,
                t_inj=t_inj,
                dt=dt,
                size=size,
                t0=t0)
    elif glitch_type=='RectangleGlitch'.lower():
        g = RectangleGlitch(level=level,
                width=width,
                inj_point=inj_point,
                t_inj=t_inj,
                dt=dt,
                size=size,
                t0=t0)
    elif glitch_type=='ShapeletGlitch'.lower():
        g = ShapeletGlitch(level=level,
                beta=beta,
                quantum_n=quantum_n,
                inj_point=inj_point,
                t_inj=t_inj,
                dt=dt,
                size=size,
                t0=t0)
    elif glitch_type=='IntegratedShapeletGlitch'.lower():
        g = IntegratedShapeletGlitch(level=level/2.*sqrt(beta),
                beta=beta,
                inj_point=inj_point,
                t_inj=t_inj,
                dt=dt,
                size=size,
                t0=t0)
    elif glitch_type=='OneSidedDoubleExpGlitch'.lower():
        g = OneSidedDoubleExpGlitch(t_rise=t_rise, 
                t_fall=t_fall, 
                level=level, 
                inj_point=inj_point,
                t_inj=t_inj,
                dt=dt,
                size=size,
                t0=t0)
    elif glitch_type=='TwoSidedDoubleExpGlitch'.lower():
        g = TwoSidedDoubleExpGlitch(t_rise=t_rise, 
                t_fall=t_fall, 
                level=level, 
                displacement=displacement,
                inj_point=inj_point,
                t_inj=t_inj,
                dt=dt,
                size=size,
                t0=t0)
    else:
        raise ValueError('Not a supported glitch type')
    
    # return the glitch
    return g


# Functions for the TM's extra velocity components (XYZ axes)
# XYZ axes are defined for the spacecraft

def vx(t, nea):
    'Returns a TM velocity component along local X axis over time vector t'
    M = nea['M']
    D = nea['D']
    V = nea['V']
    vx = GRAVITATIONAL_CONSTANT*M/(D*V)*(1+(V*t/D)/sqrt(1+V**2*t**2/D**2))
    return vx


def vy(t, nea):
    'Returns a TM velocity component along local Y axis over time vector t'
    M = nea['M']
    D = nea['D']
    V = nea['V']
    return 0*t


def vz(t, nea):
    'Returns a TM velocity component along local Z axis over time vector t'
    M = nea['M']
    D = nea['D']
    V = nea['V']
    vz = GRAVITATIONAL_CONSTANT*M/(D*V)*(-1/sqrt(1+V**2*t**2/D**2))
    return vz


# Functions for the TM's extra velocity components (UVZ axes)
# U and V axes defined differently for each test mass by rotation beta around Z

def get_beta(tm):
    '''Returns the angle (beta) between the TM's laser direction and spacecraft's local X axis 
    
    Args:
        tm (int): test mass index (1 or 2)
        
    Returns: 
        beta (radians): angle'''

    if tm == 1: 
        return pi/6
    else: 
        return -pi/6
    

def vu(tm, t, nea):
    '''Returns the TM velocity component along U axis defined by the laser direction
    
    Args:
        tm (int): test mass index (1 or 2)
        t (array): time vector
    '''
    beta = get_beta(tm)
    return cos(beta)*vx(t, nea) - sin(beta)*vy(t, nea)


def vv(tm, t, nea):
    '''Returns the TM velocity component along V axis, perpendicular to laser (U) and local Z axis
    
    Args:
        tm (int): test mass index (1 or 2)
        t (array): time vector
    '''
    beta = get_beta(tm)
    return -sin(beta)*vx(t, nea) - cos(beta)*vy(t, nea)


# Functions for the TM's extra velocity vector (UVZ axes)

def calc_tm_velocity(tm, t, nea):
    '''Returns the velocity components along axes U, V, Z relative to the TM's laser direction
    
    Args:
        tm (int): test mass index (1 or 2)
        t (array): time vector
    '''
    return np.concatenate(
    [
        vu(tm=tm, t=t, nea=nea)[:, np.newaxis],
        vv(tm=tm, t=t, nea=nea)[:, np.newaxis],
        vz(t=t, nea=nea)[:, np.newaxis]
    ], 
    axis=1
)


def make_neaglitches(nea={'M':1e14, 'D':1e4, 'V':6.8e4, 'angles':[0,0,0]}, t_inj=10000, t0=0, duration=25000, fs=16, sc=1):
    '''Make 2 NEA glitches in the local test masses
    
    Args:
        nea (dict): Dictionary of NEA parameters M, D, V and angles
            M (float): NEA mass [kg]
            D (float): impact parameter [m]
            V (float): relative velocity [m s-1]
            angles (list): Rotations around spacecraft's XYZ axes as chi, psi, omega [degrees]
        t_inj (float): injection time [s]
        t0 (float): simulation initial time [s]
        duration (float): simulation duration [s]
        fs (float): simulation sampling frequency [Hz]
        sc (int): index of the encountered spacecraft 
    
    Returns:
        glitches (list): 2 glitches in the spacecraft's local TMs
    '''
    M = nea['M']
    D = nea['D']
    V = nea['V']
    glitch_factor=20
    glitch_duration = glitch_factor*D/V
    glitch_window = utils.round_up(glitch_duration)
    print(f'Duration ({glitch_factor}x) of NEA encounter is +/- {glitch_duration:.2f} s ({glitch_window} s) around time of closest approach')
    dt = 1/fs # sampling interval [s]
    size = duration*fs # simulation size [samples]

    # time vector is centered on time of closest approach, t0=0
    t = np.linspace(-glitch_window, glitch_window, 2*glitch_window*fs+1, endpoint=True)
    print(f'Time vector covers ({min(t)} to {max(t)}) with length {len(t)}')

    # here's where the magic happens
    # lots of optimisation possible here, only need vu1 and vu2 for the glitches 
    tm1_velocity = calc_tm_velocity(tm=1, t=t, nea=nea)
    tm2_velocity = calc_tm_velocity(tm=2, t=t, nea=nea)
    my_rotation = R.from_euler('ZYX', angles=nea['angles'], degrees=True)
    tm1_velocity_rot = my_rotation.apply(tm1_velocity)
    tm2_velocity_rot = my_rotation.apply(tm2_velocity)
    # print(f'TM1 velocity covers ({min(tm1_velocity[:,0])} to {max(tm1_velocity[:,0])}) with length {len(tm1_velocity[:,0])}')
    # print(f'TM2 velocity covers ({min(tm2_velocity[:,0])} to {max(tm2_velocity[:,0])}) with length {len(tm2_velocity[:,0])}')
    # print(f'TM1 rotated velocity covers ({min(tm1_velocity_rot[:,0])} to {max(tm1_velocity_rot[:,0])}) with length {len(tm1_velocity_rot[:,0])}')
    # print(f'TM2 rotated velocity covers ({min(tm2_velocity_rot[:,0])} to {max(tm2_velocity_rot[:,0])}) with length {len(tm2_velocity_rot[:,0])}')
    # print(t_inj, dt, size)
    # make a new time vector with same length but starting at zero at t_inj
    # t = t+int(glitch_duration)
    # t = np.linspace(0, (len(tm1_velocity_rot)-1)*dt, len(tm1_velocity_rot), endpoint=True)
    # print(f'Time vector covers ({min(t)} to {max(t)}) with length {len(t)}')
    
    # select the injection points based on the spacecraft
    inj_point1, inj_point2 = get_inj_points(sc)
    
    # make the TimeSeries glitches
    g1 = TimeSeriesGlitch(t=t, tseries=tm1_velocity_rot[:,0], interp_order=1, ext='const', inj_point=inj_point1, t_inj=t_inj, dt=dt, size=size, t0=t0)
    g2 = TimeSeriesGlitch(t=t, tseries=tm2_velocity_rot[:,0], interp_order=1, ext='const', inj_point=inj_point2, t_inj=t_inj, dt=dt, size=size, t0=t0)
    
    return [g1,g2]


def get_inj_points(sc=1):
    # select the injection points based on the spacecraft
    if sc==1: 
        return 'tm_12', 'tm_13'
    elif sc==2:
        return 'tm_23', 'tm_21'
    elif sc==3:
        return 'tm_31', 'tm_32'
    else:
        raise ValueError('Invalid spacecraft specificed')


def make_instrument(duration=25000, t0=None, fs=4, aafilter=('kaiser', 240, 1.1, 2.9), orbits='static', gws=None, glitches=None, lock='N1-12'):
    '''Initialise a new LISA instrument (simulation)
    
    Args:
        duration (float): simulation duration [s]
        t0 (float): simulation start time [s]
        aafilter (tuple): antialiasing filter function (see lisainstrument.Instrument)
        orbits (str): path to orbit file, 'rigid' for dictionary of constant PPRs (8.33 s), 'static'
            for a set of static PPRs corresponding to a fit of Keplerian orbits around t = 0
        gws (str): path to gravitational-wave file
        glitches (str): path to glitch file
        lock (str): pre-defined laser locking configuration (e.g. 'N1-12' non-swap N1 with 12 primary 
            laser) or 'six' for 6 lasers locked on cavities

    Returns:
        an initialised instrument
    '''    
    dt = 1/fs
    size = duration*fs

    if t0 is None: t0='orbits'

    if orbits=='rigid':
        # dictionary of constant PPRs to simulate a rigid constellation
        ppr = 2.5e9/SPEED_OF_LIGHT # LISA arm length = 2.5 million km
        orbits = {'12': ppr, '23': ppr, '31': ppr, '13': ppr, '32': ppr, '21': ppr}

    return Instrument(size=size, 
                        dt=dt, 
                        t0=t0, 
                        physics_upsampling=4, 
                        aafilter=aafilter, 
                        orbits=orbits, 
                        orbit_dataset='tps/ppr', 
                        gws=gws, 
                        interpolation=('lagrange', 31), 
                        glitches=glitches,  
                        lock=lock, 
                        offsets_freqs='default', 
                        laser_asds=28.2, 
                        central_freq=281600000000000, # 281,600,000,000,000
                        modulation_asds='default', 
                        modulation_freqs='default', 
                        tdir_tone=None, 
                        clock_asds=6.32e-14, 
                        clock_offsets=0, 
                        clock_freqoffsets='default', 
                        clock_freqlindrifts='default', 
                        clock_freqquaddrifts='default', 
                        clockinv_tolerance=1e-10, 
                        clockinv_maxiter=5, 
                        backlink_asds=3e-12, 
                        backlink_fknees=0.002, 
                        testmass_asds=2.4e-15, 
                        testmass_fknees=0.0004, 
                        oms_asds=(6.35e-12, 1.25e-11, 1.42e-12, 3.38e-12, 3.32e-12, 7.9e-12), 
                        oms_fknees=0.002, 
                        ttl_coeffs='default', 
                        sc_jitter_asds=(1e-8, 1e-8, 1e-8), 
                        mosa_jitter_asds=1e-8, 
                        mosa_angles='default', 
                        dws_asds=7e-8 / 335, 
                        ranging_biases=0, 
                        ranging_asds=3e-9)


def make_combinations(data, type='michelson', order=1, verbose=False):
    '''Compute the Michelson and quasi-orthoginal TDI combinations
    
    Args:
        data: data extracted from a measurements file
        type (str): 'michelson' or 'ortho'
        order (int): the order of the TDI combination
        
    Returns:
        Tuple of numpy arrays for the 3 TDI combinations: XYZ or AET
    '''
    if (type=='michelson') & (order==1):
        if verbose: print('Evaluating 1st order Michelson combinations')
        data_X1 = michelson.X1.build(**data.args)(data.measurements)
        data_Y1 = michelson.Y1.build(**data.args)(data.measurements)
        data_Z1 = michelson.Z1.build(**data.args)(data.measurements)
        return data_X1, data_Y1, data_Z1
    elif (type=='michelson') & (order==2):
        if verbose: print('Evaluating 2nd order Michelson combinations')
        data_X2 = michelson.X2.build(**data.args)(data.measurements)
        data_Y2 = michelson.Y2.build(**data.args)(data.measurements)
        data_Z2 = michelson.Z2.build(**data.args)(data.measurements)
        return data_X2, data_Y2, data_Z2
    elif (type=='ortho') & (order==1):
        if verbose: print('Evaluating 1st order orthogonal combinations')
        data_A1 = ortho.A1.build(**data.args)(data.measurements)
        data_E1 = ortho.E1.build(**data.args)(data.measurements)
        data_T1 = ortho.T1.build(**data.args)(data.measurements)
        return data_A1, data_E1, data_T1
    elif (type=='ortho') & (order==2):
        if verbose: print('Evaluating 2nd order orthogonal combinations')
        data_A2 = ortho.A2.build(**data.args)(data.measurements)
        data_E2 = ortho.E2.build(**data.args)(data.measurements)
        data_T2 = ortho.T2.build(**data.args)(data.measurements)
        return data_A2, data_E2, data_T2
    else:
        raise ValueError('Unknown type and/or order')


def make_gwburst(width=2, level=3e-20, t_inj=10000, orbits_t0=12160, duration=25000, fs=16, model='Rectangle', sigma=1):
    
    # Waveform parameters for the GW (used for both hplus and hcross)
    
    model = model.lower()
    t = np.linspace(orbits_t0, orbits_t0+duration, duration*fs)

    if model=='Rectangle'.lower():
        
        # Reusing the definition from LISA Glitch's RectangleGlitch()
        inside = np.logical_and(t >= t_inj, t < t_inj + width)
        rectangle = np.where(inside, level, 0)

        hplus = rectangle
        hcross = rectangle

    elif model=='Gaussian'.lower():
        
        # make a gaussian waveform for the GW
        # Examples from:
        #  https://github.com/jmcginn/McGANn/blob/master/cGAN/GAN_training_data.py
        #  https://labcit.ligo.caltech.edu/~ajw/bursts/burstsim.html
        h_1 = level * np.exp(-(t-t_inj)**2/(2*sigma**2))

        hplus = h_1
        hcross = h_1

    else:
        ValueError('Unknown GW burst model type')

    return t, hplus, hcross


def save_glitch_file(glitches, path):
    delete_file_if_exists(path)
    if lisaglitch.__version__=='1.0':
        for g in glitches: g.write(path)
    else:
        for g in glitches: g.write(path, mode='a')
    return path


def save_gws_file(source, path):
    delete_file_if_exists(path)
    source.write(path)
    return path


def save_measurements_file(instru, path):
    delete_file_if_exists(path)
    instru.write(path, 'w')
    return path


def delete_file_if_exists(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass