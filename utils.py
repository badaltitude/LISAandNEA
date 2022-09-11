### Consolidated common imports
import base64
from calendar import monthrange
from h5py import File
from io import BytesIO
from ipynb.fs.defs import utils
from IPython.display import Image, display
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos, tan, arcsin, arccos, arctan, pi, sqrt, dot
import os
import pandas as pd
import PIL
import PIL.Image
import requests
from scipy.signal import welch
from scipy.spatial.transform import Rotation as R
from scipy.stats import zscore

import warnings
warnings.simplefilter('ignore', FutureWarning)

### Common but un-used imports
# import math as m

### Less common imports
# import tensorflow as tf
# from sklearn import preprocessing


def hello_world(who):
    print(f'hello {who}')


def create_directory(directory_path):
    '''Create a new directory, returns None if directory_path already exists or an error occurs'''
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            return None
        return directory_path


def get_output_file_name(output_dir, output_file, file_type):
    return f'{output_dir}/{output_file}.{file_type}'


def describe_hdf5(path):
    '''Helper function to describe the structure of an HDF5 file'''
    
    with File(path, 'r') as hdf5:
        print('> Root level attributes (file.attrs) ***')
        for key in hdf5.attrs:
            print(f'{key}')
        print('> Root level datasets (file.keys) ***')
        for dataset in hdf5.keys():     
            print(f'{dataset}')
            if dataset == 't':
                print(f' {hdf5[dataset][:-3]}')
            else:
                print(' > Attributes (file[dataset].attrs[key]) ***')
                for key in hdf5[dataset].attrs:
                    print(f' {key} : {hdf5[dataset].attrs[key]}')
                print(' > Datasets (file[dataset].keys) ***')
                for dataset2 in hdf5[dataset].keys():
                    print(f' {dataset2}')


def describe_instrument(i, skip=1000, output='all'):
    '''Helper function to describe a simulated LISA instrument'''

    # print about info
    print(f'LISA Instrument')
    print(f' Start: {i.t0} s')
    print(f' Duration: {i.duration} s')
    print(f' Sampling frequency: {i.fs} Hz')
    print(f' Sampling interval: {i.dt} s')
    print(f' Size: {i.size} samples')
    try:
        lock_config_N, lock_config_LA = i.lock_config
        print(f' Locking config: {lock_config_N}-LA{lock_config_LA}')
    except:
        print(f' Locking config: {i.lock_config}')
    
    if output in ['all','graph']:
        # plot simulated instrument
        i.plot_offsets(skip=skip)
        i.plot_fluctuations(skip=skip)
        i.plot_totals(skip=skip)
        i.plot_mprs(skip=skip)


def describe_glitch(g):
    '''Helper function to describe a Glitch'''
    
    # print about info
    print(f'Glitch')
    print(f' Start: {g.t0} s')
    print(f' Duration: {g.duration} s')
    print(f' Sampling frequency: {g.fs} Hz')
    print(f' Sampling interval: {g.dt} s')
    print(f' Size: {g.size} samples')
    print(f' Injection point: {g.inj_point}')

    # plot simulation data around glitch
    g.plot()
    
    # plot whole simulation data (default simulation duration determined by size=2592000 and dt=0.0625)
    g.plot(tmin=g.t0, tmax=g.t0+g.duration)

    return None


def round_up(number): 
    '''Round up a number to next largest integer

    >>> round_up(-15.3), round_up(-0.1), round_up(0.), round_up(0.2), round_up(15.3)
    (16, -1, 0, 1, -16)
    
    Args:
        number (float or int)
        
    Returns:
        integer (int)
    '''
    if number>=0:
        return int(number) + (number % 1 > 0)
    else:
        return int(number) - (number % 1 > 0)


def get_attr(orbits, attr):
    '''Get the value of a named attribute from a LISA orbits file
    
    Args:
        orbits (str): path to the orbits file
        attr (str): name of attribute

    Returns:
        value of the named attribute 
    '''
    with File(orbits, 'r') as orbitf:
        return orbitf.attrs[attr]


def get_pos(orbits, sc=1, index=0):
    '''Get a spacecraft's position from a LISA orbits file
    
    Args: 
        orbits (str): path to the orbits file
        sc (int): index of the spacecraft
        index (int): index of the time vector. Note sampling interval can be large (~hours)

    Returns:
        position (tuple): x, y, z 
    '''
    with File(orbits, 'r') as orbitf:
        x = orbitf[f'tcb/sc_{sc}']['x'][index]
        y = orbitf[f'tcb/sc_{sc}']['y'][index]
        z = orbitf[f'tcb/sc_{sc}']['z'][index]
    return x,y,z


def describe_df(df):
    '''Describe a dataframe
    
    Args:
        dataframe (dataframe): A dataframe
    
    Returns:
        Another dataframe describing the numeric fields
    '''
    # Strip non-numerics
    df = df.select_dtypes(include=['int', 'float'])

    headers = list(df.columns.values)
    fields = []

    for field in headers:
        fields.append({
            'name' : field,
            'min': df[field].min(),
            'max': df[field].max(),
            'mean': df[field].mean(),
            'var': df[field].var(),
            'sdev': df[field].std()
        })

    return pd.DataFrame(fields)


def plot_strain(t, hplus, hcross, axs, row=0, col=0):
    '''Plot the strain'''
    axs[row, col].plot(t, hplus, label=r'$h_+$')
    axs[row, col].plot(t, hcross, label=r'$h_\times$')
    axs[row, col].set(ylabel='Strain')
    axs[row, col].legend(ncol=2)


def plot_gws(t, y, axs, row=0, col=0):
    '''Plot the link responses to the GW'''
    LINKS = ['12', '23', '31', '13', '32', '21']
    for i in range(len(LINKS)): axs[row, col].plot(t, y[i], label=f'{LINKS[i]}')
    axs[row, col].set(ylabel='Link response')


def plot_blank(axs, row=0, col=0):
    '''Plot the glitches'''
    axs[row, col].plot(0, 1e-20, label=f' ')
    axs[row, col].set(ylabel='Blank')
    return


def format_axs(axs):
    '''Apply common formatting to the matplotlib axs'''
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i,j].grid()
            axs[i,j].legend(loc='upper right')
            # axs[i,j].set(xlim=(12110,12360))
            if i == axs.shape[0]-1:
                axs[i,j].set(xlabel='Time (s)')


def format_fig(fig, axs):
    '''Format the matplotlib figure and axs'''
    format_axs(axs)
    # fig.text(0.5, 0, 'Time (s)', ha='center')
    # fig.text(-0.0, 0.5, 'TDI combinations', va='center', rotation='vertical')
    fig.patch.set_alpha(1)
    fig.tight_layout()


def plot_glitches(t, glitches, axs, row=0, col=0):
    '''Plot the TM glitches'''
    for g in glitches: axs[row, col].plot(t, g.compute_signal(t), label=f'{g.inj_point}')
    axs[row, col].set(ylabel='TM velocity (m s⁻¹)')

def plot_combos(t, combo0, combo1, combo2, labels, axs, row, col=0, ylabel='TDI combos'):
    '''Plot a triplet of TDI combinations on one axis'''      
    axs[row, col].plot(t, combo0, label=labels[0])
    axs[row, col].plot(t, combo1, label=labels[1])
    axs[row, col].plot(t, combo2, label=labels[2])
    axs[row, col].set(ylabel=ylabel) 