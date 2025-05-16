import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import time,glob
import os
import sys, random
import time
from pytest import skip
from os import listdir,walk
from os.path import isfile,join
from natsort import natsorted


def listfiles(index):
    d = os.getcwd()+'/'
    myfiles = []
    files = [f for f in listdir(d) if isfile(join(d, f))]
    for i in files:
        if index in i:
            myfiles.append(d+i)
    return myfiles

def getimage(x):
    d = os.getcwd()+'/'
    file = files(x,'.fits')[0] 
    image = fits.getdata(file)
    return image

def image_fits(x):
    d = os.getcwd()+'/'
    file = d+x
    image = fits.getdata(file)
    return image

def pulltimes(files):
    times = []
    n = len(files)
    sortedfiles = natsorted(files)
    data0 = fits.open(sortedfiles[0])
    data0time = data0[0].header['DATE-OBS']
    time0 = Time(data0time,format='fits')
    jdtime0 = time0.jd
    for i in range(n):
        data = fits.open(sortedfiles[i])
        datetime = data[0].header['DATE-OBS']
        time = Time(datetime,format='fits')
        jdtime = (time.jd - jdtime0)*86400
        times.append(jdtime)
    return times

def pulltimes_aia(files):
    times = []
    n = len(files)
    sortedfiles = natsorted(files)
    data0 = fits.open(sortedfiles[0])
    data0time = data0[1].header['DATE-OBS']
    time0 = Time(data0time,format='fits')
    jdtime0 = time0.jd
    for i in range(n):
        data = fits.open(sortedfiles[i])
        datetime = data[1].header['DATE-OBS']
        time = Time(datetime,format='fits')
        jdtime = (time.jd - jdtime0)*86400
        times.append(jdtime)
    return times

def aia_datetimes(files):
    times = []
    sortedfiles = natsorted(files)
    n = len(files)
    for i in range(n):
        data = fits.open(sortedfiles[i])
        datetime = data[1].header['DATE-OBS']
        times.append(datetime)
    return times

def aia_timediffs(files):
    times = []
    time_diffs = []
    times_over = []
    sortedfiles = natsorted(files)
    n = len(files)
    data0 = fits.open(sortedfiles[0])
    data0time = data0[1].header['DATE-OBS']
    time0 = Time(data0time,format='fits')
    jdtime0 = time0.jd
    for i in range(n):
        data = fits.open(sortedfiles[i])
        datetime = data[1].header['DATE-OBS']
        time = Time(datetime.format('fits'))
        jdtime = (time.jd - jdtime0)*86400
        times.append(jdtime)
    for j in range(n):
        if j == 0:
            pass
        elif j == n:
            pass
        else:
            k = j - 1
            t = times[j] - times[k]
            #if t > 13:
            #    times_over.append(j)
            #time_diffs.append(t)
            time_diffs.append(t)
    return time_diffs

def aia_exptime(files):
    exptimes = []
    sortedfiles = natsorted(files)
    n = len(files)
    for i in range(n):
        data = fits.open(sortedfiles[i])
        exptime = data[1].header['EXPTIME']
        exptimes.append(exptime)
    return exptimes

def obsdate(index):
    d = os.getcwd()+'/'
    file = files(d,index,'.fits')[0]
    data = fits.open(file)
    dateobs = data[0].header['DATE-OBS']
    return dateobs

def imstretch_suvi(image,cmap_input):
    image = np.flip(image,axis=0)
    med = np.median(image)
    std = np.std(image)
    vminimum = med-std
    vmaximum = med+std
    plt.imshow(image,vmin=vminimum,vmax=vmaximum, cmap = str(cmap_input))
    plt.show()

def exptime(x):
    d = os.getcwd() + '/'
    file = files(d,x,'.fits')[0]
    data = fits.open(file)
    time = data[0].header['exptime']
    return time

def header(file):
    hdu1 = fits.open(file)
    hdr = hdu1[0].header
    print(hdr)

def uniquetimes(x):
    d = os.getcwd() + '/'
    filelist = files(x,'.fits')
    times = []
    for i in filelist:
        print(i)
        data = fits.open(i, ignore_missing_simple=True)
        cmd_exp = data[1].header['EXPTIME']
        times.append(cmd_exp)
    return np.unique(times)   

def separate_times(files, x):
    for i in files:
        data = fits.open(i, ignore_missing_simple=True)
        cmd_exp = data[1].header['EXPTIME']
        if cmd_exp < x == True:
            new_name = 'short_'+str(i)
            os.rename(i, new_name)
        else:
            new_name = 'long_'+str(i)  
            os.rename(i, new_name)
        print('Files renamed')

def suvitimes(x):
    d = os.getcwd() + '/'
    filelist = files('SUVI','fits')
    files1 = []
    for i in filelist:
        data = fits.open(i)
        cmd_exp = data[0].header['CMD_EXP']
        if cmd_exp == x:
            files1.append(i)
    return files1

def suvitimes_files(x,y,t):
    file_list = files(x,y)
    file_list2 = []
    for i in file_list:
        data = fits.open(i)
        t_0 = data[0].header['CMD_EXP']
        if t_0 == t:
            file_list2.append(i)
    return file_list2

def get_suvi_date(file):
    d = os.getcwd() + '/'
    data = fits.open(file)
    date_obs = data[0].header['DATE-OBS']
    return date_obs

def files(x,y):
    d = os.getcwd()+'/'
    myfiles = []
    files = [f for f in listdir(d) if isfile(join(d, f))]
    for i in files:
        if x.lower() in i.lower() and y.lower() in i.lower():
            myfiles.append(i)
    return myfiles