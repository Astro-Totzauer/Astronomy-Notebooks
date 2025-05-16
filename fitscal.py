import numpy as np
from astropy.io import fits
from astropy.time import Time
import time,glob
import os
from os import listdir,walk
from os.path import isfile,join
import matplotlib
import matplotlib.image as mgimg
from pylab import *
from random import randint
import matplotlib.pyplot as plt
import scipy as sp
import scipy.ndimage
import shutil
import csv
import pandas as pd
import string
import scipy
from scipy import signal,ndimage
import sunpy.map
from sunpy.image.coalignment import calculate_match_template_shift as mc_shift 
from sunpy.image.coalignment import apply_shifts as mc_apply_shifts
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons
from astropy.visualization import ContrastBiasStretch,ManualInterval,LinearStretch,MinMaxInterval,ImageNormalize, SqrtStretch,LogStretch,PowerDistStretch,PowerStretch,SinhStretch,SquaredStretch,AsinhStretch,PercentileInterval
from scipy.ndimage.interpolation import shift
import sunpy.io.fits as sunfits
from natsort import natsorted
from collections import Counter
import sys
from astropy import units as u
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft, AiryDisk2DKernel, Box2DKernel, MexicanHat2DKernel, Ring2DKernel, Tophat2DKernel, TrapezoidDisk2DKernel
from skimage.transform import warp
from skimage.transform import ProjectiveTransform
import argparse
from collections import deque
import imageio
from sunpy.visualization import animator as imageanimator
plt.rcParams['animation.ffmpeg_path'] = '/opt/local/bin/ffmpeg'

def quakes(file):
    years = []
    events = []
    df = pd.read_csv(file)
    df['time'] = pd.to_datetime(df['time'])
    group = df['time'].dt.year
    counts = Counter(group)
    counts2 = list(counts.items())
    n = len(counts2)
    for i in range(n):
        year = counts2[i][0]
        event = counts2[i][1]
        years.append(year)
        events.append(event)
    return years, events
    
    

# Reads in a list of files, generates a data cube.
def cube(files):
    n = len(files)
    dataz = np.zeros((1280,1280,n))
    for i in range(len(files)):
        data = fits.getdata(files[i])
        dataz[:,:,i] = data
    return dataz

def imcube(images):
    n = len(images)
    y = images[0].shape[0]
    x = images[0].shape[1]
    dataz = np.zeros((x,y,n))
    for i in range(len(files)):
        dataz[:,:,i] = images[i]
    return dataz

def csvread(file):
    datafile = open(file,'r')
    myreader = csv.reader(datafile)
    text = []
    for row in myreader:
        text.append(row)
    return text


def fix(file):
    with open(file, 'r') as f_in:
        data = f_in.read().splitlines(True)
    with open(file, 'w') as f_out:
        f_out.writelines(data[17:-1])

def clean(x,y):
    d = os.getcwd()+'/'
    for i in files(d,x,y):
        fix(i)

def spdata(file):
    data = np.genfromtxt(file)
    return data

def spmax(data):
    peakind = signal.find_peaks_cwt(data[:,1],np.arange(30,60))
    index = np.array(peakind)
    peaks = []
    for i in peakind:
        y = data[:,1][i]
        peaks.append(y)
    new = np.column_stack([index,peaks])
    new2 = new[new[:,1].argsort()[::-1]]
    return new2

def spcor(data,x):
    cordata = data + [x]
    return cordata

def spplot(data):
    x = data[:,0]
    y = data[:,1]
    plt.plot(x,y)
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Intensity')
    plt.show()

def spfull(file,x):
    data = spdata(file)
    maxpoints = spmax(data)
    corrected = spcor(maxpoints,x)
    spplot(data)
    return corrected

def sppeaks(file,x):
    data = spfull(file,x)
    return ((0.43945*data[:,0])+200)*10
    

# Generates a list of file names. 'd' = directory path,
# 'index' = keyword in file names to search for.

def listfiles(index):
    d = os.getcwd()+'/'
    myfiles = []
    files = [f for f in listdir(d) if isfile(join(d, f))]
    for i in files:
        if index in i:
            myfiles.append(d+i)
    return myfiles
    
def rotsub(shifts,times,slope):
    new_vals = []
    new_shifts = shifts/u.arcsec
    n = len(shifts)
    for i in range(n):
        new_val = new_shifts[i] - (times[i]*slope)
        new_vals.append(new_val)
    return new_vals
    

# Loads fits data from directory and search term

def getimage(x):
    d = os.getcwd()+'/'
    file = files(x,'.fits')[0] 
    image = fits.getdata(file)
    return image

def mcslice(files):
    n = len(files)
    for i in range(n):
        image = sunpy.map.Map(files[i])
        data = image.data
        hdu = fits.open(files[i])
        hdr = hdu[0].header
        dslice = data[490:790,490:790]
        sunfits.write('temp'+str(i)+'.fits',dslice,hdr)

def cubeslice(cube,a,b,c,d):
    n = cube.shape[2]
    x = b-a
    y = d-c
    dataz = np.zeros((x,y,n))
    for i in range(n):
        image = cube[:,:,i]
        im_slice = image[a:b,c:d]
        dataz[:,:,i] = im_slice
    return dataz
        

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

def obsdate(index):
    d = os.getcwd()+'/'
    file = files(d,index,'.fits')[0]
    data = fits.open(file)
    dateobs = data[0].header['DATE-OBS']
    return dateobs
    
    
# Displays a stretched image

def imagestretch(image,cmap_input):
    med = np.median(image)
    std = 2*np.std(image)
    vminimum = med-std
    vmaximum = med+std
    plt.imshow(image,vmin=vminimum,vmax=vmaximum, cmap=str(cmap_input))
    plt.show()

def imstretch_suvi(image,cmap_input):
    image = np.flip(image,axis=0)
    med = np.median(image)
    std = np.std(image)
    vminimum = med-std
    vmaximum = med+std
    plt.imshow(image,vmin=vminimum,vmax=vmaximum, cmap = str(cmap_input))
    plt.show()

def stretchanimate(files,x):
    fig = plt.figure()
    n = len(files)
    images = []
    for i in range(n):
        image = fits.getdata(files[i])
        norm = ImageNormalize(image, interval=MinMaxInterval(),stretch=SqrtStretch())
        im = plt.imshow(image,norm=norm, animated=True,cmap = "gray")
        images.append([im])
    ani = animation.ArtistAnimation(fig,images,interval = x,blit=True)
    plt.show()    

def arraystretch(image,minimum,maximum):
    image = (image - minimum) / (maximum - minimum)
    return image

def normim(cube):
    n = cube.shape[2]
    x = cube.shape[0]
    y = cube.shape[1]
    dataz = np.zeros((x,y,n))
    for i in range(n):
        image = cube[:,:,i]
        norm = np.arctan(image)
        dataz[:,:,i] = norm
    return dataz

def animate_images(images):
    fig = plt.figure()
    n = len(images)
    imlist = []
    for i in range(n):
        image = images[i]
        im = plt.imshow(image, animated=True, cmap = 'gray')
        images.append([im])
    ani = animation.ArtistAnimation(fig,images,interval = 20, blit=True)
    plt.show()

def anicube_SG(cube,v_min,v_max):
    fig = plt.figure()
    n = cube.shape[0]
    images = []
    for i in range(n):
        image = np.flip(cube[i,:,:],axis=0)
        norm = ImageNormalize(image, interval=ManualInterval(vmin=v_min,vmax=v_max),stretch=ContrastBiasStretch(0.2,0.8))
        im = plt.imshow(image, norm=norm, animated=True, cmap = 'gray')
        images.append([im])
    ani = animation.ArtistAnimation(fig,images,interval = 50,blit=False)
    #cid = fig.canvas.mpl_connect('button_press_event', on_click)
    #Writer = animation.FFMpegWriter
    #writer = Writer(fps=30,bitrate=1800)
    #ani.save('demo.mp4',writer=writer)
    plt.show()

def anicube(cube,v_min,v_max):
    fig = plt.figure()
    n = cube.shape[2]
    images = []
    for i in range(n):
        image = np.flip(cube[:,:,i],axis=0)
        norm = ImageNormalize(image, interval=ManualInterval(vmin=v_min,vmax=v_max),stretch=ContrastBiasStretch(0.2,0.6))
        im = plt.imshow(image, norm=norm, animated=True, cmap = 'gray')
        images.append([im])
    ani = animation.ArtistAnimation(fig,images,interval = 50,blit=False)
    #cid = fig.canvas.mpl_connect('button_press_event', on_click)
    #Writer = animation.FFMpegWriter
    #writer = Writer(fps=30,bitrate=1800)
    #ani.save('demo.mp4',writer=writer)
    plt.show()

class animate_images(object):

    def __init__(self,data_set,v_min,v_max,dtype=cube):
        self.data_type = type(data_set)
        if self.data_type is np.ndarray:
            self.data = data_set
        if self.data_type is list:
            self.data = imcube(sorted(data_set))
        if self.data_type is sunpy.map.mapsequence.MapSequence:
            self.data = data_type.as_array()
        self.initial_image = 0
        self.ax_slider = plt.axes([0.25, 0.03, 0.50, 0.02])
        self.im_slider = Slider(self.ax_slider, 'Image', 0, 1, valinit = self.initial_image)
        self.is_manual = False
        self.v_min = v_min
        self.v_max = v_max
    
    def make_movie(self):
        images = []
        n = self.data.shape[2]
        for i in range(n):
            title = plt.text(1.5, -2.01, 'Image Index = '+str(i), horizontalalignment='center',verticalalignment='bottom')
            data = np.flip(self.data[:,:,i],axis=0)
            norm = ImageNormalize(data, interval=ManualInterval(vmin=self.v_min,vmax=self.v_max),stretch=SqrtStretch())
            image = plt.imshow(data,norm=norm,cmap='gray')
            images.append([image,title])
        return images

    def animate(self):
        images = self.make_movie()
        n = len(images)
        fig = plt.figure()
        axcolor = 'lightgoldenrodyellow'
        image_ax = plt.axes([0.25, 0.03, 0.50,0.02])
        image_slider = Slider(image_ax, 'Image',0,n,valinit=0,valstep=1)
        is_manual = False
        interval = 50
        def update_slider(val):
            global is_manual
            is_manual = True
            update(val)
        
            

            

        self.ani = animation.ArtistAnimation(self.fig, self.images, interval=200, blit=False, repeat = True)
        plt.show()


def anicube_sample(cube,n,v_min,v_max):
    fig = plt.figure()
    image = np.flip(cube[600:900,900:1280,n],axis=0)
    norm = ImageNormalize(image,interval=ManualInterval(vmin=v_min,vmax=v_max),stretch=SqrtStretch())
    plt.imshow(image,norm=norm,cmap='gray')
    plt.show()

def anicube_sample_step(cube,v_min,v_max, step, int_step):
    fig, ax = plt.subplots()
    pause = False
    n = cube.shape[2]
    n2 = range(n)
    n3 = n2[0::int(step)]
    images = []
    for i in n3:
        image = np.flip(cube[600:900,900:1280,i],axis=0)
        norm = ImageNormalize(image, interval=ManualInterval(vmin=v_min,vmax=v_max),stretch=ContrastBiasStretch(0.3,0.5))
        im = plt.imshow(image,norm=norm, animated=True, cmap = 'gray')
        images.append([im])
    ani = animation.ArtistAnimation(fig,images,interval = int(int_step),blit=False)
    fig.canvas.mpl_connect('button_press_event', onClick)
    plt.show()

class ani_test(object):

    def __init__(self, data_cube, v_min, v_max):
        self.cube = data_cube
        self.pause = False
        n = self.cube.shape[2]
        self.fig, ax = plt.subplots()
        ax.set_aspect("equal")
        self.movie = []
        #self.axamp = plt.axes([0.25,0.15,0.65,0.03])
        #self.axfreq = plt.axes([0.25,0.1,0.65,0.03])
        for t in range(n):
            # For Sept. 10 event, use 600:900, 900:1280
            image = np.flip(self.cube[600:900,900:1280,t],axis=0)
            norm = ImageNormalize(image, interval=ManualInterval(vmin=v_min,vmax=v_max),stretch=ContrastBiasStretch(0.3,0.5))
            im = plt.imshow(image,norm=norm, animated=True, cmap = 'gray')
            self.movie.append([im])
        #sfreq = Slider(self.axfreq, 'Freq', 0.1,30)
        #samp = Slider(self.axamp, 'Amp', 0.1, 10.0)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        self.ani = animation.ArtistAnimation(self.fig, self.movie, interval=100)    
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        self.ani.save('SUVI_131_Animation.mp4', writer=writer)
        plt.show()
    
    def onClick(self, event):
        if self.pause:
            self.ani.event_source.stop()
        else:
            self.ani.event_source.start()
        self.pause ^= True

class ani_test2(object):

    def __init__(self, data_cube,v_min,v_max):
        self.cube = data_cube
        self.pause = False
        n = self.cube.shape[2]
        self.fig,ax = plt.subplots()
        ax.set_aspect("equal")
        self.movie = []
        #self.axamp = plt.axes([0.25,0.15,0.65,0.03])
        #self.axfreq = plt.axes([0.25,0.1,0.65,0.03])
        for t in range(n):
            # For Sept. 10 event, use 600:900, 900:1280
            title = plt.text(1.5, 1.01, t, horizontalalignment='center',verticalalignment='bottom')
            image = np.flip(self.cube[600:900,900:1280,t],axis=0)
            norm = ImageNormalize(image, interval=ManualInterval(vmin=v_min,vmax=v_max),stretch=LogStretch())
            im = plt.imshow(image,norm=norm, animated=True, cmap = 'gray')
            self.movie.append([im,title])
        #sfreq = Slider(self.axfreq, 'Freq', 0.1,30)
        #samp = Slider(self.axamp, 'Amp', 0.1, 10.0)
        #Writer = animation.writers['ffmpeg']
        #writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        self.ani = animation.ArtistAnimation(self.fig, self.movie, interval=100)
        #self.ani2 = animation.ArtistAnimation(self.fig, self.movie2, interval=100)
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        #self.ani.save('SUVI_131_Animation_Gated.mp4', writer=writer)
        plt.show()

    def onClick(self, event):
        if self.pause:
            self.ani.event_source.stop()
        else:
            self.ani.event_source.start()
        self.pause ^= True

class ani_cubes(object):
    def __init__(self, cube1, cube2, v_min1, v_max1, v_min2, v_max2):
        self.cube1 = cube1
        self.cube2 = cube2
        self.v_min1 = v_min1
        self.v_min2 = v_min2
        self.v_max1 = v_max2
        self.v_max2 = v_max2
        self.pause = False
        
    def make_movie(self):
        n1 = self.cube1.shape[2]
        n1_list = np.arange(0,n1)
        n2 = self.cube2.shape[2]
        n2_list = np.arange(0,n2)
        n_array = np.array([n1,n2])
        n_max = np.max(n_array)
        for i in range(n_max):
            title = plt.text(1.5, 1.01, i, horizontalalignment='center',verticalalignment='bottom')
            image = np.flip(self.cube1[600:900,900:1280,i],axis=0)
            norm = ImageNormalize(image, interval=ManualInterval(vmin=self.v_min1,vmax=self.v_max1),stretch=SqrtStretch())
            image1 = plt.imshow(image,norm=norm, cmap = 'gray')
            return image1

    def make_movie2(self):
        images = []
        n1 = self.cube1.shape[2]
        n1_list = np.arange(0,n1)
        n2 = self.cube2.shape[2]
        n2_list = np.arange(0,n2)
        n_array = np.array([n1,n2])
        n_max = np.max(n_array)
        for i in range(n_max):
            if i in n1_list:
                title1 = plt.text(1.5, 1.01, i, horizontalalignment='center',verticalalignment='bottom')
                im1 = np.flip(self.cube1[600:900,900:1280,i],axis=0)
                norm1 = ImageNormalize(im1, interval=ManualInterval(vmin=self.v_min1,vmax=self.v_max1),stretch=SqrtStretch())
                image1 = self.ax1.imshow(im1,norm=norm1,cmap='gray')
            if i in n2_list:
                title2 = plt.text(1.5, 1.01, i, horizontalalignment='center',verticalalignment='bottom')
                im2 = np.flip(self.cube2[600:900,900:1280,i],axis=0)
                norm2 = ImageNormalize(im2, interval=ManualInterval(vmin=self.v_min1,vmax=self.v_max1),stretch=SqrtStretch())
                image2 = self.ax2.imshow(im2,norm=norm2,cmap='gray')
            images.append([image1, title1, image2, title2])
        return images
        

    def animate_cubes(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1,2)
        self.ax1.set_aspect("equal")
        self.ax2.set_aspect("equal")
        self.images = self.make_movie2()
        #self.ani = animation.FuncAnimation(fig, func = self.make_movie())
        self.ani = animation.ArtistAnimation(self.fig, self.images, interval=50, blit=False)
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        plt.show()

    def onClick(self, event):
        if self.pause:
            self.ani.event_source.stop()
        else:
            self.ani.event_source.start()
        self.pause ^= True
    
        
def cube_div(cube,array,sigma_x,sigma_y):
    n = cube.shape[2]
    x = cube.shape[0]
    y = cube.shape[1]
    sigma = [sigma_y,sigma_x]
    new_cube = np.zeros((x,y,n))
    smoothed_array = sp.ndimage.filters.gaussian_filter(array, sigma, mode='constant')
    for i in range(n):
        data = cube[:,:,i]
        new_data = data / smoothed_array
        new_cube[:,:,i] = new_data
    return new_cube

# Uses astropy to display the exposure time for a fits file.
# 'file' is a file name (need to include directory if outside PWD.

def exptime(x):
    d = os.getcwd() + '/'
    file = files(d,x,'.fits')[0]
    data = fits.open(file)
    time = data[0].header['exptime']
    return time

# Takes index "x" for searching, and returns the unique
# command exposure times used in the set of files searched

def uniquetimes(x):
    d = os.getcwd() + '/'
    filelist = files(d,x,'.fits')
    times = []
    for i in filelist:
        data = fits.open(i)
        cmd_exp = data[0].header['CMD_EXP']
        times.append(cmd_exp)
    return np.unique(times)

def cubealign(files):
    tmpcenter = cntrpix(files[0])
    n = len(files)
    dataz = np.zeros((1280,1280,n))
    for i in range(n):
        center = cntrpix(files[i])
        diffx = tmpcenter[0]-center[0]
        diffy =  tmpcenter[1]-center[1]
        data = fits.getdata(files[i])
        datay = np.roll(data, diffy, axis=0)
        datax = np.roll(datay,diffx, axis=1)
        dataz[:,:,i] = datax
    return dataz
        
def suvitimes(x):
    d = os.getcwd() + '/'
    filelist = files('SUVI','.fits')
    files1 = []
    for i in filelist:
        data = fits.open(i)
        cmd_exp = data[0].header['CMD_EXP']
        if cmd_exp == x:
            files1.append(i)
    return files1

def suvitimes_cropped(x):
    d = os.getcwd() + '/'
    filelist = files('Cropped','.fits')
    files1 = []
    for i in filelist:
        data = fits.open(i)
        cmd_exp = data[0].header['CMD_EXP']
        if cmd_exp == x:
            files1.append(i)
    return files1

def get_suvi_date(file):
    d = os.getcwd() + '/'
    data = fits.open(file)
    date_obs = data[0].header['DATE-OBS']
    return date_obs

def cntrpix(x):
    data = fits.open(x)
    crpix1 = data[0].header['CRPIX1']
    crpix2 = data[0].header['CRPIX2']
    return crpix1,crpix2        

# Returns the wavelength the image was taken in

def wavelength(x):
    d = os.getcwd() + '/'
    file = files(d,x,'.fits')[0]
    data = fits.open(file)
    wavelength = data[0].header['WAVELNTH']
    return wavelength

# Pulls the filter from a file's fits header

def filt(x):
    d = os.getcwd()+'/'
    file = files(d,x,'.fits')[0]
    data = fits.open(file)
    filtinfo = data[0].header['filter']
    return filtinfo

# Takes arguments 'd' = directory, 'index' = keyword in name to search for,
# 'dark' is the averaged dark file to calibrate with. 

def medflat(d,index,x,bias):
    filelist = files(d,index,x)
    n = len(filelist)
    dataz = np.zeros((1024,1024,n))
    for i in range(len(filelist)):
        data = fits.getdata(filelist[i])
        datadiv = (data-bias)/(np.median(data-bias))
        dataz[:,:,i] = datadiv
    nflats = np.median(dataz,axis=2)
    return nflats

# Just needs the directory (d) and assumes bias files include
# 'bias' in name

def medbias(d):
    files = listfiles(d,'Bias')
    nbias = np.median(cube(files),axis=2)
    return nbias

# Takes arguments 'd' = directory path and 'x' = exposure time.
# Example: normdark(d,15) will look for files with 'd15'

def meddark(d,y):
    filelist = files(d,'Darks','.fits')
    darks = []
    for f in filelist:
        t = fits.open(f)
        exptime = int(t[0].header['exptime'])
        if exptime == float(y):
            darks.append(f)
    darkstack = cube(darks)
    ndarks = np.median(darkstack,axis=2)
    return ndarks

# Scales a dark image to a new exposure time

def darkscale(d,oldtime,newtime):
    darkfiles = files(d,'Dark','.fits')
    darks = []
    for i in darkfiles:
        data = fits.open(i)
        time = data[0].header['exptime']
        if time == oldtime:
            darks.append(i)
    biasfiles = files(d,'Bias','.fits')
    biasdata = cube(biasfiles)
    bias = np.median(biasdata,axis=2)
    for f in range(len(darks)):
        data2,hd = fits.getdata(darks[f],header=True)
        datasub = data2 - bias
        dscale = datasub*int((newtime/oldtime))
        newdark = dscale + bias
        filename = darks[f].replace('Dark','DarkScale'+str(newtime))
        hd['exptime']=newtime
        fits.writeto(filename,newdark,header=hd)
    return 'Darks Scaled'

# Creates a list of exposure times used in a given set of images

def timetest(d,x):
    tfiles = files(d,x,'.fits')
    times = []
    for i in tfiles:
        t = fits.open(i)
        time = np.around(float((t[0].header['exptime'])),decimals=3)
        times.append(time)
    return np.unique(times)

# Creates a list of filters used in a given set of images

def filters(d,x):
    lights = files(d,x,'.fits')
    filts = []
    for i in lights:
        f = filt(i)
        filts.append(f)
    return np.unique(filts)

# Creates a list of file names given the unique identifiers index and y

def files(index,y):
    d = os.getcwd()+'/'
    myfiles = []
    files = [f for f in listdir(d) if isfile(join(d, f))]
    for i in files:
        if index.lower() in i.lower() and y.lower() in i.lower():
            myfiles.append(i)
    return myfiles

# Calibrates a given set of science images. Note the naming schemes used
# in the flat, dark and bias functions.

def calibrate(d,x):
    lights = files(d,x,'.fits')
    times = timetest(d,x)
    filts = filters(d,x)
    bias = medbias(d)
    for i in times:
        dark = meddark(d,i)
        filename = d+'NormDark.'+str(i)+'.fits'
        fits.writeto(filename,dark)
    for i in filts:
        flatfiles = files(d,'Flat','.fit')
        flatfiles2 = []
        for j in flatfiles:
            if filt(j) == i:
                flatfiles2.append(j)
        flat = medflat2(d,flatfiles2,bias)
        filename = d+'NormFlat.'+i+'.fits'
        fits.writeto(filename,flat)
    for i in lights:
        t = fits.open(i)
        time1 = int(t[0].header['exptime'])
        f = filt(i)
        data,hd = fits.getdata(i,header=True)
        dark = fits.getdata(d+'NormDark.'+str(time1)+'.fits')
        darksub = data-dark
        flat = fits.getdata(d+'NormFlat.'+f+'.fits')
        flatdiv = darksub/flat
        filename = i.replace('.fits','Cal'+str(time1)+'.fits')
        fits.writeto(filename,flatdiv,header=hd)
        d2 = d+'Calibrated/'+x+'/'
        if not os.path.exists(d2):
            os.makedirs(d2)
        shutil.move(filename,d2+os.path.basename(filename))
    delete(d,'Norm')
    return 'Files Calibrated'

# Runs the calibrate function on a list of targets rather than individually
# x = ['a','b','c',...]

def batchcal(d,z):
    for i in z:
        calibrate(d,i)

# Substitutes x for y in file names within location d.

def rename(x,y):
    d = os.getcwd()+'/'
    for i in glob.glob(d+'*.*'):
        new_filename = i.replace(x,y)
        os.rename(i,new_filename)

# Creates a median flat file from a list of flat files rather than generating
# its own.

def medflat2(d,x,bias):
    filelist = x
    n = len(filelist)
    dataz = np.zeros((1024,1024,n))
    for i in range(len(filelist)):
        data = fits.getdata(filelist[i])
        datadiv = (data-bias)/(np.median(data-bias))
        dataz[:,:,i] = datadiv
    nflats = np.median(dataz,axis=2)
    return nflats

# Deletes all files in location d with unique identifier x.

def deletefiles(x):
    d = os.getcwd()+'/'
    for file in glob.glob(d+'*.*'):
        if x in file:
            os.remove(file)

# Moves files with keywords x and y from location d1 to location d2

def move(d1,d2,x,y):
    filelist = files(d1,x,y)
    for i in filelist:
        shutil.move(i,d2+os.path.basename(i))
    return 'Files Moved'

def align2(cube,x_shifts,y_shifts, files):
    n = cube.shape[2]
    new_cube = np.zeros((1280,1280,n))
    for i in range(n):
        t_matrix = np.array([[1.,0.,(x_shifts[i])],
                             [0.,1.,(y_shifts[i])],
                             [0.,0.,1.]])
        new_cube[:,:,i] = warp(cube[:,:,i], ProjectiveTransform(matrix=t_matrix), mode='edge', preserve_range=True)
        data, hd = fits.getdata(files[i], header=True)
        filename = files[i].replace('.fits','-'+str(i)+'Aligned'+'.fits')
        #fits.writeto(filename, new_cube[:,:,i], header=hd)
    return new_cube

def align_suvi_images(exptime):
    filelist = suvitimes(exptime)
    deletefiles('temp')
    mcslice(filelist)
    sliced_files = natsorted(listfiles('temp'))
    cube_1 = sunpy.map.Map(sliced_files,cube=True)
    shifts = mc_shift(cube_1,layer_index=0)
    x_shifts = shifts['x']
    y_shifts = shifts['y']
    times = pulltimes(sliced_files)
    x_shifts = x_shifts/2.5
    y_shifts = y_shifts/2.5
    x_slope = np.polyfit(times,x_shifts,1)[0]
    y_slope = np.polyfit(times,y_shifts,1)[0]
    new_xvals = rotsub(x_shifts,times,x_slope)
    new_yvals = rotsub(y_shifts,times,y_slope)
    new_xvals = new_xvals * u.dimensionless_unscaled
    new_yvals = new_yvals * u.dimensionless_unscaled
    x_val_pix = new_xvals * u.pix
    y_val_pix = new_yvals * u.pix
    cube_2 = cube(filelist)
    cube_3 = align2(cube_2,new_xvals,new_yvals,filelist)
    deletefiles('temp')
    return cube_3
    
def cube_movie(cube):
    n = cube.shape[2]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    def animate():
        tstart = time.time()
        data = cube[:,:,0]
        im=plt.imshow(data)
        for i in range(n):
            data = cube[:,:,i]
            im.set_data(data)
            fig.canvas.draw()
    win = fig.canvas.manager.window
    fig.canvas.manager.window.after(100,animate)
    plt.show()

def movie2(cube):
    dpi = 100
    n = cube.shape[2]
    def ani_frame():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        im = ax.imshow(cube[:,:,i],cmap='gray')
        tight_layout()
        def update_img(i):
            tmp = cube[:,:,i]
            im.set_data(tmp)
            return im
        ani = animation.FuncAnimation(fig,update_img,300,interval=30)
        writer = animation.writers['ffmpeg'](fps=30)
        ani.save('demo.mp4',writer=writer)
        return ani

def movie3(cube):
    n = cube.shape[2]
    images = []
    for i in range(n):
        im = cube[:,:,i]
        filename = 'cube-'+str(i)+'.png'
        imageio.imwrite(filename,im)
        images.append(imageio.imread(filename))
    imageio.mimwrite('demo.mp4', images, fps = 30)
    
def cube_mean(cube, window):
    n = cube.shape[2]
    x = cube.shape[0]
    y = cube.shape[1]
    dataz = np.zeros((x,y,n))
    for i in range(n):
        if i < window:
            cube_1 = cube[:,:,0:(int(i+window))]
            window_means = np.mean(cube_1,axis=2)
            dataz[:,:,i] = window_means
        elif (n - i) < window:
            window_means = np.mean(cube[:,:,int(i-window):int(n)],axis=2)
            dataz[:,:,i] = window_means
        else:
            window_means = np.mean(cube[:,:,int(i-window):int(i+window)],axis=2)
            dataz[:,:,i] = window_means
    return dataz
            
def astro_conv(cube,arg):
    n = cube.shape[2]
    x = cube.shape[0]
    y = cube.shape[1]
    kernel = Box2DKernel(width=arg)
    dataz = np.zeros((x,y,n))
    for i in range(n):
        image = cube[:,:,i]
        astropy_conv = convolve(image, kernel)
        dataz[:,:,i] = astropy_conv
    return dataz
        
# For 06/25/18:
# Align images, run cube_mean + constant (start with 0.03)
# Divide images by result.
# Need to experiment with different kernels and normalization

def image_coords(cube,t,v_min,v_max):
    coords = []
    fig = plt.figure()
    def onclick(event):
        ix, iy = float(event.xdata), float(event.ydata)
        coord = ix, iy
        coords.append(coord)
    image = np.flip(cube[600:900,900:1280,t],axis=0)
    norm = ImageNormalize(image, interval=ManualInterval(vmin=v_min,vmax=v_max),stretch=ContrastBiasStretch(0.3,0.5))
    plt.imshow(image,norm=norm, cmap = 'gray')
    plt.show()
    fig.canvas.mpl_connect('button_press_event',onclick)
    return coords

def image_coords2(cube,t,v_min,v_max):
    coords = []
    def on_click(event):
        coord = event.xdata, event.ydata
        coords.append(coord)
    fig, ax = plt.subplots()
    image = np.flip(cube[600:900,900:1280,t],axis=0)
    norm = ImageNormalize(image, interval=ManualInterval(vmin=v_min,vmax=v_max),stretch=ContrastBiasStretch(0.3,0.5))
    plt.imshow(image,norm=norm, cmap = 'gray')
    plt.show()
    fig.canvas.mpl_connect('button_press_event', on_click)
    return coords

def onclick(event):
    xc, yc = event.xdata, event.ydata
    coords.append((xc,yc))

def find_coords(cube,i):
    global fig
    global coords
    coords = []
    fig = plt.figure()
    plt.gray()
    ax = fig.add_subplot(111)
    image = np.flip(cube[740:840,1000:1280,i],axis=0)
    norm = ImageNormalize(image, interval=ManualInterval(vmin=0.25,vmax=0.75),stretch=ContrastBiasStretch(0.3,0.5))
    ax.imshow(image, norm=norm)
    global cid
    cid = fig.canvas.mpl_connect('button_press_event',onclick)
    plt.show()
    return coords

def best_fit_sheet(cube,i,x,y,j):
    image = np.flip(cube[740:840,1000:1280,i],axis=0)
    norm = ImageNormalize(image, interval=ManualInterval(vmin=0.25,vmax=0.75),stretch=ContrastBiasStretch(0.3,0.5))
    plt.imshow(image, norm=norm,cmap='gray')
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x,y,j))(np.unique(x)))
    fit = np.poly1d(np.polyfit(x,y,j))
    plt.savefig("Sheet_With_Line.png",bbox_inches='tight')
    plt.show()
    return fit

def plot_sheet(cube,i,fit,x_vals,y_vals):
    x = np.arange(30,182,1)
    y = fit(x)
    image = np.flip(cube[740:840,1000:1280,i],axis=0)
    norm = ImageNormalize(image, interval=ManualInterval(vmin=0.25,vmax=0.75),stretch=ContrastBiasStretch(0.3,0.5))
    y = [int(np.round(yi)) for yi in y]
    #print(x)
    #print(y)
    #print(image.shape)
    #image[y,x] = 30
    plt.imshow(image, norm=norm,cmap='gray')
    plt.plot(x,y,'ro')
    plt.show()

def suvi_imshow(cube,i):
    image = np.flip(cube[740:840,1000:1280,i],axis=0)
    norm = ImageNormalize(image, interval=ManualInterval(vmin=0.25,vmax=0.75),stretch=ContrastBiasStretch(0.3,0.5))
    plt.imshow(image, norm=norm,cmap='gray')
    plt.show()

def suvi_imshow_file(file):
    data = fits.getdata(file)
    image = np.flip(data[740:840,1000:1280],axis=0)
    norm = ImageNormalize(image, interval=ManualInterval(vmin=0.25,vmax=0.75),stretch=ContrastBiasStretch(0.3,0.5))
    plt.imshow(image, norm=norm,cmap='gray')
    plt.show()

def sheet_array(cube,fit):
    n = cube.shape[2]
    x = np.arange(30,182,1)
    y = fit(x)
    n_x = len(x)
    n_y = len(y)
    cube_sliced = np.flip(cube[740:840,1000:1280,:],axis=0)
    new_cube = np.zeros((n_x,n))
    for i in range(n):
        for j in x:
            j_new = j-30
            fancy_index = int(np.round(fit(j)))
            #print(fancy_index)
            R = cube_sliced[fancy_index,j,i]
            new_cube[j_new,i] = R
    return np.flip(new_cube,axis=0)

def sheet_array_new(cube,fit):
    n = cube.shape[2]
    x = np.arange(30,182,1)
    x_2 = np.arange(70,182,1)
    y = fit(x)
    n_x = len(x)
    n_y = len(y)
    cube_sliced = np.flip(cube[740:840,1000:1280,:],axis=0)
    new_cube = np.zeros((n_x,n))
    for i in range(n):
        image = cube_sliced[:,:,i]
        #zi = scipy.ndimage.map_coordinates(image, np.vstack((x,y)),order=5,mode='nearest')
        for j in x:
            j_new = j - 30
            fancy_index = int(fit(j))
            y_array = cube_sliced[fancy_index-1:fancy_index+2,j,i]
            y_avg = np.median(y_array)
            #print(fancy_index)
            R = cube_sliced[fancy_index,j,i]
            new_cube[j_new,i] = y_avg
        #for k in x_2:
            #k_new = k-30
            #new_cube[k_new,i] = zi[k_new]
    plt.savefig("Sheet.png",bbox_inches='tight')
    return np.flip(new_cube,axis=0)

def rename_gated(files):
    n = len(files)
    d = os.getcwd()+'/'
    for i in range(n):
        file_name = files[i]
        new_filename = file_name + '.fits'
        os.rename(file_name,new_filename)
        
def get_coords_max(cube,index,fit,x_vals):
    image = np.flip(cube[740:840,1000:1280,index],axis=0)
    y_coords = []
    x_coords = np.arange(int(np.min(x_vals)),int(np.max(x_vals)),1)
    for i in x_coords:
        y = int(fit(i))
        y_min = y - 2
        y_max = y + 3
        array = image[y_min:y_max,(i-1):i]
        max_val = array.argmax()
        max_coords = np.unravel_index(array.argmax(), array.shape)
        max_y = (max_coords[0] - 2) + y
        y_coords.append(max_y)
    return x_coords, y_coords
        
def sheet_interp(cube,index,fit,x_vals):
    image = np.flip(cube[740:840,1000:1280,index],axis=0)
    x_coords = np.arange(np.min(x_vals),np.max(x_vals),1)
    y_vals = fit(x_coords)
    zi = scipy.ndimage.map_coordinates(image, np.vstack((x_vals,y_vals)))
    return zi

def line_anim(image):
    fig,ax = plt.subplots()
    im_x = image.shape[1]
    im_y = image.shape[0]
    images = []
    for i in range(im_x):
        im = plt.imshow(image,cmap='gray'),plt.axvline(x = i)
        images.append(im)
    ani = animation.ArtistAnimation(fig, images, interval = 100)
    plt.show()

def create_dual_plots(cube, sheet):
    n_cube = cube.shape[2]
    n_sheet = sheet.shape[1]
    for i in range(n_cube):
        fig, (ax1, ax2) = plt.subplots(1,2)
        title = plt.text(-70.5, 25.01, 'Image Index = '+str(i), horizontalalignment='center',verticalalignment='bottom')
        data = np.flip(cube[600:900,900:1280,i],axis=0)
        norm = ImageNormalize(data, interval=ManualInterval(vmin=0.3,vmax=1),stretch=SqrtStretch())
        image = ax1.imshow(data,norm=norm,cmap='gray')
        im = ax2.imshow(sheet, cmap='gray'), ax2.axvline(x = i)
        plt.savefig('ani_plot_'+str(i)+'.png')
    
class suvi_animation(object):
    def __init__(self, cube, sheet, v_min1, v_max1):
        self.cube = cube
        self.n = self.cube.shape[2]
        self.sheet = sheet
        self.v_min1 = v_min1
        self.v_max1 = v_max1
        self.pause = False
        self.SqrtStretch = SqrtStretch()
        self.ContrastBiasStretch = ContrastBiasStretch(0.3,0.5)
        self.LogStretch = LogStretch(3)
        #self.stretch = self.click_stretch()
        
    def make_movie(self):
        images = []
        n = self.cube.shape[2]
        for i in range(n):
            title = plt.text(-80.5, 20.01, 'Image Index = '+str(i), horizontalalignment='center',verticalalignment='bottom')
            data = np.flip(self.cube[600:900,900:1280,i],axis=0)
            norm = ImageNormalize(data, interval=ManualInterval(vmin=self.v_min1,vmax=self.v_max1),stretch=SqrtStretch())
            image = self.ax1.imshow(data,norm=norm,cmap='gray')
            images.append([image,title])
        return images
    
    def make_line_movie(self):
        im_x = self.sheet.shape[1]
        y_range = np.arange(0,153,5)
        y_labels = np.arange(130,283,5)
        images = []
        for i in range(im_x):
            im = self.ax2.imshow(self.sheet, cmap='gray'), self.ax2.axvline(x = i, linewidth=2)
            images.append(im)
        return images
    
    def animate(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1,2, figsize = (16,9))
        self.ax1.axis('off')
        self.images = self.make_movie()
        self.images2 = self.make_line_movie()
        #image = np.flip(self.cube[740:840,1000:1280,self.index],axis=0)
        #norm = ImageNormalize(image, interval=ManualInterval(vmin=0.25,vmax=0.75),stretch=ContrastBiasStretch(0.3,0.5))
        self.ani = animation.ArtistAnimation(self.fig, self.images, interval=200, blit=False, repeat = True)
        self.ani2 = animation.ArtistAnimation(self.fig,self.images2, interval=200,blit=False, repeat = True)
        #self.axpause = plt.axes([0.4, 0.05, 0.1, 0.075])
        #self.axscale = plt.axes([0.2, 1.05, 1.1, 1.075])
        #self.bpause = Button(self.axpause, "Pause")
        #self.axstart = plt.axes([0.55, 0.05, 0.1, 0.075])
        #self.bstart = Button(self.axstart, "Restart")
        #self.bstart.on_clicked(self.click_restart)
        #self.bpause.on_clicked(self.onClick)
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        #Writer = animation.writers['ffmpeg']
        #writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        #self.radioscale = RadioButtons(self.axscale, ('SqrtStretch','ContrastBiasStretch','LogStretch'))
        #self.radioscale.on_clicked(self.click_stretch())
        plt.show()

    #def click_stretch(self, label):
#        stretch_dict = {'SqrtStretch': self.SqrtStretch, 'ContrastBiasStretch': self.ContrastBiasStretch, 'LogStretch': self.LogStretch} 
#        stretch = stretch_dict[label]
#        return stretch
    
    def click_restart(self, event):
        self.animate()
    
    def onClick(self, event):
        if self.pause:
            self.ani.event_source.stop()
            self.ani2.event_source.stop()
        else:
            self.ani.event_source.start()
            self.ani2.event_source.start()
        self.pause ^= True       
        
class suvi_animation2(object):
    def __init__(self, cube, v_min,v_max):
        #self.files = natsorted(files)
        #self.n = len(files)
        self.pause = False
        self.cube = cube
        self.n = cube.shape[2]
        self.v_min = v_min
        self.v_max = v_max
        
    def make_movie(self):
        images = []
        n = self.cube.shape[2]
        for i in range(n):
            title = plt.text(1.5, -2.01, 'Image Index = '+str(i), horizontalalignment='center',verticalalignment='bottom')
            data = np.flip(self.cube[600:900,900:1280,i],axis=0)
            norm = ImageNormalize(data, interval=ManualInterval(vmin=self.v_min,vmax=self.v_max),stretch=SqrtStretch())
            image = plt.imshow(data,norm=norm,cmap='gray')
            images.append([image,title])
        return images
    
    def animate(self):
        self.fig = plt.figure()
        self.images = self.make_movie()
        #self.ax.set_aspect("equal")
        #self.ax2.set_aspect("equal")
        #Writer = animation.writers['ffmpeg']
        #writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate = 1800)
        #self.ani = animation.FuncAnimation(self.fig, func = self.make_movie2())
        #self.ani2 = animation.FuncAnimation(self.fig, func = self.make_line_movie2())
        self.ani = animation.ArtistAnimation(self.fig, self.images, interval=500)
        #self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        #self.ani.save('SUVI_131_Sheet_Animation_Gated.mp4', writer=writer)
        self.axpause = plt.axes([0.4, 0.02, 0.1, 0.05])
        self.bpause = Button(self.axpause, "Pause")
        self.axstart = plt.axes([0.55, 0.02, 0.1, 0.05])
        self.bstart = Button(self.axstart, "Restart")
        self.bstart.on_clicked(self.click_restart)
        self.bpause.on_clicked(self.onClick)

        plt.show()

    #def click_stretch(self, label):
#        stretch_dict = {'SqrtStretch': self.SqrtStretch, 'ContrastBiasStretch': self.ContrastBiasStretch, 'LogStretch': self.LogStretch} 
#        stretch = stretch_dict[label]
#        return stretch
    
    def click_restart(self, event):
        self.animate()
    
    def onClick(self, event):
        if self.pause:
            self.ani.event_source.stop()
            #self.ani2.event_source.stop()
        else:
            self.ani.event_source.start()
            #self.ani2.event_source.start()
        self.pause ^= True 

def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

def suvi_imshow(cube,v_min,v_max,i):
    data = np.flip(cube[600:900,900:1280,i],axis=0)
    norm = ImageNormalize(data, interval=ManualInterval(vmin=v_min,vmax=v_max),stretch=SqrtStretch())
    image = plt.imshow(data,norm=norm,cmap='gray')
    plt.axis('off')
    plt.savefig("SUVI_Example_Gated.png",bbox_inches='tight')
    plt.show()


def test_animation(cube):
    n = cube.shape[2]
    fig, ax = plt.subplots()
    def update(i):
        data = np.flip(cube[:,:,i],axis=0)
        image = plt.imshow(data,cmap='gray')
        ax.set_title('Frame '+str(i))
        return image
    ani = animation.FuncAnimation(fig, update, frames=np.arange(n), interval=10)
    plt.show()

def test_animation2(cube):
    n = cube.shape[2]
    i = np.arange(n)
    fig, ax = plt.subplots()
    im = np.flip(cube[:,:,i])
    ax_slider = plt.axes([0.25,0.03,0.50,0.02])
    slider = Slider(ax_slider, 'Images',0, n, valinit=0,valstep=1)
    is_manual = False
    def update_slider(val):
        global is_manual
        is_manual = True
        update(val)

    def update(val):
        im = np.flip(cube[:,:,val])
        ax.set_title('Image '+str(val))
        plt.imshow(im)

    def update_plot(num):
        global is_manual
        if is_manual:
            return im,
        val = slider.val
        slider.set_val(val)
        is_manual = False
        return im,

    def on_click(event):
        (xm,ym),(xM,yM) = slider.label.clipbox.get_points()
        if xm < event.x < xM and ym < event.y < yM:
            return
        else:
            global is_manual
            is_manual = False

    slider.on_changed(update_slider)
    fig.canvas.mpl_connect('button_press_event', on_click)
    ani = animation.FuncAnimation(fig, update_plot, interval = 10)
    plt.show()

    

