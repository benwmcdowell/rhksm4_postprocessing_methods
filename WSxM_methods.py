import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class profile():
    def __init__(self,ifile):
        self.x=[]
        self.y=[]
        with open(ifile) as f:
            read_data=False
            while True:
                line=f.readline()
                if not line:
                    break
                if read_data:
                    line=line.split()
                    self.x.append(float(line[0]))
                    self.y.append(float(line[1]))
                if '[Header end]' in line:
                    read_data=True
        self.x=np.array(self.x)
        self.y=np.array(self.y)
        
    def plot_profile(self):
        self.fig,self.ax=plt.subplots(1,1)
        
    def add_savgol_filter(self,w,o):
        self.y=savgol_filter(self.y,w,o)
        
    def add_gaussian_filter(self,s):
        self.y=gaussian_filter(self.y,s)
    
    def fit_peak(self,**args):
        def gauss(x,a,x0,s,y0):
            y=y0+a*np.exp(-(x-x0)**2/s/2)
            return y
        
        if 'xrange' in args:
            xrange=args['xrange']
            for i in range(2):
                xrange[i]=np.argmin(abs(self.x-xrange[i]))
        else:
            xrange=[0,len(self.x)]
        
        popt,pcov=curve_fit(gauss,self.x[xrange[0]:xrange[1]],self.y[xrange[0]:xrange[1]],p0=[max(self.y[xrange[0]:xrange[1]]),np.average(self.x[xrange[0]:xrange[1]]),0.05,0],bounds=([0,np.min(self.x[xrange[0]:xrange[1]]),0,0],[np.max(self.y[xrange[0]:xrange[1]])*1.5,np.max(self.x[xrange[0]:xrange[1]]),self.x[-1]-self.x[0],np.max(self.y[xrange[0]:xrange[1]])]))
        pcov=np.sqrt(np.diag(pcov))
        print('peak is centered at {} +/- {}'.format(popt[1],pcov[1]))
        
    def find_peak(self):
        peak_center=self.x[np.argmax(self.y)]
        print('peak is centered at {}'.format(peak_center))