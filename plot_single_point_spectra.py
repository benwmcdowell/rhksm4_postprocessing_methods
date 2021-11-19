import matplotlib.pyplot as plt
import rhksm4
from scipy.optimize import curve_fit
import numpy as np
from scipy.signal import find_peaks

def get_single_point(ifile,**args):
    f=rhksm4.load(ifile)
    scan_num=np.shape(f[0].data)[0]
    if 'average_scans' in args:
        average=args['average_scans']
    else:
        average=[i for i in range(scan_num)]
    
    ydata=np.zeros(np.shape(f[0].data)[1])
    xdata=np.zeros(np.shape(f[0].data)[1])
    for i in average:
        ydata+=f[0].data[i]*f[0].attrs['RHK_Zscale']+f[0].attrs['RHK_Zoffset'] #LIA current in pA
    ydata/=scan_num
    xdata+=[f[0].attrs['RHK_Xoffset']+i*f[0].attrs['RHK_Xscale'] for i in range(len(ydata))] #bias in V
    setpoint=f[0].attrs['RHK_Current']/1e-12 #current setpoint in pA
    
    #converts current data to pA
    ydata/=1e-12
    
    #reverses data to be in logical bias progression
    xdata=xdata[::-1]
    ydata=ydata[::-1]
    
    return xdata,ydata,setpoint

def plot_single_point(ifiles,**args):
    if 'normalize' in args:
        normalize=True
        #normalizes relative to max dI/dV point
    else:
        normalize=False
        
    if 'fit_lineshape' in args:
        fit_lineshape=args['fit_lineshape']
    else:
        fit_lineshape=None
        
    if 'find_peak' in args:
        find_peak=True
    else:
        find_peak=False
        
    if 'labels' in args:
        labels=args['labels']
    else:
        labels=[None for i in range(len(ifiles))]
        
    ydata=[]
    xdata=[]
    setpoints=[]
    num=len(ifiles)
    for i in ifiles:
        tempvar=get_single_point(i)
        xdata.append(tempvar[0])
        ydata.append(tempvar[1])
        setpoints.append(tempvar[2])
        if normalize:
            ydata[-1]/=max(ydata[-1])
        if 'fit_range' in args:
            fit_range=args['fit_range']
        else:
            fit_range=[np.min(xdata),np.max(xdata)]
        for j in range(2):
            fit_range[j]=np.argmin(abs(fit_range[j]-xdata[-1]))
        print(fit_range)
            
    fit_params=[]
    fit_error=[]
    if fit_lineshape=='fano':
        for i in range(num):
            fit_params.append(curve_fit(fano_fit,xdata[i],ydata[i],p0=[-5,2.6,0.1],bounds=([-20,2.2,0.05],[0,2.9,1.0]))[0])
    if fit_lineshape=='gaussian':
        for i in range(num):
            popt,pcov=curve_fit(gaussian_fit,xdata[i][min(fit_range):max(fit_range)],ydata[i][min(fit_range):max(fit_range)],bounds=([0,0,np.min(xdata[i][min(fit_range):max(fit_range)]),-np.inf],[np.inf,np.inf,np.max(xdata[i][min(fit_range):max(fit_range)]),np.inf]))
            fit_params.append(popt)
            pcov=np.sqrt(np.diag(pcov))
            fit_error.append(pcov)
            print('peak at: {} +/- {} eV'.format(popt[2],pcov[2]))
            
    plt.figure()
    for i in range(num):
        plt.scatter(xdata[i],ydata[i],label=labels[i])
        if fit_lineshape=='fano':
            plt.plot(xdata[i],fano_fit(xdata[i],fit_params[i][0],fit_params[i][1],fit_params[i][2]))
        if find_peak:
            print('single point #{}:'.format(i+1))
            peak_width=5
            peak_height=0.25*np.max(ydata[i])-np.min(ydata[i])
            peak_indices=find_peaks(ydata[i],width=peak_width,height=peak_height)[0]
            for j in peak_indices:
                plt.plot([xdata[i][j],xdata[i][j]],[np.min(ydata[i]),np.max(ydata[i])])
                print('peak at: {} V'.format(xdata[i][j]))
    plt.ylabel('LIA current / pA')
    plt.xlabel('bias / V')
    if 'labels' in args:
        plt.legend()
    plt.show()
    
def fano_fit(x,q,E,G):
    ered=2*(x-E)/G
    f=(q+ered)**2/(1+ered**2)-1
    return f

def gaussian_fit(x,A,s,x0,y0):
    y=A*np.exp(-(x-x0)**2/s/2)+y0
    return y