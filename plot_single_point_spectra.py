import matplotlib.pyplot as plt
import rhksm4
from scipy.optimize import curve_fit
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import savgol_filter

def get_single_point(ifile,filter_params=(0,0),**args):
    f=rhksm4.load(ifile)
    scan_num=np.shape(f[0].data)[0]
    if 'average_scans' in args:
        average=args['average_scans']
    else:
        average=[i for i in range(scan_num)]
        
    if 'total_current' in args:
        if args['total_current']!=True:
            page_num=0
        else:
            page_num=1
    else:
        page_num=0
    
    ydata=np.zeros(np.shape(f[0].data)[1])
    xdata=np.zeros(np.shape(f[0].data)[1])
    zdata=np.zeros(np.shape(f[0].data)[1])
    for i in average:
        tempvar=f[page_num].data[i]*f[page_num].attrs['RHK_Zscale']+f[page_num].attrs['RHK_Zoffset'] #LIA current in pA
        if filter_params!=(0,0):
            tempvar=savgol_filter(tempvar,filter_params[0],filter_params[1])
        ydata+=tempvar
    ydata/=len(average)
    try:
        for i in average:
            tempvar=f[2].data[i]*f[2].attrs['RHK_Zscale']+f[2].attrs['RHK_Zoffset'] #LIA current in pA
            if filter_params!=(0,0):
                tempvar=savgol_filter(tempvar,filter_params[0],filter_params[1])
            zdata+=tempvar
        zdata/=len(average)
    except IndexError:
        pass
    xdata+=[f[0].attrs['RHK_Xoffset']+i*f[0].attrs['RHK_Xscale'] for i in range(len(ydata))] #bias in V
    setpoint=f[0].attrs['RHK_Current']/1e-12 #current setpoint in pA
    
    #converts current data to pA
    ydata/=1e-12
    
    #reverses data to be in logical bias progression
    xdata=xdata[::-1]
    ydata=ydata[::-1]
    
    return xdata,ydata,setpoint,scan_num,zdata

def copy_peak_data(ifile):
    xdata,ydata,setpoint,scan_num,zdata=get_single_point(ifile)
    peak_width=5
    peak_height=0.25*np.max(ydata)-np.min(ydata)
    peak_indices=find_peaks(ydata,width=peak_width,height=peak_height)[0]
    peak_energies=[xdata[i] for i in peak_indices]
    z_vals=[zdata[i]*1e9 for i in peak_indices]
    
    return peak_energies,z_vals

def plot_single_point(ifiles,**args):
    if 'normalize' in args:
        normalize=True
        #normalizes relative to voltage range specified in normalize argument
        norm_range=args['normalize']
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
        
    if 'total_current' in args:
        total_current=True
    else:
        total_current=False
        
    if 'savgol_filter' in args:
        filter_params=args['savgol_filter']
    else:
        filter_params=(0,0)
        
    ydata=[]
    xdata=[]
    setpoints=[]
    num=len(ifiles)
    for i in ifiles:
        tempvar=get_single_point(i,total_current=total_current,filter_params=filter_params)[:4]
        xdata.append(tempvar[0])
        ydata.append(tempvar[1])
        setpoints.append(tempvar[2])
        if normalize:
            temp_range=[]
            for i in [min(norm_range),max(norm_range)]:
                temp_range.append(np.argmin(abs(i-xdata[-1])))
            ydata[-1]/=sum(ydata[-1][min(temp_range):max(temp_range)])
        if 'fit_range' in args:
            fit_range=args['fit_range']
        else:
            fit_range=[np.min(xdata[-1]),np.max(xdata[-1])]
        for j in range(2):
            fit_range[j]=np.argmin(abs(fit_range[j]-xdata[-1]))
            
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
        plt.plot(xdata[i],ydata[i],label=labels[i])
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
    
def plot_fr_avg(ifile):
    scan_num=get_single_point(ifile)[-1]
    fnum=[]
    rnum=[]
    for i in range(scan_num):
        if (i+1)%2==0:
            rnum.append(i)
        else:
            fnum.append(i)
    xf,yf=get_single_point(ifile,average_scans=fnum)[:2]
    xr,yr=get_single_point(ifile,average_scans=rnum)[:2]
    xf_tot,yf_tot=get_single_point(ifile,average_scans=fnum,total_current=True)[:2]
    xr_tot,yr_tot=get_single_point(ifile,average_scans=rnum,total_current=True)[:2]
    
    fig,ax=plt.subplots(2,1,sharex=True)
    ax[1].plot(xf,yf,label='forward',color='red')
    ax[1].plot(xr,yr,label='reverse',color='blue')
    ax[0].plot(xf_tot,yf_tot,color='red')
    ax[0].plot(xr_tot,yr_tot,color='blue')
    ax[0].set(xlabel='bias / V')
    ax[1].set(ylabel='LIA current / pA')
    ax[0].set(ylabel='current / pA')
    fig.legend()
    fig.show()
    
def fano_fit(x,q,E,G):
    ered=2*(x-E)/G
    f=(q+ered)**2/(1+ered**2)-1
    return f

def gaussian_fit(x,A,s,x0,y0):
    y=A*np.exp(-(x-x0)**2/s/2)+y0
    return y