import matplotlib.pyplot as plt
import rhksm4
from scipy.optimize import curve_fit
import numpy as np

def get_single_point(ifile):
        
    f=rhksm4.load(ifile)
    ydata=f[0].data[0]*f[0].attrs['RHK_Zscale']+f[0].attrs['RHK_Zoffset']
    xdata=[f[0].attrs['RHK_Xoffset']+i*f[0].attrs['RHK_Xscale'] for i in range(len(ydata))]
    
    #converts current data to pA
    ydata/=1e-12
    
    return xdata,ydata

def plot_single_point(ifiles,labels,**args):
    if 'normalize' in args:
        normalize=True
        #normalizes relative to max dI/dV point
    else:
        normalize=False
        
    if 'fit' in args:
        fit=True
    else:
        fit=False
        
    ydata=[]
    xdata=[]
    num=len(ifiles)
    for i in ifiles:
        tempvar=get_single_point(i)
        xdata.append(tempvar[0])
        ydata.append(tempvar[1])
        if normalize:
            ydata[-1]/=max(ydata[-1])
    
    if fit:
        fit_params=[]
        for i in range(num):
            fit_params.append(curve_fit(fano_fit,xdata[i][40:],ydata[i][40:],p0=[-5,2.4,0.5],bounds=([-200,2.2,0.05],[-.5,2.6,0.6]))[0])
        print(fit_params)
            
    plt.figure()
    for i in range(num):
        plt.scatter(xdata[i],ydata[i],label=labels[i])
        if fit:
            plt.plot(xdata[i],fano_fit(xdata[i],fit_params[i][0],fit_params[i][1],fit_params[i][2]))
    plt.ylabel('LIA current / pA')
    plt.xlabel('bias / V')
    plt.legend()
    plt.show()
    
def fano_fit(x,q,E,G):
    ered=2*(x-E)/G
    f=(q+ered)**2/(1+ered**2)-1
    return f