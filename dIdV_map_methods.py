import rhksm4
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

class dIdV_map():
    def __init__(self,ifile,**args):
        self.f=rhksm4.load(ifile)
        if 'scalefactor' in args:
            self.sf=args['scalefactor']
        else:
            self.sf=(1.0,1.0,1.0)
            
        if 'scan_direction' in args:
            self.scan_direction=args['scan_direction']
        else:
            self.scan_direction='forward'
            
        if 'normalize' in args:
            if args['normalize']==False:
                normalize=True
            else:
                normalize=True
        else:
            normalize=True
            
        self.f=rhksm4.load(ifile)
        self.npts=np.shape(self.f[1].data)
        if self.scan_direction=='forward':
            self.topo=(self.f[2].data*self.f[2].attrs['RHK_Zscale']+self.f[2].attrs['RHK_Zoffset'])*1.0e9/self.sf[2]
        else:
            self.topo=(self.f[3].data*self.f[3].attrs['RHK_Zscale']+self.f[3].attrs['RHK_Zoffset'])*1.0e9/self.sf[2]
            
        self.x=np.array([self.f[3].attrs['RHK_Xoffset']+i*self.f[3].attrs['RHK_Xscale'] for i in range(self.npts[0])])*1.0e9/self.sf[0]
        self.y=np.array([self.f[3].attrs['RHK_Yoffset']+i*self.f[3].attrs['RHK_Yscale'] for i in range(self.npts[1])])*1.0e9/self.sf[1]
        
        self.x-=np.min(self.x)
        self.y-=np.min(self.y)
        
        self.epoints=np.array([self.f[4].attrs['RHK_Xoffset']+self.f[4].attrs['RHK_Xscale']*i for i in range(np.shape(self.f[4].data)[1])])
        self.z=np.zeros((len(self.epoints),self.npts[0],self.npts[1]))
        for i in range(np.shape(self.f[4].data)[0]):
            for j in range(np.shape(self.f[4].data)[1]):
                self.z[j,i//self.npts[0],i%self.npts[0]]=self.f[4].data[i,j]*self.f[4].attrs['RHK_Zscale']+self.f[4].attrs['RHK_Zoffset']

        if normalize:
            for i in range(len(self.epoints)):
                self.z[i,:,:]-=np.min(self.z[i,:,:])
                self.z[i,:,:]/=np.max(self.z[i,:,:])
                
    def add_savgol_filter(self,w,o,**args):
        for i in range(len(self.epoints)):
            for j in range(self.npts[0]):
                self.z[i,j,:]=savgol_filter(self.z[i,j,:],w,o)
            for j in range(self.npts[1]):
                self.z[i,:,j]=savgol_filter(self.z[i,:,j],w,o)
                
    def flatten_map(self):
        def linear_fit(x,a,b):
            y=a*x+b
            return y
        for i in range((len(self.epoints))):
            for j in range(self.npts[0]):
                popt,pcov=curve_fit(linear_fit,self.x,self.z[i,j,:])
                yfit=linear_fit(self.x,popt[0],popt[1])
                self.z[i,j,:]-=yfit
                
    def plot_slice(self,pos,**args):
        if 'orientation' in args:
            if args['orientation']=='vertical':
                horizontal=False
            else:
                horizontal=True
        else:
            horizontal=True
            
        if horizontal:
            pos=np.argmin(abs(self.y-pos))
        else:
            pos=np.argmin(abs(self.x-pos))
            
        self.slice_fig,self.slice_ax=plt.subplots(1,1)
        for i in range(len(self.epoints)):
            if horizontal:
                plt.plot(self.x,self.z[i,pos,:],label='{} V'.format(self.epoints[i]))
            else:
                plt.plot(self.y,self.z[i,:,pos],label='{} V'.format(self.epoints[i]))
        self.slice_ax.set(xlabel='position / nm',ylabel='normalized LIA current')
        self.slice_ax.legend()
        self.slice_fig.show()
        
    def plot_fft(self,**args):
        if 'orientation' in args:
            if args['orientation']=='vertical':
                horizontal=False
            else:
                horizontal=True
        else:
            horizontal=True
                
    def plot_maps(self,**args):
        if 'cmap' in args:
            cmap=args['cmap']
        else:
            cmap=plt.rcParams['image.cmap']
            
        for i in range(len(self.epoints)):
            plt.figure()
            plt.title('{} V'.format(self.epoints[i]))
            plt.pcolormesh([self.x for i in range(len(self.y))],[[self.y[j] for i in range(len(self.x))] for j in range(len(self.y))],self.z[i],shading='nearest',cmap=cmap,vmin=np.min(self.z[np.nonzero(self.z)]))
            plt.xlabel('position / nm')
            plt.ylabel('position / nm')
            plt.colorbar()
            plt.show()
            
    def export_to_ASCII(self,filepath):
        os.chdir(filepath)
        for i in range(len(self.epoints)):
            with open(Path(str(self.epoints[i])),'w+') as output:
                for j in range(self.npts[0]):
                    for k in range(self.npts[1]):
                        output.write(np.format_float_scientific(self.z[i,j,k],exp_digits=3,precision=5))
                        if k<self.npts[1]-1:
                            output.write('\t')
                    output.write('\n')