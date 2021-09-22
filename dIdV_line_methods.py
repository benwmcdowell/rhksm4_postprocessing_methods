import matplotlib.pyplot as plt
import rhksm4
from scipy.optimize import curve_fit
from scipy.fft import fft,fftfreq

import numpy as np

class dIdV_line:
    def __init__(self,ifile,**args):
        if 'line_num' not in args:
            self.line_num=0
        else:
            self.line_num=args['line_num']
            
        if 'scalefactor' in args:
            self.sf=args['scalefactor']
        else:
            self.sf=1.0
            
        self.f=rhksm4.load(ifile)
        self.size=int(np.shape(self.f[0].data)[0])
        self.npts=int(np.shape(self.f[4].data)[1])
        self.energy=np.array([self.f[4].attrs['RHK_Xoffset']+i*self.f[4].attrs['RHK_Xscale'] for i in range(self.npts)])
        self.pos=np.array([self.f[0].attrs['RHK_Xoffset']+i*self.f[0].attrs['RHK_Xscale'] for i in range(self.size)])*1.0e10/self.sf 
        self.LIAcurrent=np.array([self.f[4].data[self.size*self.line_num:self.size*(self.line_num+1)]])
        self.current=np.array([self.f[5].data[self.size*self.line_num:self.size*(self.line_num+1)]])
        
        self.energy=self.energy[::-1] #bias in V
        self.pos=abs(self.pos-max(self.pos)) #postion along line in $\AA$
        self.LIAcurrent=self.LIAcurrent[0,:,:].T[::-1]*1.0e12 #LIAcurrent in pA
        self.current=self.current[0,:,:].T[::-1]*1.0e12 #current in pA
    def normalize(self,**args):
        if 'range' in args:
            norm_range=[]
            for i in range(self.npts):
                if self.energies[i]>args['range'][0] and len(norm_range):
                    norm_range.append(i)
                if self.energies[i]>args['range'][1]:
                    norm_range.append(i)
                    break
        else:
            norm_range=[0,self.npts]
        
        for i in range(self.size):
            self.LIAcurrent[:,i]/=sum(self.LIAcurrent[norm_range[0]:norm_range[1],i])
            self.current[:,i]/=sum(self.current[norm_range[0]:norm_range[1],i])
            
    def normalize_periodic(self,k):
        # k must be in $\AA$
        def model_cosine(x,A,phi,y0):
            y=A*np.cos(phi+x*2.0*np.pi/k)+y0
            return y
        
        for i in range(self.npts):
            params=curve_fit(model_cosine,self.pos,self.LIAcurrent[i])
            fit=model_cosine(self.pos,params[0][0],params[0][1],params[0][2])
            self.LIAcurrent[i]-=fit
        
    def plot_dIdV_line(self):
        fig,ax=plt.subplots(1,1)
        x=np.array([[self.pos[i] for i in range(self.size)] for j in range(self.npts)])
        y=np.array([[self.energy[j] for i in range(self.size)] for j in range(self.npts)])
        dIdVmap=ax.pcolormesh(x,y,self.LIAcurrent,cmap='jet',shading='nearest')
        ax.set(xlabel='position / $\AA$')
        ax.set(ylabel='bias / V')
        plt.tight_layout()
        fig.show()
        
    def plot_fft(self):
        fig,ax=plt.subplots(1,1)
        zf=np.zeros((self.npts,self.size//2))
        xf=np.zeros((self.npts,self.size//2))
        for i in range(self.npts):
            zf[i]+=np.abs(fft(self.LIAcurrent[i])[0:self.size//2])*2.0/self.size
            xf[i]+=fftfreq(self.size,(self.pos[-1]-self.pos[0])/(self.size-1))[:self.size//2]
        y=np.array([[self.energy[j] for i in range(self.size//2)] for j in range(self.npts)])
        plt.pcolormesh(xf,y,zf,cmap='jet',shading='nearest')
        plt.show()