import matplotlib.pyplot as plt
import rhksm4
from scipy.optimize import curve_fit
from scipy.fft import fft,fftfreq
from scipy.signal.windows import hann
from scipy.signal import savgol_filter
from copy import deepcopy
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
        
    def plot_dIdV_line(self,**args):
        if 'cmap' in args:
            cmap=args['cmap']
        else:
            cmap=plt.rcParams['image.cmap']
        
        self.fig_main,self.ax_main=plt.subplots(1,1,tight_layout=True)
        x=np.array([[self.pos[i] for i in range(self.size)] for j in range(self.npts)])
        y=np.array([[self.energy[j] for i in range(self.size)] for j in range(self.npts)])
        self.dIdVmap=self.ax_main.pcolormesh(x,y,self.LIAcurrent,cmap=cmap,shading='nearest')
        self.ax_main.set(xlabel='position / $\AA$')
        self.ax_main.set(ylabel='bias / V')
        self.fig_main.show()
        
    def plot_energy_slice(self,energy):
        self.fig_eslice,self.ax_eslice=plt.subplots(1,1,tight_layout=True)
        if type(energy)==list:
            for e in energy:
                i=np.argmin(abs(self.energy-e))
                self.ax_eslice.plot(self.pos,self.LIAcurrent[i],label=e)
                self.ax_main.plot([self.pos[0],self.pos[-1]],[e,e])
        else:
            i=np.argmin(abs(self.energy-energy))
            self.ax_eslice.plot(self.pos,self.LIAcurrent[i])
            self.ax_main.plot([self.pos[0],self.pos[-1]],[e,e])
        self.ax_eslice.set(xlabel='position / $\AA$')
        self.ax_eslice.set(ylabel='dI/dV / pA')
        self.ax_eslice.legend()
        self.fig_eslice.show()
        
    def overlay_bounds(self,pos):
        for i in pos:
            self.boundline=self.ax_main.plot([self.pos[0],self.pos[-1]],[i,i],color='white',linestyle='dashed')
        self.fig_main.canvas.draw()
        
    def overlay_center(self,pos):
        self.centerline=self.ax_main.plot([pos,pos],[self.energy[0],self.energy[-1]],color='white',linestyle='dashed')
        
    def find_scattering_length(self,emin,emax,center,**args):
        def gauss_fit(x,A,s,x1,x2,y0):
            y=A*(np.exp(-(x-x1)**2/s)+np.exp(-(x-x2)**2/s))+y0
            return y
        
        center=np.argmin(abs(self.pos-center))
        emin=np.argmin(abs(self.energy-emin))
        emax=np.argmin(abs(self.energy-emax))
        
        if 'xrange' in args:
            xmin=np.argmin(abs(self.pos-args['xrange'][0]))
            xmax=np.argmin(abs(self.pos-args['xrange'][1]))
        else:
            xmin=0
            xmax=len(self.pos)-1
        
        if 'overlay_peaks' in args:
            overlay_peaks=args['overlay_peaks']
        else:
            overlay_peaks=True
            
        energies=[]
        lengths=[]
        peak_pos=[]
        peak_energies=[]
        errors=[]
        
        for i in range(emin,emax):
            p0=[max(self.LIAcurrent[i,xmin:xmax])-min(self.LIAcurrent[i,xmin:xmax]),0.5,(self.pos[center]-self.pos[xmin])/2,(self.pos[xmax]-self.pos[center])/2,min(self.LIAcurrent[i,xmin:xmax])]
            popt,pcov=curve_fit(gauss_fit,self.pos[xmin:xmax],self.LIAcurrent[i,xmin:xmax],p0=p0)
                           
            lengths.append(abs(popt[2]-popt[3]))
            for j in range(2,4):
                peak_pos.append(popt[j])
                peak_energies.append(self.energy[i])
                pcov=np.sqrt(np.diag(pcov))
                errors.append(np.sqrt(pcov[2]**2+pcov[3]**2))
            
        if overlay_peaks:
            self.ax_main.scatter(peak_pos,peak_energies)
            
        return energies,lengths,errors
        
    def find_scattering_length_old(self,emin,emax,center,**args):
        center=np.argmin(abs(self.pos-center))
        emin=np.argmin(abs(self.energy-emin))
        emax=np.argmin(abs(self.energy-emax))
        if 'xrange' in args:
            xmin=np.argmin(abs(self.pos-args['xrange'][0]))
            xmax=np.argmin(abs(self.pos-args['xrange'][1]))
        
        if 'peak_width' in args:
            peak_width=int(args['peak_width'])
        else:
            peak_width=5
            
        if 'overlay_peaks' in args:
            overlay_peaks=args['overlay_peaks']
        else:
            overlay_peaks=True
            
        energies=[]
        lengths=[]
        peak_pos=[]
        peak_energies=[]
        
        for i in range(emin,emax):
            energies.append(self.energy[i])
            tempmax=0.0
            for j in range(center-xmin-peak_width):
                tempvar=np.average(self.LIAcurrent[i,(center-j)-peak_width:(center-j)+peak_width+1])
                if tempvar>tempmax:
                    tempmax=tempvar
                    max_index=center-j
                    
            left_peak=max_index
                 
            tempmax=0.0
            for j in range(len(self.pos)-center-peak_width-1-(len(self.pos)-xmax)):
                tempvar=np.average(self.LIAcurrent[i,(center+j)-peak_width:(center+j)+peak_width+1])
                if tempvar>tempmax:
                    tempmax=tempvar
                    max_index=center+j
                        
            right_peak=max_index
                           
            lengths.append(self.pos[right_peak]-self.pos[left_peak])
            for j in [right_peak,left_peak]:
                peak_pos.append(self.pos[j])
                peak_energies.append(self.energy[i])
            
        if overlay_peaks:
            self.ax_main.scatter(peak_pos,peak_energies)
            
        return energies,lengths
        
    def plot_fft(self,**args):
        fig,ax=plt.subplots(1,1)
        if 'window' in args:
            w=hann(self.size,sym=True)
        else:
            w=np.array([1.0 for i in range(self.size)])
        zf=np.zeros((self.npts,self.size//2))
        xf=np.zeros((self.npts,self.size//2))
        for i in range(self.npts):
            zf[i]+=np.abs(fft(self.LIAcurrent[i]*w)[0:self.size//2])*2.0/self.size
            xf[i]+=fftfreq(self.size,(self.pos[-1]-self.pos[0])/(self.size-1))[:self.size//2]
        y=np.array([[self.energy[j] for i in range(self.size//2)] for j in range(self.npts)])
        plt.pcolormesh(xf,y,zf,cmap='jet',shading='nearest')
        ax.set(ylabel='bias / eV')
        ax.set(xlabel='momentum / $\AA^{-1}$')
        fig.show()
        
    def add_savgol_filter(self,w,o):
        for i in range(len(self.energy)):
            self.LIAcurrent[i]=savgol_filter(self.LIAcurrent[i],w,o)
        for i in range(len(self.pos)):
            self.LIAcurrent[:,i]=savgol_filter(self.LIAcurrent[:,i],w,o)