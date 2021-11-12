import matplotlib.pyplot as plt
import rhksm4
from scipy.optimize import curve_fit
from scipy.fft import fft,fftfreq
from scipy.signal.windows import hann
from scipy.signal import savgol_filter
import numpy as np
import sys

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
        self.LIAfit=np.zeros((self.npts,self.size))
        
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
            
    def plot_fit_residuals(self,**args):
        if 'cmap' in args:
            cmap=args['cmap']
        else:
            cmap='seismic'
            
        self.fig_res,self.ax_res=plt.subplots(1,1,tight_layout=True)
        x=np.array([[self.pos[i] for i in range(self.size)] for j in range(self.npts)])
        y=np.array([[self.energy[j] for i in range(self.size)] for j in range(self.npts)])
        tempvar=np.zeros((self.npts,self.size))
        for i in range(self.npts):
            if max(self.LIAfit[i])>0.0:
                tempvar[i]+=self.LIAfit[i]-self.LIAcurrent[i]
        self.residualmap=self.ax_res.pcolormesh(x,y,tempvar,cmap=cmap,shading='nearest')
        self.ax_res.set(xlabel='position / $\AA$')
        self.ax_res.set(ylabel='bias / V')
        self.fig_res.show()
        
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
        
    def clear_energy_axes(self):
        self.ax_eslice.clear()
        
    def plot_energy_slice(self,energy):
        if not hasattr(self,'fig_slice'):
            self.fig_eslice,self.ax_eslice=plt.subplots(1,1,tight_layout=True)
        if type(energy)==list:
            for e in energy:
                i=np.argmin(abs(self.energy-e))
                self.ax_eslice.plot(self.pos,self.LIAcurrent[i],label='{} eV'.format(e))
                self.ax_main.plot([self.pos[0],self.pos[-1]],[e,e])
        else:
            i=np.argmin(abs(self.energy-energy))
            self.ax_eslice.plot(self.pos,self.LIAcurrent[i])
            self.ax_main.plot([self.pos[0],self.pos[-1]],[energy,energy])
        self.ax_eslice.set(xlabel='position / $\AA$')
        self.ax_eslice.set(ylabel='dI/dV / pA')
        self.ax_eslice.legend()
        self.fig_eslice.show()
        
    def plot_position_slice(self,pos,**args):
        if 'find_onset' in args:
            find_onset=True
            onsets=[]
            onset_range=args['find_onset']
            for i in range(2):
                onset_range[i]=np.argmin(abs(self.energy-onset_range[i]))
        else:
            find_onset=False
        
        self.fig_pslice,self.ax_pslice=plt.subplots(1,1,tight_layout=True)
        if type(pos) != list:
            pos=[pos]
        for p in pos:
            i=np.argmin(abs(self.pos-p))
            self.ax_pslice.plot(self.energy,self.LIAcurrent[:,i],label='{} $\AA$'.format(p))
            self.ax_main.plot([p,p],[self.energy[0],self.energy[-1]])
            if find_onset:
                onset_height=(np.max(self.LIAcurrent[onset_range[0]:onset_range[1],i])+np.min(self.LIAcurrent[onset_range[0]:onset_range[1],i]))/2
                onsets.append(self.energy[np.argmin(abs(self.LIAcurrent[:,i]-onset_height))])
                self.ax_pslice.plot([onsets[-1] for j in range(2)],[min(self.LIAcurrent[[onset_range[0],onset_range[1]],i]),max(self.LIAcurrent[[onset_range[0],onset_range[1]],i])],label='onset')
                if p==pos[-1]:
                    print('average 2d band onset: {} +/- {} eV'.format(np.mean(onsets),np.std(onsets)))
        self.ax_pslice.set(xlabel='bias / eV')
        self.ax_pslice.set(ylabel='dI/dV / pA')
        self.ax_pslice.legend()
        self.fig_pslice.show()
        
    def overlay_bounds(self,pos):
        for i in pos:
            self.boundline=self.ax_main.plot([self.pos[0],self.pos[-1]],[i,i],color='white',linestyle='dashed')
        self.fig_main.canvas.draw()
        
    def overlay_center(self,pos):
        self.centerline=self.ax_main.plot([pos,pos],[self.energy[0],self.energy[-1]],color='white',linestyle='dashed')
        
    def find_scattering_length(self,emin,emax,center,**args):
        def gauss_fit(x,x1,x2,A1,A2,s,y0):
            y=A1*np.exp(-(x-x1)**2/s/2)+A2*np.exp(-(x-x2)**2/s/2)+y0
            return y
        
        def line_fit(x,a,b):
            y=a*x+b
            return y
        
        def edependent_line_fit(x,a,b,c):
            tempx=h/np.sqrt(x-c)/np.sqrt(2)
            y=a*tempx+b
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
            
        if 'linear_fit' in args:
            linear_fit='e_dependent'
        else:
            linear_fit='e_independent'
        
        if 'overlay_peaks' in args:
            overlay_peaks=args['overlay_peaks']
        else:
            overlay_peaks=True
            
        if 'onset' in args:
            onset_energy=args['onset']
        else:
            onset_energy=0.0
            
        #scatter_side determines which side of the scattering source to model
        #default is both; options are left, right, both
        if 'scatter_side' in args:
            scatter_side=args['scatter_side']
        else:
            scatter_side='both'
            
        if 'plot_fits' in args:
            plot_fits=args['plot_fits']
            for i in range(len(plot_fits)):
                plot_fits[i]=np.argmin(abs(self.energy-plot_fits[i]))
            if not hasattr(self,'fig_eslice'):
                self.plot_energy_slice([self.energy[j] for j in plot_fits])
        else:
            plot_fits=[]
            
        energies=[]
        lengths=[]
        errors=[]
        peak_pos=[]
        peak_energies=[]
        peak_errors=[]
        
        for i in range(emin,emax+1):
            p0=[(self.pos[center]+self.pos[xmin])/2,(self.pos[xmax]+self.pos[center])/2,max(self.LIAcurrent[i,xmin:xmax])-min(self.LIAcurrent[i,xmin:xmax]),max(self.LIAcurrent[i,xmin:xmax])-min(self.LIAcurrent[i,xmin:xmax]),0.5,min(self.LIAcurrent[i,xmin:xmax])]
            bounds=([self.pos[xmin],self.pos[center],0,0,0,-np.inf],[self.pos[center],self.pos[xmax],np.inf,np.inf,np.inf,np.inf])
            popt,pcov=curve_fit(gauss_fit,self.pos[xmin:xmax],self.LIAcurrent[i,xmin:xmax],p0=p0,bounds=bounds)
            self.LIAfit[i,:]+=gauss_fit(self.pos,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])
            
            if i in plot_fits:
                self.ax_eslice.plot(self.pos,self.LIAfit[i],label='{} eV'.format(self.energy[i]))
                         
            energies.append(self.energy[i])
            pcov=np.sqrt(np.diag(pcov))
            if scatter_side=='both':
                lengths.append(abs(popt[0]-popt[1]))
                errors.append(np.sqrt(pcov[0]**2+pcov[1]**2))
                temprange=range(2)
            if scatter_side=='left':
                lengths.append(self.pos[center]-popt[0])
                errors.append(pcov[0])
                temprange=range(1)
            if scatter_side=='right':
                lengths.append(popt[1]-self.pos[center])
                errors.append(pcov[1])
                temprange=range(1,2)

            for j in temprange:
                peak_pos.append(popt[j])
                peak_energies.append(self.energy[i])
                peak_errors.append(pcov[j])
            
        if overlay_peaks:
            self.ax_main.errorbar(peak_pos,peak_energies,xerr=peak_errors,fmt='o')
            
        self.fig_fit,self.ax_fit=plt.subplots(1,1,tight_layout=True)
        energies=np.array(energies)
        lengths=np.array(lengths)
        errors=np.array(errors)
        k=1.6022e-19 #J/eV
        energies-=onset_energy
        energies*=k
        lengths*=1e-10
        if scatter_side!='both':
            lengths*=2.0
        errors*=1e-10
        h=6.626e-34 #J*s
        m=9.10938356e-31 #kg
        tempx=h/np.sqrt(energies)/np.sqrt(2)
        self.ax_fit.scatter(tempx,lengths,label='raw data')
        if linear_fit=='e_independent':
            popt,pcov=curve_fit(line_fit,tempx,lengths,p0=[3/np.sqrt(m),0.0],sigma=errors)
            self.ax_fit.plot(tempx,line_fit(tempx,popt[0],popt[1]),label='fit')
        else:
            popt,pcov=curve_fit(edependent_line_fit,energies,lengths,p0=[3/np.sqrt(m),0.0,0.0],sigma=errors)
            self.ax_fit.plot(tempx,edependent_line_fit(energies,popt[0],popt[1],popt[2]),label='fit')
        self.ax_fit.legend()
        self.ax_fit.set(xlabel='$2^{-1/2}$h$E^{-1/2}$ / m $kg^{1/2}$')
        self.ax_fit.set(ylabel='d / m')
        self.fig_fit.show()
        pcov=np.sqrt(np.diag(pcov))
        print('m* = {} +/- {}'.format(popt[0]**-2/m,pcov[0]/popt[0]**3/m))
        print('R = {} +/- {} Angstroms'.format(popt[1]*1e10,pcov[1]*1e10))
            
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