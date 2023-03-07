import matplotlib.pyplot as plt
import rhksm4
from scipy.optimize import curve_fit
from scipy.fft import fft,fftfreq
from scipy.signal.windows import hann
from scipy.signal import savgol_filter
from scipy.special import j0,y0
import numpy as np
import pyperclip
import csv

class dIdV_line:
    def __init__(self,ifile,fb_off=True,norm_z=True,**args):
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
        self.pos=np.array([self.f[0].attrs['RHK_Xoffset']+i*self.f[0].attrs['RHK_Xscale'] for i in range(self.size)])*1.0e10/self.sf 
        self.fb_off=fb_off
        self.norm_z=norm_z
        self.peak_energies=[]
        self.bessel_energies=[]
        
        if self.fb_off:
            self.npts=int(np.shape(self.f[4].data)[1])
            self.LIAcurrent=np.array([self.f[4].data[self.size*self.line_num:self.size*(self.line_num+1)]])
            self.current=np.array([self.f[5].data[self.size*self.line_num:self.size*(self.line_num+1)]])
            self.energy=np.array([self.f[4].attrs['RHK_Xoffset']+i*self.f[4].attrs['RHK_Xscale'] for i in range(self.npts)])
            
        else:
            self.npts=int(np.shape(self.f[6].data)[1])
            self.LIAcurrent=np.array([self.f[6].data[self.size*self.line_num:self.size*(self.line_num+1)]])
            self.current=np.array([self.f[7].data[self.size*self.line_num:self.size*(self.line_num+1)]])
            self.energy=np.array([self.f[6].attrs['RHK_Xoffset']+i*self.f[6].attrs['RHK_Xscale'] for i in range(self.npts)])
            self.z_fbon=np.array([self.f[8].data[self.size*self.line_num:self.size*(self.line_num+1)]])
            self.z_fbon=self.z_fbon[0,:,:].T[::-1]*self.f[8].attrs['RHK_Zscale']*1e9
            if self.norm_z:
                for i in range(self.size):
                    self.z_fbon[:,i]-=np.min(self.z_fbon[:,i])
        
        self.energy=self.energy[::-1] #bias in V
        self.pos=abs(self.pos-max(self.pos)) #postion along line in $\AA$
        self.LIAcurrent=self.LIAcurrent[0,:,:].T[::-1]*1.0e12 #LIAcurrent in pA
        self.current=self.current[0,:,:].T[::-1]*1.0e12 #current in pA
        self.LIAfit=np.zeros((self.npts,self.size))
        
    def normalize(self,**args):
        if 'range' in args:
            norm_range=[]
            for i in args['norm_range']:
                norm_range.append(np.argmin(abs(i-self.energy)))
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
        
        self.periodic_potential_params=[]
        
        for i in range(self.npts):
            params=curve_fit(model_cosine,self.pos,self.LIAcurrent[i])
            fit=model_cosine(self.pos,params[0][0],params[0][1],params[0][2])
            self.periodic_potential_params.append(params)
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
        self.ax_main.set_xlim(np.min(self.pos)-(self.pos[1]-self.pos[0])/2, np.max(self.pos)-(self.pos[-1]-self.pos[-2])/2)
        self.ax_main.set_ylim(np.min(self.energy)-(self.energy[1]-self.energy[0])/2, np.max(self.energy)-(self.energy[-1]-self.energy[-2])/2)
        self.cbar_main=self.fig_main.colorbar(self.dIdVmap)
        self.cbar_main.set_label('LIA current / pA')
        self.fig_main.show()
        
        if not self.fb_off:
            self.fig_z,self.ax_z=plt.subplots(1,1,tight_layout=True)
            self.zmap=self.ax_z.pcolormesh(x,y,self.z_fbon,cmap=cmap,shading='nearest')
            self.ax_z.set(xlabel='position / $\AA$')
            self.ax_z.set(ylabel='bias / V')
            self.cbar_z=self.fig_z.colorbar(self.zmap)
            self.cbar_z.set_label('height / nm')
            self.fig_z.show()
        
    def clear_energy_axes(self):
        self.ax_eslice.clear()
        
    def plot_energy_slice(self,energy,plot_bessel_fits=False):
        if not hasattr(self,'fig_slice'):
            self.fig_eslice,self.ax_eslice=plt.subplots(1,1,tight_layout=True)
        counter=0
        if type(energy)==list:
            for e in energy:
                i=np.argmin(abs(self.energy-e))
                tempdata=self.ax_eslice.plot(self.pos,self.LIAcurrent[i],label='{} eV'.format(e))
                color=tempdata[0].get_color()
                self.ax_main.plot([self.pos[0],self.pos[-1]],[e,e],c=color)
                if self.energy[i] in self.peak_energies:
                    for j in range(len(self.peak_energies)):
                        if self.energy[i]==self.peak_energies[j]:
                            x=np.argmin(abs(self.peak_pos[j]-self.pos))
                            self.ax_eslice.errorbar(self.peak_pos[j],self.LIAcurrent[i,x],xerr=self.peak_errors[j],c='black',fmt='o')
                if self.energy[i] in self.bessel_energies and plot_bessel_fits:
                    for j in range(len(self.bessel_energies)):
                        if self.energy[i]==self.bessel_energies[j]:
                            self.ax_eslice.plot(self.bessel_x[j],self.bessel_y[j],c=color)
        else:
            i=np.argmin(abs(self.energy-energy))
            self.ax_eslice.plot(self.pos,self.LIAcurrent[i])
            self.ax_main.plot([self.pos[0],self.pos[-1]],[energy,energy])
        self.ax_eslice.set(xlabel='position / $\AA$')
        self.ax_eslice.set(ylabel='LIA current / pA')
        self.ax_eslice.legend()
        self.fig_eslice.show()
        
    def plot_position_slice(self,pos,plot_onsets=False,exclude_range=None,print_onsets=False,**args):
        def cos(x,a,b,f,phi):
            y=a*np.cos(f*2*np.pi*x+phi)+b
            return y
        
        def lorentzian(x,x0,g,a,y0):
            y=a*(0.5*g/((x-x0)**2+(.5*g)**2))+y0
            return y
        
        if 'find_onset' in args:
            find_onset=True
            onsets=[]
            onset_pos=[]
            onset_range=args['find_onset']
            for i in range(2):
                onset_range[i]=np.argmin(abs(self.energy-onset_range[i]))
        else:
            find_onset=False
            
        if exclude_range:
            if len(np.shape(exclude_range))==1:
                for i in range(2):
                    exclude_range[i]=np.argmin(abs(self.pos-exclude_range[i]))
                exclude_range=[exclude_range]
            else:
                for i in range(len(exclude_range)):
                    for j in range(2):
                        exclude_range[i][j]=np.argmin(abs(self.pos-exclude_range[i][j]))
        else:
            exclude_range=[[-5,-5]]
                        
        self.fig_pslice,self.ax_pslice=plt.subplots(1,1,tight_layout=True)
        if not self.fb_off:
            self.fig_pslice_z,self.ax_pslice_z=plt.subplots(1,1,tight_layout=True)
            
        try:
            tempvar=len(pos)
        except TypeError:
            pos=[pos]
        for p in pos:
            i=np.argmin(abs(self.pos-p))
            self.ax_main.plot([p,p],[self.energy[0],self.energy[-1]])
            self.ax_pslice.plot(self.energy,self.LIAcurrent[:,i],label='{} $\AA$'.format(p))
            if not self.fb_off:
                self.ax_pslice_z.plot(self.energy,self.z_fbon[:,i],label='{} $\AA$'.format(p))
            if find_onset:
                for j in exclude_range:
                    if p>j[0] and p<j[1]:
                        break
                    else:
                        onset_height=(np.max(self.LIAcurrent[onset_range[0]:onset_range[1],i])+np.min(self.LIAcurrent[onset_range[0]:onset_range[1],i]))/2
        
                        onsets.append(self.energy[onset_range[0]+np.argmin(abs(self.LIAcurrent[onset_range[0]:onset_range[1],i]-onset_height))])
                        self.ax_pslice.plot([onsets[-1] for j in range(2)],[min(self.LIAcurrent[[onset_range[0],onset_range[1]],i]),max(self.LIAcurrent[[onset_range[0],onset_range[1]],i])],label='onset')
                        onset_pos.append(p)
                        
        if find_onset:
            print('average 2d band onset: {} +/- {} eV'.format(np.mean(onsets),np.std(onsets)))
            
        if print_onsets:
            for i in onsets:
                print(i)
            
        if plot_onsets:
            self.fig_onsets,self.ax_onsets=plt.subplots(1,1,tight_layout=True)
            self.ax_onsets.scatter(onset_pos,onsets)
            self.ax_onsets.set(xlabel='position / $\AA$')
            self.ax_onsets.set(ylabel='bias / eV')
            self.fig_onsets.show()
            
        self.ax_pslice.set(xlabel='bias / eV')
        self.ax_pslice.set(ylabel='LIA current / pA')
        self.ax_pslice.legend()
        self.fig_pslice.show()
        
        if not self.fb_off:
            self.ax_pslice_z.set(xlabel='bias / eV')
            self.ax_pslice_z.set(ylabel='height / nm')
            self.ax_pslice_z.legend()
            self.fig_pslice_z.show()
            
    def overlay_bounds(self,pos):
        for i in pos:
            self.boundline=self.ax_main.plot([self.pos[0],self.pos[-1]],[i,i],color='white',linestyle='dashed')
        self.fig_main.canvas.draw()
        
    def overlay_center(self,pos):
        self.centerline=self.ax_main.plot([pos,pos],[self.energy[0],self.energy[-1]],color='white',linestyle='dashed')
        
    def find_scattering_length(self,emin,emax,center,exclude_from_fit=None,tempsf=1.0,**args):
        def gauss_fit(x,x1,x2,A1,A2,s,y0):
            y=A1*np.exp(-(x-x1)**2/s/2)+A2*np.exp(-(x-x2)**2/s/2)+y0
            return y
        
        def line_fit(x,a,b):
            #b=1.6342441425316037e-20
            y=a*x+b
            return y
        
        def edependent_line_fit(x,a,b,c):
            h=6.626e-34 #J*s
            y=a*h/np.sqrt(x-c)/np.sqrt(2)+b
            return y
        
        def bessel_fit(x,k,u,x0,b,max_val):
            #max_val=1
            y=-2*u*j0(k*abs(x-x0))*y0(k*abs(x-x0))+b
            if not self.exclude_from_fit:
                for i in range(len(y)):
                    if abs(y[i]-b)>max_val*abs(u):
                        y[i]=u*max_val+b
            return y
        
        center=np.argmin(abs(self.pos-center))
        emin=np.argmin(abs(self.energy-emin))
        emax=np.argmin(abs(self.energy-emax))
        
        if exclude_from_fit:
            self.exclude_from_fit=[np.argmin(abs(self.pos-i)) for i in exclude_from_fit]
        else:
            self.exclude_from_fit=None
        
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
            if linear_fit=='e_dependent':
                print('warning: onset energy is selected as an optimizable parameter and specified as an input value. remove onset from the input arguments or set linear_fit=e_independent')
        else:
            onset_energy=0.1
            
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
        pot_fit=[]
        pot_errors=[]
        k_fit=[]
        k_errors=[]
        x0_fit=[]
        x0_errors=[]
        bessel_x=[]
        bessel_y=[]
        bessel_energies=[]
        self.bessel_fit_params=[]
        self.bessel_fit_errors=[]
        
        for i in range(emin,emax+1):
            p0=[(self.pos[center]+self.pos[xmin])/2,(self.pos[xmax]+self.pos[center])/2,max(self.LIAcurrent[i,xmin:xmax])-min(self.LIAcurrent[i,xmin:xmax]),max(self.LIAcurrent[i,xmin:xmax])-min(self.LIAcurrent[i,xmin:xmax]),0.5,min(self.LIAcurrent[i,xmin:xmax])]
            bounds=([self.pos[xmin],self.pos[center],0,0,0,-np.inf],[self.pos[center],self.pos[xmax],np.inf,np.inf,np.inf,np.inf])
            popt,pcov=curve_fit(gauss_fit,self.pos[xmin:xmax],self.LIAcurrent[i,xmin:xmax],p0=p0,bounds=bounds)
            self.LIAfit[i,:]+=gauss_fit(self.pos,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])
            
            if i in plot_fits:
                self.ax_eslice.plot(self.pos,self.LIAfit[i],label='{} eV'.format(self.energy[i]))
                         
            energies.append(self.energy[i])
            pcov=np.sqrt(np.diag(pcov))
            
            bounds=([3/np.abs(self.pos[xmax]-self.pos[xmin]),-.01,self.pos[center]-30,-np.inf,0],[np.inf,.01,self.pos[center]+30,np.inf,0.8])
            
            if not self.exclude_from_fit:
                p0=[3/np.abs(popt[0]-popt[1]),-0.001,self.pos[center],np.average(self.LIAcurrent[i,xmin:xmax]),0.5]
                popt_b,pcov_b=curve_fit(bessel_fit,self.pos[xmin:xmax],self.LIAcurrent[i,xmin:xmax],p0=p0,bounds=bounds)
                bessel_x.append(self.pos[xmin:xmax])
                bessel_y.append(bessel_fit(self.pos[xmin:xmax],popt_b[0],popt_b[1],popt_b[2],popt_b[3],popt_b[4]))
            else:
                middle=np.average(np.concatenate((self.LIAcurrent[i,xmin:self.exclude_from_fit[0]],self.LIAcurrent[i,self.exclude_from_fit[1]:xmax])))
                peak=-1*np.max(abs(np.concatenate((self.LIAcurrent[i,xmin:self.exclude_from_fit[0]],self.LIAcurrent[i,self.exclude_from_fit[1]:xmax]))-middle))*2/5
                p0=[3/np.abs(popt[0]-popt[1]),peak,self.pos[center],middle,0.5]
                
                if scatter_side=='both':
                    popt_b,pcov_b=curve_fit(bessel_fit,np.concatenate((self.pos[xmin:self.exclude_from_fit[0]],self.pos[self.exclude_from_fit[1]:xmax])),np.concatenate((self.LIAcurrent[i,xmin:self.exclude_from_fit[0]],self.LIAcurrent[i,self.exclude_from_fit[1]:xmax])),p0=p0,bounds=bounds,maxfev=25000)
                    bessel_x.append(np.concatenate((self.pos[xmin:self.exclude_from_fit[0]],self.pos[self.exclude_from_fit[1]:xmax])))
                    bessel_y.append(bessel_fit(np.concatenate((self.pos[xmin:self.exclude_from_fit[0]],self.pos[self.exclude_from_fit[1]:xmax])),popt_b[0],popt_b[1],popt_b[2],popt_b[3],popt_b[4]))
                    
                if scatter_side=='left':
                    popt_b,pcov_b=curve_fit(bessel_fit,self.pos[xmin:self.exclude_from_fit[0]],self.LIAcurrent[i,xmin:self.exclude_from_fit[0]],p0=p0,bounds=bounds,maxfev=25000)
                    bessel_x.append(self.pos[xmin:self.exclude_from_fit[0]])
                    bessel_y.append(bessel_fit(self.pos[xmin:self.exclude_from_fit[0]],popt_b[0],popt_b[1],popt_b[2],popt_b[3],popt_b[4]))
                    
                if scatter_side=='right':
                    popt_b,pcov_b=curve_fit(bessel_fit,self.pos[self.exclude_from_fit[1]:xmax],self.LIAcurrent[i,self.exclude_from_fit[1]:xmax],p0=p0,bounds=bounds,maxfev=5000)
                    bessel_x.append(self.pos[self.exclude_from_fit[1]:xmax])
                    bessel_y.append(bessel_fit(self.pos[self.exclude_from_fit[1]:xmax],popt_b[0],popt_b[1],popt_b[2],popt_b[3],popt_b[4]))
                    
            pcov_b=np.sqrt(np.diag(pcov_b))
            self.bessel_fit_params.append(popt_b)
            self.bessel_fit_errors.append(pcov_b)
            k_fit.append(popt_b[0])
            pot_fit.append(popt_b[1])
            x0_fit.append(popt_b[2])
            k_errors.append(pcov_b[0])
            pot_errors.append(pcov_b[1])
            x0_errors.append(popt_b[2])
            bessel_energies.append(self.energy[i])
            
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
            
        self.peak_pos=peak_pos
        self.peak_energies=peak_energies
        self.peak_errors=peak_errors
        self.bessel_y=bessel_y
        self.bessel_x=bessel_x
        self.bessel_energies=bessel_energies
            
        self.fig_fit,self.ax_fit=plt.subplots(1,1,tight_layout=True)
        energies=np.array(energies)
        lengths=np.array(lengths)
        errors=np.array(errors)
        k_fit=np.array(k_fit)
        pot_fit=np.array(pot_fit)
        x0_fit=np.array(x0_fit)
        k_errors=np.array(k_errors)
        pot_errors=np.array(pot_errors)
        x0_errors=np.array(x0_errors)
        k=1.6022e-19 #J/eV
        if linear_fit=='e_independent':
            energies-=onset_energy
        energies*=k
        lengths*=1e-10
        lengths/=tempsf
        if scatter_side!='both':
            lengths*=2.0
        errors*=1e-10
        errors/=tempsf
        h=6.626e-34 #J*s
        hbar=h/2/np.pi
        m=9.10938356e-31 #kg
        self.energies=energies
        self.lengths=lengths
        self.errors=errors
        if linear_fit=='e_independent':
            tempx=h/np.sqrt(energies)/np.sqrt(2)
            popt,pcov=curve_fit(line_fit,tempx,lengths,p0=[3/np.sqrt(m),0.0],sigma=errors)
        else:
            popt,pcov=curve_fit(edependent_line_fit,energies,lengths,p0=[1/np.sqrt(m),5*1e-10,0.1*k],sigma=errors)
            tempx=h/np.sqrt(energies-popt[2])/np.sqrt(2)
        self.ax_fit.errorbar(tempx/np.sqrt(m)*1e9,lengths*1e9,yerr=errors*1e9,label='raw data',fmt='o')
        self.ax_fit.plot(tempx/np.sqrt(m)*1e9,line_fit(tempx,popt[0],popt[1])*1e9,label='fit')
        self.ax_fit.legend()
        self.ax_fit.set(xlabel='h$(2E$m_e$)^{-1/2}$ / nm')
        self.ax_fit.set(ylabel='d / nm')
        self.fig_fit.show()
        pcov=np.sqrt(np.diag(pcov))
        print('calculated from infinitely-tall potential')
        print('m* = {} +/- {}'.format(popt[0]**-2/m,pcov[0]/popt[0]**3/m))
        print('R = {} +/- {} Angstroms'.format(popt[1]*1e10,pcov[1]*1e10))
        if len(popt)>2:
            print('band onset = {} +/- {} eV'.format(popt[2]/k,pcov[2]/k))
            
        k_fit*=1e10*tempsf
        k_errors*=1e10*tempsf
        tempx=np.array([(k_fit**2)[i] for i in [0,-1]])
        tempy=np.array([(energies)[i] for i in [0,-1]])
        p0=((tempy[1]-tempy[0])/(tempx[1]-tempx[0]),tempy[0]-(tempy[1]-tempy[0])/(tempx[1]-tempx[0])*tempx[0])
        plt.plot(tempx,line_fit(tempx,p0[0],p0[1]))
        popt,pcov=curve_fit(line_fit,(k_fit)**2,energies,p0=p0,sigma=(k_errors)**2)
        pcov=np.sqrt(np.diag(pcov))
        self.fig_bfit,self.ax_bfit=plt.subplots(1,1,tight_layout=True)
        self.ax_bfit.errorbar(k_fit**2,energies,xerr=k_errors**2,fmt='o')
        self.ax_bfit.plot(k_fit**2,line_fit(k_fit**2,popt[0],popt[1]))
        self.fig_bfit.show()
        
        print('calculated from Dirac potential, Bessel functions:')
        print('m* = {} +/- {}'.format(hbar**2/popt[0]/2/m,hbar**2/popt[0]**2/2/m*pcov[0]*(hbar**2/popt[0]/2/m)))
        print('u = {} +/- {}'.format(np.average(pot_fit),np.sqrt(sum(pot_errors**2))))
            
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
        
    def add_savgol_filter(self,w,o,horizontal=True,vertical=True):
        if horizontal:
            for i in range(len(self.energy)):
                self.LIAcurrent[i]=savgol_filter(self.LIAcurrent[i],w,o)
        if vertical:
            for i in range(len(self.pos)):
                self.LIAcurrent[:,i]=savgol_filter(self.LIAcurrent[:,i],w,o)
            
    def copy_peaks(self,cutoff=1e-9):
        tempvar=''
        for i in range(len(self.errors)):
            if self.errors[i]<cutoff:
                for k in [self.energies[i],self.lengths[i],self.errors[i]]:
                    tempvar+=str(k)
                    tempvar+='\t'
                tempvar+='\n'
        pyperclip.copy(tempvar)
        
def read_peaks(fp,scatter_side='both',linear_fit='e_independent',onset_energy=0.1,erange=(-np.inf,np.inf)):
    erange=[i*1.6022e-19 for i in erange]
    file=open(fp,'r')
    csvfile=csv.reader(file,delimiter=',')
    lines=[]
    for i in csvfile:
        lines.append(i)
    lines=np.array(lines)
    energies=[]
    lengths=[]
    errors=[]
    for i in range(np.shape(lines)[1]):
        if 'energies' in lines[2,i]:
            for j in range(3,np.shape(lines)[0]):
                if len(lines[j,i])==0:
                    break
                else:
                    if float(lines[j,i])>np.min(erange) and float(lines[j,i])<np.max(erange):
                        energies.append(float(lines[j,i]))
        if 'lengths' in lines[2,i]:
            for j in range(3,np.shape(lines)[0]):
                if len(lines[j,i])==0:
                    break
                else:
                    if float(lines[j,i-1])>np.min(erange) and float(lines[j,i-1])<np.max(erange):
                        lengths.append(float(lines[j,i]))
        if 'errors' in lines[2,i]:
            for j in range(3,np.shape(lines)[0]):
                if len(lines[j,i])==0:
                    break
                else:
                    if float(lines[j,i-2])>np.min(erange) and float(lines[j,i-2])<np.max(erange):
                        errors.append(float(lines[j,i]))
            
    energies=np.array(energies)
    lengths=np.array(lengths)
    errors=np.array(errors)
    min_error=np.min(errors[np.nonzero(errors)])
    for i in range(len(errors)):
        if errors[i]==0:
            errors[i]=min_error
            
    fig_fit,ax_fit=plt.subplots(1,1,tight_layout=True)
    k=1.6022e-19 #J/eV
    h=6.626e-34 #J*s
    m=9.10938356e-31 #kg
    if linear_fit=='e_independent':
        tempx=h/np.sqrt(energies)/np.sqrt(2)
        popt,pcov=curve_fit(line_fit,tempx,lengths,p0=[2/np.sqrt(m),-1],sigma=errors)
    else:
        popt,pcov=curve_fit(edependent_line_fit,energies,lengths,p0=[1/np.sqrt(m),5*1e-10,0.1*k],sigma=errors)
        tempx=h/np.sqrt(energies-popt[2])/np.sqrt(2)
    ax_fit.errorbar(tempx/np.sqrt(m)*1e9,lengths*1e9,yerr=errors*1e9,label='raw data',fmt='o')
    ax_fit.plot(tempx/np.sqrt(m)*1e9,line_fit(tempx,popt[0],popt[1])*1e9,label='fit')
    ax_fit.legend()
    ax_fit.set(xlabel='h$(2E$m_e$)^{-1/2}$ / nm')
    ax_fit.set(ylabel='d / nm')
    fig_fit.show()
    pcov=np.sqrt(np.diag(pcov))
    print('m* = {} +/- {}'.format(popt[0]**-2/m,pcov[0]/popt[0]**3/m))
    print('R = {} +/- {} Angstroms'.format(popt[1]*1e10,pcov[1]*1e10))
    if len(popt)>2:
        print('band onset = {} +/- {} eV'.format(popt[2]/k,pcov[2]/k))
        
    return energies, lengths, errors
        
def line_fit(x,a,b):
    y=a*x+b
    return y

def edependent_line_fit(x,a,b,c):
    h=6.626e-34 #J*s
    y=a*h/np.sqrt(x-c)/np.sqrt(2)+b
    return y