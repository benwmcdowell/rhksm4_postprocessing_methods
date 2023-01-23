import rhksm4
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter,find_peaks
from scipy.interpolate import griddata
from pathos.multiprocessing import ProcessPool
import json
import os

class topography:
    def __init__(self,ifile,invert=False,**args):
        #scale factor should be input as tuple: (xscale,yscale,zscale)
        if 'scalefactor' in args:
            self.sf=args['scalefactor']
        else:
            self.sf=(1.0,1.0,1.0)
            
        if 'scan_direction' in args:
            self.scan_direction=args['scan_direction']
        else:
            self.scan_direction='forward'
            
        self.f=rhksm4.load(ifile)
        if self.scan_direction=='forward':
            self.data=(self.f[2].data*self.f[2].attrs['RHK_Zscale']+self.f[2].attrs['RHK_Zoffset'])*1.0e9/self.sf[2]
        else:
            self.data=(self.f[3].data*self.f[3].attrs['RHK_Zscale']+self.f[3].attrs['RHK_Zoffset'])*1.0e9/self.sf[2]
            
        self.x=np.array([self.f[3].attrs['RHK_Xoffset']+i*self.f[3].attrs['RHK_Xscale'] for i in range(np.shape(self.data)[1])])*1.0e9/self.sf[0]
        self.y=np.array([self.f[3].attrs['RHK_Yoffset']+i*self.f[3].attrs['RHK_Yscale'] for i in range(np.shape(self.data)[0])])*1.0e9/self.sf[1]
        
        if invert:
            self.data*=-1.0
        
        self.data-=np.min(self.data)
        self.x-=np.min(self.x)
        self.y-=np.min(self.y)
        self.npts=np.shape(self.data)
        
    def add_savgol_filter(self,w,o,**args):
        for i in range(self.npts[0]):
            self.data[i,:]=savgol_filter(self.data[i,:],w,o)
        for i in range(self.npts[1]):
            self.data[:,i]=savgol_filter(self.data[:,i],w,o)
        
    def line_slope_subtract(self,**args):
        def linear_fit(x,a,b):
            y=a*x+b
            return y
        
        if 'range_select' in args:
            fit_range=np.array(args['range_select'])
            for i in range(2):
                for j in range(2):
                    fit_range[i,j]=np.argmin(abs(fit_range[i,j]-self.x))
        else:
            fit_range=np.array([[0,self.npts],[0,self.npts]])
            
        fit_exclude=[[],[]]
        if 'range_exclude' in args:
            for i in range(2):
                temprange=[np.argmin(abs(args['range_exclude'][i][j]-[self.x,self.y][j])) for j in range(2)]
                for j in range(min(temprange),max(temprange)):
                    fit_exclude[i].append(j)
        
        if 'slope_subtract_range' in args:
            slope_subtract_range=args['slope_subtract_range']
            slope_subtract_range=np.argmin(abs(args['slope_subtract_range']-self.y))
            print('averaging {} lines together in slope subtract'.format(2*slope_subtract_range+1))
        else:
            slope_subtract_range=0
            
        for i in range(self.npts[0]):
            tempdata=[]
            tempx=[]
            for j in range(self.npts[1]):
                for k in range(-1*slope_subtract_range,slope_subtract_range+1):
                    if i+k>=0 and i+k<len(self.x):
                        if i+k in fit_exclude[0] and j in fit_exclude[1]:
                            pass
                        else:
                            tempdata.append(self.data[i+k,j])
                            tempx.append(self.x[j])
            popt,pcov=curve_fit(linear_fit,tempx,tempdata)
            yfit=linear_fit(self.x,popt[0],popt[1])
            self.data[i,:]-=yfit
            
    def plot_horizontal_slice(self,pos):
        if not hasattr(self,'fig_hslice'):
            self.fig_hslice,self.ax_hslice=plt.subplots(1,1,tight_layout=True)
        if type(pos)==list:
            for p in pos:
                i=np.argmin(abs(self.x-p))
                self.ax_hslice.plot(self.x,self.data[i,:],label='{} nm'.format(p))
                self.ax_main.plot([self.x[0],self.x[-1]],[p,p])
        else:
            i=np.argmin(abs(self.x-pos))
            self.ax_hslice.plot(self.x,self.data[i,:])
            self.ax_main.plot([self.x[0],self.x[-1]],[pos,pos])
        self.ax_hslice.set(xlabel='position / $\AA$')
        self.ax_hslice.set(ylabel='topography height / nm')
        self.ax_hslice.legend()
        self.fig_hslice.show()
        
    def plot_vertical_slice(self,pos):
        if not hasattr(self,'fig_hslice'):
            self.fig_vslice,self.ax_vslice=plt.subplots(1,1,tight_layout=True)
        if type(pos)==list:
            for p in pos:
                i=np.argmin(abs(self.y-p))
                self.ax_vslice.plot(self.y,self.data[:,i],label='{} nm'.format(p))
                self.ax_main.plot([p,p],[self.y[0],self.y[-1]])
        else:
            i=np.argmin(abs(self.y-pos))
            self.ax_vslice.plot(self.y,self.data[:,i])
            self.ax_main.plot([pos,pos],[self.y[0],self.y[-1]])
        self.ax_vslice.set(xlabel='position / $\AA$')
        self.ax_vslice.set(ylabel='topography height / nm')
        self.ax_vslice.legend()
        self.fig_vslice.show()
        
    def plot_topo(self,norm=True,**args):
        if 'cmap' in args:
            cmap=args['cmap']
        else:
            cmap=plt.rcParams['image.cmap']
            
        if norm:
            self.data/=np.max(self.data)
            self.data-=np.min(self.data[np.nonzero(self.data)])
            
        self.fig_main,self.ax_main=plt.subplots(1,1,tight_layout=True)
        self.ax_main.pcolormesh([self.x for i in range(len(self.y))],[[self.y[j] for i in range(len(self.x))] for j in range(len(self.y))],self.data,cmap=cmap,shading='nearest')
        self.ax_main.set(xlabel='position / nm')
        self.ax_main.set(ylabel='position / nm')
        self.ax_main.set_aspect('equal')
        self.fig_main.show()
        
    def drift_correct(self,v):
        coord=np.array([[self.x[j],self.y[i]] for i in range(self.npts[0]) for j in range(self.npts[1])])
        raw_data=np.array([self.data[i,j] for i in range(self.npts[0]) for j in range(self.npts[1])])
        
        drift_coord=np.array([[self.x[j]+v[0]*(j+self.npts[0]*i),self.y[i]+v[1]*(j+self.npts[0]*i)] for i in range(self.npts[0]) for j in range(self.npts[1])])
        
        self.data=griddata(drift_coord,raw_data,coord,method='nearest',fill_value=0.0).reshape(self.npts[0],self.npts[1])
        
    def take_2dfft(self,**args):
        scaling='linear'
        if 'scaling' in args:
            if args['scaling']=='log':
                scaling='log'
            if args['scaling']=='sqrt':
                scaling='sqrt'
            
        self.fdata=np.fft.fftshift(abs(np.fft.fft2(self.data)))
        self.fx=np.fft.fftshift(np.fft.fftfreq(self.npts[1],abs(self.x[-1]-self.x[0])/(self.npts[1]-1)))*np.pi*2
        self.fy=np.fft.fftshift(np.fft.fftfreq(self.npts[0],abs(self.y[-1]-self.y[0])/(self.npts[0]-1)))*np.pi*2
        
        if scaling=='log':
            self.fdata=np.log(self.fdata)
        if scaling=='sqrt':
            self.fdata=np.sqrt(self.fdata)
            
    #default of filter_type is circle: argument is the radius of the circle to exclude
    #if filter_type=rectangle: argument should be a tuple containing the width and height of the filter
    def filter_2dfft(self,dim,filter_shape='circle',filter_type='pass'):
        if filter_type=='pass':
            filter_scale=1.0
        elif filter_type=='cut':
            filter_scale=0.0
            
        for i in range(self.npts[0]):
            for j in range(self.npts[1]):
                if filter_shape=='circle':
                    if np.linalg.norm(np.array([self.fx[j],self.fy[i]]))<dim:
                        self.fdata[i,j]*=filter_scale
                if filter_shape=='square':
                    if abs(self.fy[i])<dim[1] and abs(self.fx[i])<dim[0]:
                        self.fdata[i,j]*=filter_scale
        
    def find_2dfft_peaks(self,height,distance,mag=0.0,dmag=0.2):
        #mag selects the magnitude of reciprocal lattice vector that is returned. if mag is zero, all peaks are returned. otherwise, only peaks within dmag of mag are returned
        self.peak_list=[]
        for i in range(self.npts[1]):
            for j in range(self.npts[0]):
                if i in find_peaks(self.fdata[:,j],height=height,distance=distance)[0] and j in find_peaks(self.fdata[i,:],height=height,distance=distance)[0]:
                    if mag==0.0 or abs(mag-np.linalg.norm(np.array([self.fx[j],self.fy[i]])))<dmag:
                        self.peak_list.append(np.array([self.fx[j],self.fy[i]]))
                        
                    
        self.peak_list=np.array(self.peak_list)
        #print('peaks found at:')
        #for i in self.peak_list:
        #    print(i)
            
    def plot_2dfft(self,**args):
        if 'cmap' in args:
            cmap=args['cmap']
        else:
            cmap=plt.rcParams['image.cmap']
            
        if 'normalize' in args:
            if args['normalize']==False:
                normalize=False
            else:
                normalize=True
        else:
            normalize=True
            
        if normalize:
            self.fdata-=np.min(self.fdata)
            self.fdata/=np.max(self.fdata)
            
        self.fig_fft,self.ax_fft=plt.subplots(1,1,tight_layout=True)
        self.ax_fft.pcolormesh([self.fx for i in range(len(self.fy))],[[self.fy[j] for i in range(len(self.fx))] for j in range(len(self.fy))],self.fdata,cmap=cmap,shading='nearest')
        self.ax_fft.set(xlabel='position / 2$\pi$ $nm^{-1}$')
        self.ax_fft.set(ylabel='position / 2$\pi$ $nm^{-1}$')
        self.ax_fft.set_aspect('equal')
        self.fig_fft.show()
        
    def opt_drift_via_lattice(self,dpts,drange,mag,lattice_angle=90,angle_tol=0.1,scaling='sqrt',height=0.8,distance=5,nprocs=1):
        min_angle=lattice_angle
        min_drift=np.array([0.0,0.0])
        self.dpts=dpts
        self.drange=drange
        self.mag=mag
        self.lattice_angle=lattice_angle
        self.angle_tol=angle_tol
        self.fft_scaling=scaling
        self.height=height
        self.distance=distance
        self.nprocs=nprocs
        
        pool=ProcessPool(self.nprocs)
        output=pool.map(self.drift_correct_and_find_angle, [i for i in range(-dpts,dpts+1) for j in range(-dpts,dpts+1)], [j for i in range(-dpts,dpts+1) for j in range(-dpts,dpts+1)])
        self.min_angles=np.array(output)[0,:]
        self.drifts=np.array(output)[1,:]
        pool.close()
        
    def write_drift_calc_output(self,ofile):
        with open(ofile,'w+') as f:
            f.write(json.dumps([list(i) for i in [self.min_angles,self.drifts[:,0],self.drifts[:,1]]]))
            
    def read_file(self,ifile):
        with open(ifile,'r') as f:
            data=json.load(f)
        self.min_angles=data[:,0]
        self.drifts=data[:,1:]
        
    def drift_correct_and_find_angle(self,i,j):
        v=np.array([i*self.drange/(self.dpts-1),j*self.drange/(self.dpts-1)])
        self.drift_correct(v)
        self.take_2dfft(scaling=self.fft_scaling)
        self.find_2dfft_peaks(self.peak_height,self.peak_distance,mag=self.mag)
        a=[]
        if len(self.peak_list)>3:
            for j in self.peak_list:
                angle=np.arctan(j[1]/j[0])/np.pi*180
                for k in a:
                    if abs(angle-k)<self.angle_tol:
                        break
                else:
                    a.append(angle)
            if len(a)>1:
                min_angle=abs(abs(a[0]-a[1])-self.lattice_angle)
            else:
                min_angle=self.lattice_angle
        else:
            min_angle=self.lattice_angle
                
        return min_angle,v
        
def calc_drift_from_dir(dpath,w=5,o=3):
    files=os.listdir(dpath)
    os.chdir(dpath)
    peaks=[]
    for i in files:
        peaks.append([])
        topo=topography(i)
        topo.line_slope_subtract()
        topo.add_savgol_filter(w,o)
        for j in range(2,len(topo.y)-3):
            for k in range(2,len(topo.x)-3):
                if np.argmax(topo.data[j-2:j+3,k])==2 and np.argmax(topo.data[j,k-2:k+3])==2:
                    peaks[-1].append(np.array([topo.x[k],topo.y[j]]))
        peaks[-1]=np.array(peaks[-1])
    peaks=np.array(peaks)
        
    displacements=np.zeros(len(peaks)-1)
    for i in range(1,len(peaks)):
        tempvar=np.zeros(len(peaks[i])*len(peaks[i-1]))
        for j in range(len(peaks[i])):
            for k in range(len(peaks[i-1])):
                tempvar[j*len(peaks[i-1])+k]=np.linalg.norm(peaks[i][j]-peaks[i-1][k])
        displacements[i-1]=np.min(tempvar)
        
    print('95% CI for drift is {} nm unscaled'.format(np.average(displacements)+2*np.std(displacements)))
        
    return displacements,peaks