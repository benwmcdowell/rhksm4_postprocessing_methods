import rhksm4
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter,find_peaks
from scipy.interpolate import griddata

class topography:
    def __init__(self,ifile,**args):
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
        
        self.data-=np.min(self.data)
        self.x-=np.min(self.x)
        self.y-=np.min(self.y)
        self.npts=np.shape(self.data)
        
    def add_savgol_filter(self,w,o,**args):
        for i in range(self.npts[0]):
            self.data[i,:]=savgol_filter(self.data[i,:],w,o)
        for i in range(self.npts[1]):
            self.data[:,i]=savgol_filter(self.data[:,i],w,o)
        
    def line_slope_subtract(self,*args):
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
                temprange=[np.argmin(abs(fit_exclude[i][j]-self.y)) for j in range(2)]
                for j in range(temprange[0],temprange[1]+1):
                    fit_exclude[i].append(j)
        
        for i in range(self.npts[0]):
            tempdata=[]
            tempx=[]
            for j in range(self.npts[1]):
                if i in fit_exclude[0] and j in fit_exclude[1]:
                    pass
                else:
                    tempdata.append(self.data[i,j])
                    tempx.append(self.x[j])
            popt,pcov=curve_fit(linear_fit,tempx,tempdata)
            yfit=linear_fit(self.x,popt[0],popt[1])
            self.data[i,:]-=yfit
        
    def plot_topo(self,**args):
        if 'cmap' in args:
            cmap=args['cmap']
        else:
            cmap=plt.rcParams['image.cmap']
            
        self.fig_main,self.ax_main=plt.subplots(1,1,tight_layout=True)
        self.ax_main.pcolormesh([self.x for i in range(len(self.y))],[[self.y[j] for i in range(len(self.x))] for j in range(len(self.y))],self.data,cmap=cmap,shading='nearest')
        self.ax_main.set(xlabel='position / nm')
        self.ax_main.set(ylabel='position / nm')
        self.ax_main.set_aspect('equal')
        self.fig_main.show()
        
    def drift_correct(self,v):
        coord=np.array([[self.x[j],self.y[i]] for j in range(self.npts[1]) for i in range(self.npts[0])])
        raw_data=np.array([self.data for j in range(self.npts[1]) for i in range(self.npts[0])])
        
        drift_coord=np.array([[self.x[j]+v[0]*(j+self.npts[1]*i),self.y[i]+v[1]*(j+self.npts[1]*i)] for j in range(self.npts[1]) for i in range(self.npts[0])])
        
        
        
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
            
    def find_2dfft_peaks(self,height,distance):
        self.peak_list=[]
        for i in range(self.npts[1]):
            for j in range(self.npts[0]):
                if i in find_peaks(self.fdata[:,j],height=height,distance=distance)[0] and j in find_peaks(self.fdata[i,:],height=height,distance=distance)[0]:
                    self.peak_list.append(np.array([self.fx[j],self.fy[i]]))
                    
        self.peak_list=np.array(self.peak_list)
        print('peaks found at:')
        for i in self.peak_list:
            print(i)
        
            
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