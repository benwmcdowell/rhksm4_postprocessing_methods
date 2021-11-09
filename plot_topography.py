import rhksm4
import numpy as np
import matplotlib.pyplot as plt

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
        self.npts=np.shape(self.f)
        if self.scan_direction=='forward':
            self.data=(self.f[2].data*self.f[2].attrs['RHK_Zscale']+self.f[2].attrs['RHK_Zoffset'])*1.0e9/self.sf[2]
        else:
            self.data=(self.f[3].data*self.f[3].attrs['RHK_Zscale']+self.f[3].attrs['RHK_Zoffset'])*1.0e9/self.sf[2]
            
        self.x=np.array([self.f[3].attrs['RHK_Xoffset']+i*self.f[3].attrs['RHK_Xscale'] for i in range(np.shape(self.data)[1])])*1.0e9/self.sf[0]
        self.y=np.array([self.f[3].attrs['RHK_Yoffset']+i*self.f[3].attrs['RHK_Yscale'] for i in range(np.shape(self.data)[0])])*1.0e9/self.sf[1]
        
        self.data-=np.min(self.data)
        self.x-=np.min(self.x)
        self.y-=np.min(self.y)
        
    def plot_topo(self,**args):
        if 'cmap' in args:
            cmap=args['cmap']
        else:
            cmap=plt.rcParams['image.cmap']
            
        self.fig_main,self.ax_main=plt.subplots(1,1,tight_layout=True)
        self.ax_main.pcolormesh([self.x for i in range(len(self.y))],[[self.y[j] for i in range(len(self.x))] for j in range(len(self.y))],self.data,cmap=cmap,shading='nearest')
        self.ax_main.set(xlabel='position / nm')
        self.ax_main.set(ylabel='position / nm')
        self.fig_main.show()