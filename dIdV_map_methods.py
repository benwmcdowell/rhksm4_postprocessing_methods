import rhksm4
import matplotlib.pyplot as plt
import numpy as np

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
            
        self.f=rhksm4.load(ifile)
        self.npts=np.shape(self.f)
        if self.scan_direction=='forward':
            self.topo=(self.f[2].data*self.f[2].attrs['RHK_Zscale']+self.f[2].attrs['RHK_Zoffset'])*1.0e9/self.sf[2]
        else:
            self.topo=(self.f[3].data*self.f[3].attrs['RHK_Zscale']+self.f[3].attrs['RHK_Zoffset'])*1.0e9/self.sf[2]
            
        self.x=np.array([self.f[3].attrs['RHK_Xoffset']+i*self.f[3].attrs['RHK_Xscale'] for i in range(np.shape(self.data)[1])])*1.0e9/self.sf[0]
        self.y=np.array([self.f[3].attrs['RHK_Yoffset']+i*self.f[3].attrs['RHK_Yscale'] for i in range(np.shape(self.data)[0])])*1.0e9/self.sf[1]
        
        self.epoints=np.array([self.f[4].attrs['RHK_Xoffset']+self.f[4].attrs['RHK_Xscale']*i for i in range(np.shape(self.f[4].data)[1])])
        self.z=np.zeros((len(self.epoints),self.npts[0],self.npts[1]))
        for i in range(np.shape(self.f[4].data)[0]):
            for j in range(np.shape(self.f[4].data)[1]):
                self.z[j,i//self.npts[0],i%self.npts[0]]=self.f[4].data[i,j]
                
    def plot_maps(self,*args):
        if 'cmap' in args:
            cmap=args['cmap']
        else:
            cmap=plt.rcParams['image.cmap']
            
        for i in range(len(self.epoints)):
            plt.figure()
            plt.title('{} V'.format(self.epoints[i]))
            plt.pcolormesh(self.x,self.y,self.z[i],shading='nearest',cmap=cmap)
            plt.xlabel('position / nm')
            plt.ylabel('position / nm')
            plt.colorbar()
            plt.show()