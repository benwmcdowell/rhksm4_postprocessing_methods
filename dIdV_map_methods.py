import rhksm4
import matplotlib.pyplot as plt
import numpy as np

class dIdV_map():
    def __init__(self,ifile):
        f=rhksm4.load(ifile)