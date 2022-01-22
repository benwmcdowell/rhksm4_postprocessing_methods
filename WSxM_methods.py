

class profile():
    def __init__(self,ifile):
        self.x=[]
        self.y=[]
        with open(ifile) as f:
            read_data=False
            while True:
                line=f.readline()
                if not line:
                    break
                if read_data:
                    line=line.split()
                    self.x.append(line[0])
                    self.y.append(line[1])
                if '[Header end]' in line:
                    read_data=True
