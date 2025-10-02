from tools import *
import sys
import matplotlib.pyplot as plt

if __name__=="__main__":
    path=sys.argv[1]
    if len(sys.argv)==3:
        N=int(sys.argv[2])
    else: 
        N=0
    item_names=os.listdir(current_folder)
    data=readFile(path)
    data=data[N,:]
    with open(path+"/input.dat") as f:
        lines=f.readlines()
        xlim=float(lines[1].strip)
        ylim=float(lines[3].strip)

#plt.plot