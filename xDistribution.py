import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from math import *
from moviepy import *
import shutil

def makeVideo(path):
    item_name=os.listdir(path)
    item_name=[file for file in item_name if file.endswith(".png") and "pic" in file]
    item_name=sorted(item_name, key=lambda x: int(x.split(".")[0].split("c")[1]))
    videoName=(path+"/video.mp4")
    item_name=[path+"/"+item for item in item_name]
    print(item_name)
    clip=ImageSequenceClip(item_name,fps=10)
    clip.write_videofile(videoName, codec="libx264")
    


path=sys.argv[1]

item_name=os.listdir(path)
item_name=[file for file in item_name if file.endswith(".dat") and "conf" in file]
item_name=sorted(item_name, key=lambda x: int(x.split("_")[1].split(".")[0]))

x__,y__=[],[]
i=0
lenth=0.01
print(len(item_name))
# for item in item_name[:len(item_name)//5]:
for item in item_name:
    x_=np.zeros(ceil(1/lenth))
    print("%.2f%%\r"%(i*100/len(item_name)),end="")
    with open(os.path.join(path,item)) as f:
        for line in f:
            data=line.split()
            # x_.append(float(data[0]))
            # y_.append(float(floor(data[1]/len)))
            x_[ceil(float(data[0])/lenth)-1]+=1
    # x__.append(x_)
    # y__.append(y_)
    i+=1
    plt.plot(range(ceil(1/lenth)),x_)
    plt.ylim(150,240)
    plt.savefig("../pics/pic"+str(i), dpi=300)
    plt.close()

# x__=np.array(x__)
# print("\n",x__.shape)

makeVideo("../pics")