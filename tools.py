import numpy as np
import matplotlib.pyplot as plt
from math import *
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import os
import shutil
from moviepy import *

def readFile(filePath,printSize=1):
    try:
        with open(filePath,'r',encoding='utf-8') as file:
            data=np.loadtxt(filePath)
            if printSize==1:
                print(f"data shape: {data.shape}")
            return data
    except Exception as e:
        raise IOError(f"读取文件失败：{e}")

def plotParticles(x,y,xlim,ylim,figurePath,
                  dpi=300,x1=None,y1=None,
                  color1="red",color2="blue",
                  markerSize=5, propertyColor=None,
                  cmap='viridis',colorbarLabel=property,
                  label=False):
    fig,ax=plt.subplots()
    if propertyColor is not None:
        sc=ax.scatter(x,y,s=markerSize,c=propertyColor,
                      cmap=cmap,label='Primary')
        cbar=plt.colorbar(sc)
        cbar.set_label(colorbarLabel)
    else:
        ax.scatter(x,y,s=markerSize,c=color1,label="Primary")
    if x1 is not None and y1 is not None:
        ax.scatter(x1,y1,s=markerSize,c=color2,label='secondary')
    
    ax.set_xlim([0,xlim])
    ax.set_ylim([0,ylim])
    ax.set_aspect("equal")

    if x1 is not None and y1 is not None and label:
        ax.legend()
    
    plt.savefig(figurePath,dpi=dpi,bbox_inches='tight')
    plt.close()

def distancePBC(x1,x2,y1,y2,lx,ly):
    dx=x1-x2
    dy=y1-y2
    dx=dx-round(dx/lx)*lx
    dy=dy-round(dy/ly)*ly
    dr=sqrt(dx**2+dy**2)
    return dr

def DBSCANPeriodic(data,lxBox,lyBox,eps,minSamples,
                    picturePath="./periodic_dbscan.png"):
    print("data size:",data.shape)
    N=len(data)
    distanceMartix=np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            delta=distancePBC(data[i][0],data[j][0],
                              data[i][1],data[j][1],lxBox,lyBox)
            distanceMartix[i,j]=delta
            distanceMartix[j,i]=delta
    print(distanceMartix.shape)
    labels=DBSCAN(eps=eps,min_samples=minSamples,
                  metric='precomputed').fit_predict(distanceMartix)
    plotParticles(data[:,0], data[:,1], lxBox, lyBox,
                  picturePath, propertyColor=labels,
                  colorbarLabel="Cluster Labels", cmap='viridis')
    print("聚类结果统计:", np.unique(labels, return_counts=True))
    # 结果分析
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Found {n_clusters} clusters")
    print(f"Noise points: {np.sum(labels == -1)}")
    
    # 计算各团簇大小并找出最大团簇
    cluster_sizes = {}
    for label in labels:
        if label != -1:  # 排除噪声点
            cluster_sizes[label] = cluster_sizes.get(label, 0) + 1

    if cluster_sizes:  # 如果有找到团簇
        max_cluster_label = max(cluster_sizes, key=cluster_sizes.get)
        max_cluster_size = cluster_sizes[max_cluster_label]
        print(f"Largest cluster size: {max_cluster_size} particles")
        print(f"Largest cluster label: {max_cluster_label}")
    else:
        print("No clusters found (only noise points)")

def phi(data,lxBox,lyBox,threshold=1.2,filePath="./phi6.png",order=6,
       drawPciture=True,returnSumPhi=False):
    tree=cKDTree(data,boxsize=(lxBox,lyBox))
    neighbors=tree.query_ball_tree(tree, r=threshold)
    phi=np.zeros(len(data),dtype=np.complex128)
    for i in range(len(data)):
        xi,yi=data[i]
        neighborIndices=neighbors[i]
        neighborIndices=[j for j in neighborIndices if j != i]
        if len(neighborIndices) == 0:
            phi[i] = 0
            continue
        dx=data[neighborIndices, 0]-xi
        dy=data[neighborIndices, 1]-yi
        dx=dx-np.round(dx/lxBox)*lxBox
        dy=dy-np.round(dy/lyBox)*lyBox
        r=np.sqrt(dx**2 + dy**2)
        theta=np.arctan2(dy, dx)
        phi[i]=np.mean(np.exp(1j*order*theta))
    phiMagnitude = np.abs(phi)
    if drawPciture:
        plotParticles(data[:,0], data[:,1], lxBox, lyBox,
                    filePath, propertyColor=phiMagnitude,
                    colorbarLabel="|Phi6|", cmap='viridis')
    if returnSumPhi:
        return np.sum(phiMagnitude)

def readParameter(filePath,parameters):
    result=[]
    parametersSet=set(parameters)
    foundParameters={}
    try:
        with open(filePath,'r') as file:
            for line in file:
                line=line.strip()
                if not line or line.startswith("#"):
                    continue
                if '=' in line:
                    key,value=line.split("=",1)
                    key=key.strip()
                    if key in parametersSet:
                        value=value.split('#')[0].split('//')[0].split('%')[0].split(';')[0]
                        foundParameters[key]=value.strip()
                        if len(foundParameters)==len(parametersSet):
                            break
    except FileNotFoundError:
        print(f"error: file {filePath} is not exit")
        return [None] * len(parameters)
    except Exception as e:
        print(f"error with reading file: {e}")
        return [None] * len(parameters)
    return [foundParameters.get(name) for name in parameters]
        
def makeVideo(dataPath,startStr,endStr,
                xlim,ylim,plotStep=1,
                particleTypeMode=0,fps=10,shutilPicture=False):
    picturePath=os.path.join(dataPath,"pictures")
    if os.path.exists(picturePath):
        shutil.rmtree(picturePath)
    os.makedirs(picturePath)
    currentFolder=dataPath
    itemNames=os.listdir(currentFolder)
    itemNames=[file for file in itemNames if file.endswith(endStr) and file.startswith(startStr)]
    itemNames=sorted(itemNames,key=lambda x: int(x[len(startStr):-len(endStr)]))
    print(itemNames)
    pictureNames=[]
    i=0
    itemNum=len(itemNames)
    for item in itemNames:
        if i%plotStep==0:
            data=readFile(os.path.join(dataPath,item),printSize=0)
            pictureNames.append(f"{i}.png")
            if particleTypeMode!=0:
                data1=data[:particleTypeMode]
                data2=data[particleTypeMode:]
                plotParticles(data1[:,0],data1[:,1],xlim,ylim,
                os.path.join(picturePath,f"{i}.png"),x1=data2[:,0],y1=data2[:,1])
            else:
                plotParticles(data[:,0],data[:,1],xlim,ylim,
                os.path.join(picturePath,f"{i}.png"))
        i+=1
        progress = i / itemNum * 100
        print(f"\rplotting...{progress:6.2f}%", end="", flush=True)  # 固定宽度（如 6 字符）
    print("\n PictureNum=",len(pictureNames))
    if not pictureNames:
        raise IOError("None Pictures Found to Create Video.")
    
    clip=ImageSequenceClip([os.path.join(picturePath,pictureName) for pictureName in pictureNames],fps=fps)
    clip.write_videofile(os.path.join(dataPath,"video.mp4"),codec='libx264')
    if shutilPicture:
        shutil.rmtree(picturePath)
    else:
        print(f"Pictures saved in {picturePath}")

def readAllDatas(path,includeString,startstr,endstr,readStep=1):
    item_names=os.listdir(path)
    item_names=[files for files in item_names if files.endswith(endstr) and files.startswith(startstr)]
    item_names=sorted(item_names, key=lambda x: int(x.split('_')[1].split('.')[0]))
    print("item_names", item_names)
    data_=[]
    i=0
    for item in item_names:
        if i%readStep==0:
            data=readFile(os.path.join(path,item),printSize=0)
            data_.append(data)
        i+=1
    data_=np.array(data_)
    print(data_.shape)
    return(data_)
            

if __name__ == "__main__":
    data=readFile("./conf_200.dat")
    x=data[:,0]
    y=data[:,1]
    x1=x[:456]
    y1=y[:456]
    x2=x[456:]
    y2=y[456:]
    plotParticles(x1,y1,60,60,"./figure.jpg",x1=x2,y1=y2)
    print(distancePBC(1,9,1,9,10,10))
    DBSCANPeriodic(data,60,60,3,20)
    phi(data[0:456],60,60,1.2,"./phi6.png")
    print(readParameter("conf-data/test.dat",["python","rust"]))
    makeVideo("./conf-data","conf_",".dat",60,60,plotStep=10,particleTypeMode=456)
    readAllDatas("conf-data","conf","conf",".dat")

