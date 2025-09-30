import matplotlib.pyplot as plt
import sys
import os
import shutil
from moviepy import *

try:
    from moviepy.editor import ImageSequenceClip
except ImportError:
    print("ImageSequenceClip do not belong to moviepy.editor")
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

def main():
    if len(sys.argv) != 5 and len(sys.argv) != 4:
        print("Usage: python plot.py <path> <plotStep> <xlim> <ylim> or python plot.py <path> <plotStep>")
        sys.exit(1)
    elif len(sys.argv) == 5:
        path = sys.argv[1]
        plotStep = int(sys.argv[2])
        xlim = float(sys.argv[3])
        ylim = float(sys.argv[4]) 
    elif len(sys.argv) == 4:
        path = sys.argv[1]
        plotStep = int(sys.argv[2])
        kAdd=float(sys.argv[3])
        with open(os.path.join(path, "input.dat")) as f:
            lines = f.readlines()
            xlim = float(lines[1].strip())
            ylim = float(lines[3].strip())
            tExpo = float(lines[37].strip())
            kMin = float(lines[23].strip())
            kMax = kMin + kAdd
            N = float(lines[43].strip())
            rOff = float(lines[41].strip())
            openImgVideoDirect = int(lines[57].strip())
        print(f"xlim: {xlim}, ylim: {ylim}, tExpo: {tExpo}, kBT: {kMin}, kMax: {kMax}, N: {N}, rOff: {rOff}")
    
    picture_dir = os.path.join(path, "picture")

    # 删除旧的目录
    if os.path.exists(picture_dir):
        shutil.rmtree(picture_dir)
    os.makedirs(picture_dir)

    curren_folder = path
    item_names = os.listdir(curren_folder)
    item_names1 = [files for files in item_names if files.endswith(".dat") and 'conf' in files]
    item_names1 = sorted(item_names1, key=lambda x: int(x.split('_')[1].split('.')[0]))
    print("data number:",len(item_names1))

    itemNum = len(item_names1)
    if plotStep == 0:
        plotStep = itemNum // 100 + 1
        print("plotStep",plotStep)

    i = 0
    picture_names = []
    finalPicName = ""
   
    #unic
    kMax=0
    for item in [item_names1[itemNum//2],item_names1[0],item_names1[-1]]:
        with open(os.path.join(path, item)) as f:
            color_ = []
            for line in f:
                data = line.split()
                color_.append(float(data[2]))
        colorMax = max(color_)
        if colorMax > kMax:
            kMax = colorMax

    for item in item_names1:
        if i % plotStep == 0:
            x_, y_, color_ = [], [], []
            with open(os.path.join(path, item)) as f:
                for line in f:
                    data = line.split()
                    x_.append(float(data[0]))
                    y_.append(float(data[1]))
                    color_.append(float(data[2]))
            scatter = plt.scatter(x_, y_, s=0.2, c=color_, cmap="jet", vmin=kMin, vmax=kMax)
            plt.colorbar(scatter)
            plt.xlim(0, xlim)
            plt.ylim(0, ylim)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("t=" + str(i * tExpo))
            plt.axis('equal')
            plt.axis('off')
            plt.margins(0, 0)
            picture_name = os.path.join(picture_dir, f"{i}.png")
            plt.savefig(picture_name, dpi=300, pad_inches=0, bbox_inches='tight')
            picture_names.append(picture_name)
            plt.close()
            finalPicName = picture_name
        i += 1
        print(f"\rplotting...{i / itemNum * 100}%\t", end="")

    print("\npictureNum:",len(picture_names))
    #将最后一张图片复制出来
    distinationFinalPic = path.rstrip("/") + "kBT" + ".jpg"
    shutil.copy(finalPicName, distinationFinalPic)

    video_name = path.rstrip("/") + ".mp4"  # 使用mp4格式
    if not picture_names:
        print("No pictures found to create video.")
        return

    # 使用 moviepy 创建视频
    clip = ImageSequenceClip(picture_names, fps=10)  # 设置帧率为10
    clip.write_videofile(video_name, codec='libx264')  # 使用libx264编解码器

    # 可选地删除图片目录，如果不需要图片了
    shutil.rmtree(picture_dir)

    if openImgVideoDirect:
        os.system("eog "+ distinationFinalPic)
        os.system("vlc "+ video_name + " > /dev/null 2>&1" )

if __name__ == "__main__":
    main()
