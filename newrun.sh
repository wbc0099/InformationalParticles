#! /bin/bash
#!/bin/bash

# 定义每个变量的值，可以根据需要进行修改
boxX=40
boxY=$boxX
cellListCountX=10
cellListCountY=$cellListCountX
cellListSizeX=4
cellListSizeY=$cellListSizeX
maxParticlesPerCell=2000
rForce=0
rNeighborList=2.5 #automatic
minDistance=0.1 # no use
equilibriumDistance=0.1 
forceCoefficient=1 #epsilon
kBT=0.0000
viscosityCoefficient=1
neighborUpdateThreshold=0.1
totalParticles=40000
startTime=0
endTime=0.05
timeStep=0.00005
tExpo=0.0005
rOff=1.0
rOffIn=0
n=5
forceCoefficient2=0 #0.01 #calculate interaction force
plotStep=5 #30
openImgVideoDirect=0
kBTChangeMode=1
kBTChangePM0=220
visionConeXLen=-1


# 检查是否传入了参数
if [ -z "$1" ]; then
    nthGPU=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | \
    awk -F "," '$2 == 0 {print $1; exit} {if (NR == 1 || $2 < min) {min = $2; gpu = $1}} END {print gpu}')
    nthGPU=$(echo $nthGPU | awk '{print $1}')


    if [ -z "$nthGPU" ]; then
        echo "未找到 GPU"
    else
        echo "选择的 GPU：$nthGPU"
    fi
else
    nthGPU=$1
fi

time=$(date "+%Y%m%d-%H%M%S")
dirName="boxX${boxX}_boxY${boxY}_particles${totalParticles}_endTime${endTime}_kBT${kBT}_N${n}_forceCoefficient${forceCoefficient}_timeStep${timeStep}_rOff${rOff}_rOffInner${rOffIn}_Mode${kBTChangeMode}_AroundNumMiniKBT${kBTChangePM0}_VisionConeXLen${visionConeXLen}"
echo $dirName
mkdir $dirName
cp kernel.cu $dirName
cp newrun.sh $dirName
cd $dirName

# 创建并写入 input.dat 文件
cat > input.dat << EOF
# box X length
$boxX
# box Y length
$boxY
# x方向celllist大小
$cellListSizeX
# y方向celllist大小
$cellListSizeY
# x方向celllist个数
$cellListCountX
# y方向celllist个数
$cellListCountY
# 每个cell里粒子最大个数
$maxParticlesPerCell
# force距离
$rForce
# 每个粒子间的最小距离，用于位运算
$minDistance
# 粒子平衡距离r0
$equilibriumDistance
# 作用力的系数
$forceCoefficient
# kBT
$kBT
# 粘滞系数
$viscosityCoefficient
# 近邻表更新临界位移
$neighborUpdateThreshold
# 粒子总数
$totalParticles
# 开始时间
$startTime
# 结束时间
$endTime
# 时间步长
$timeStep
# tExpo
$tExpo
# nthGPU
$nthGPU
# rOff
$rOff
# rOffInner
$rOffIn
# N
$n
# forceCoefficient2
$forceCoefficient2
# rNeighborList
$rNeighborList
# kBTChangeMode
$kBTChangeMode
# kBTChangePM0
$kBTChangePM0
# visionConeXLen
$visionConeXLen
# openImgVideoDirect 
$openImgVideoDirect
EOF

echo "input.dat 文件已生成。"

nvcc kernel.cu --disable-warnings -o kernel 
./kernel

PYTHON_CMD=$(command -v python || command -v python3 || echo "")
if [[ -z "$PYTHON_CMD" ]]; then
    echo "Python is not installed"
    exit 1
fi

echo -e "plotStep is:  $plotStep\n"
cd .. && "$PYTHON_CMD" plot.py "$dirName" "$plotStep" "0"
#绘制粒子轨迹
#"$PYTHON_CMD" trace.py "$dirName" 1 
folderTimeFormat=$(date +%-m.%-d)  # 不带前导零的月份和日期
mkdir "$folderTimeFormat"
mv box* "$folderTimeFormat"

