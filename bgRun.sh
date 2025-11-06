#!/bin/bash

#bash bgRun.sh [GPU_ID]
#用于在后台运行 run.sh 脚本，并将输出重定向到指定文件
#参数 GPU_ID 可选，指定使用的 GPU 编号

# 文件路径
file="newrun.sh"

# 提取 kBTMin 的值
# N=$(grep "^N=" "$file" | awk -F '=' '{print $2}')
# kBT=$(grep "^kBT=" "$file" | awk -F '=' '{print $2}')
num=$(grep "^totalParticles=" "$file" | awk -F '=' '{print $2}')
time=$(grep "^endTime=" "$file" | awk -F '=' '{print $2}')
theta=$(grep "^theta=" "$file" | awk -F '=' '{print $2}')

# 输出结果
# echo "N: $N"
# echo "kBT: $kBT"
echo "time: $time"
echo "num: $num"
echo "theta: $theta"


# 定义输出文件名
mkdir -p ../log
# name=./log/N_${N}_kbt_${kBT}_time_${time}_num_${num}.log
name=../log/time_${time}_num_${num}_theta_${theta}.log

# 启动 run.sh 并将输出重定向到指定文件
if [ -z "$1" ]; then
    nohup bash newrun.sh > "$name" 2>&1 &
else
    nohup bash newrun.sh $1 > "$name" 2>&1 &
fi

echo "The script has been launched. name: $name"
