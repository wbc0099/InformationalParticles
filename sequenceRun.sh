#!/bin/bash
#bash sequenceRun.sh
#改变文件内容，根据是否有GPU空闲，提交任务
#运行bgRun.sh脚本，提交任务到GPU上
#循环检测GPU是否空闲，有空闲GPU则提交任务，否则等待

#!/bin/bash

# 定义任务提交函数
submit_task() {
    local gpu_id=$1
    local particleNum=$2
    local theta=$3
    echo "Submitting task to GPU $gpu_id with particle number $particleNum."
    # 修改 run.sh 参数
    sed -i "s!totalParticles=[0-9]*!totalParticles=${particleNum}!" ./newrun.sh
    sed -i "s!theta=[0-9]*!theta=${theta}!" ./newrun.sh
    # 提交任务
    bash bgRun.sh $gpu_id
}

# 定义 GPU 空闲判定条件
is_gpu_free() {
    local gpu_id=$1
    # 检查 GPU 的内存使用情况，空闲条件可以根据需求调整
    local usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id)
    if [ "$usage" -lt 50 ]; then  # 这里假设内存使用小于 500MB 表示空闲
        return 0  # GPU 空闲
    else
        return 1  # GPU 正在使用
    fi
}

# 循环检测 GPU 和任务提交
# for i in 200 500 1000 1500 2000 2500 3000; do
# for i in 12000 14000 16000 18000; do
#     for j in 45 90 135 180; do
#         particelNum=$(bc <<< "scale=1; $i")
#         theta=$(bc <<< "scale=1; $j")

#         while true; do
#             available_gpus=()
#             for gpu_id in $(nvidia-smi --query-gpu=index --format=csv,noheader,nounits); do
#                 if is_gpu_free $gpu_id; then
#                     available_gpus+=($gpu_id)
#                 fi
#             done

#             if [ ${#available_gpus[@]} -gt 0 ]; then
#                 for gpu_id in "${available_gpus[@]}"; do
#                     echo "GPU $gpu_id is free."
#                     submit_task $gpu_id $particelNum $theta
#                     sleep 10  # 避免短时间重复提交
#                     break 2  # 跳出两个循环
#                 done
#             else
#                 echo "Waiting for available GPUs..."
#                 sleep 10  # 等待间隔
#             fi
#         done
#     done
# done

循环检测 GPU 和任务提交
for i in 6000 9000 12000 15000 18000; do
    for j in 45 90 135 180; do
        particelNum=$(bc <<< "scale=1; $i")
        theta=$(bc <<< "scale=1; $j")

        while true; do
            available_gpus=()
            for gpu_id in $(nvidia-smi --query-gpu=index --format=csv,noheader,nounits); do
                if is_gpu_free $gpu_id; then
                    available_gpus+=($gpu_id)
                fi
            done

            if [ ${#available_gpus[@]} -gt 0 ]; then
                for gpu_id in "${available_gpus[@]}"; do
                    echo "GPU $gpu_id is free."
                    submit_task $gpu_id $particelNum $theta
                    sleep 10  # 避免短时间重复提交
                    break 2  # 跳出两个循环
                done
            else
                echo "Waiting for available GPUs..."
                sleep 10  # 等待间隔
            fi
        done
    done
done