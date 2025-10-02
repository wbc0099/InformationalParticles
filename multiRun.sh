#!/bin/bash

#for i in 7 9 11 13
for i in $(seq 25 5 65);
do
	#for j in 30 60 90 120 150 180
	#do
	sed -i "s/kBTChangeRho=[0-9.]\+/kBTChangeRho=$i/"  ./newrun.sh
	#sed -i "s/theta=[0-9.]\+/theta=$j/"  ./newrun.sh
	mkdir ../log
	./newrun.sh  
	#nohup ./newrun.sh > ../log/PM0-$i.txt 2>&1 &
	# nohup ./newrun.sh > ../log/PM0-$i-theta-$j.txt 2>&1 &
	echo PM0=$i finished
	# echo PM0=$i theta=$j finished
	sleep 2
	#done
done


