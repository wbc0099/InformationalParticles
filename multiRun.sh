#!/bin/bash
#for i in $(seq 0 5 70);

#for i in 5 15 25 
#do
#for j in 1.1 1.2 1.3 1.5 2.0 3.0
#do
#sed -i "s/kBTChangePM0=[0-9.]\+/kBTChangePM0=$i/"  ./newrun.sh
#sed -i "s/rOff=[0-9.]\+/rOff=$j/" ./newrun.sh
#bash newrun.sh > ./error.txt 2>&1
#echo PM0=$i finished
#echo rOff=$j finished
#done
#done

#for i in 7 9 11 13
for i in $(seq 25 5 65);
do
	for j in 30 60 90 120 150 180
	do
	sed -i "s/kBTChangeRho=[0-9.]\+/kBTChangeRho=$i/"  ./newrun.sh
	sed -i "s/theta=[0-9.]\+/theta=$j/"  ./newrun.sh
	mkdir ../log
	nohup ./newrun.sh > ../log/PM0-$i-theta-$j.txt 2>&1 &
	echo PM0=$i theta=$j finished
	sleep 2
	done
done


