#!/bin/bash

for dir in $1/*/; do
    dir_name="${dir%/}"
    echo "Processing directory: $dir_name"
    nohup python plot.py "$dir_name" 1 10 > ../plot.log &
done