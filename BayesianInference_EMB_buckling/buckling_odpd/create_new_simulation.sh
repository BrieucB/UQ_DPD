#!/bin/bash

mkdir -p simulations

dirname=${1:-"test_sim"}

mkdir -p simulations/$dirname

newdir="simulations/$dirname"

rsync -avP src/* ${newdir} #--exclude=create_new_simulation.sh
