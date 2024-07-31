#!/bin/bash

#default values
ddir="./output_equil"
dLx=25.0
dLy=25.0
dLz=25.0
doutputf="positions.xyz"

#read arguments
directory=${1:-$ddir}
Lx=${2:-$dLx}
Ly=${3:-$dLy}
Lz=${4:-$dLz}
outputf=${5:-$doutputf}

#delete any previous files
rm $outputf 2> /dev/null
#rm test0.txt 2> /dev/null
rm test1.txt 2> /dev/null

#iterate through xyz files
i=0
for file in "$directory"/*.xyz
do
	#count the number of atoms in files
	numatoms=$(awk 'FNR > 2 {print $1}' $file | wc -l | awk '{print $1}')
	#ids from 1 to $numatoms
	#a=$(seq -s "\n" $numatoms)
	#echo -e $a > test0.txt
	#types are all 1
	shuf -i 1-1 -n ${numatoms} -r > test1.txt
	
	#write lammps trajectory file headers
	echo "$numatoms" >> $outputf
	echo "frame $i" >> $outputf
	
	awk -v a="$numatoms" 'FNR > 2 && FNR < a+3 { print $2, $3, $4 }' $file > "tmp$i.txt"
	paste test1.txt tmp$i.txt >> $outputf
	rm tmp$i.txt	
	echo "$file done!"
	i=$(echo "$i+1" | bc -l)
done

#delete temporary files
#rm test0.txt
rm test1.txt
	
