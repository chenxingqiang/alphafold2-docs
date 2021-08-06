#!/bin/sh

if [ $# -lt 1 ]; then
        echo $0 modelFolder [savefile]
	echo "modelFolder: a folder for relaxed models generated by PyRosetta"
	echo "savefile: the file for result saving, default is the basename of modelFolder followed by .score"
        exit 1 
fi

if [[ -z "$DistanceFoldingHome" ]]; then
	echo "ERROR: please set environmental variable DistanceFoldingHome, e.g., $HOME/RaptorX-3DModeling/Folding"
	exit 1
fi

modelFolder=$1
if [ ! -d $modelFolder ]; then
	echo "ERROR: $modelFolder is not a valid folder!"
	exit 1
fi

scorefile=`basename $modelFolder`.score
if [ $# -ge 2 ]; then
	scorefile=$2
fi

cp /dev/null $scorefile
for i in $modelFolder/*.pdb
do
	if [ ! -f $i ]; then
		continue
	fi
        scores=` grep ^pose $i | cut -f15-17 -d' ' `
        weights=` grep ^weights $i | cut -f15-17 -d' ' `
	numResidues=` grep CtermProteinFull $i | cut -f1 -d' ' | cut -f2 -d':' | cut -f2 -d'_'`
	#echo $scores $weights $numResidues

	## claculate per-residue unweighted score
        avg=` echo $scores $weights $numResidues | awk '{a=$1/$4/$7; b=$2/$5/$7; c=$3/$6/$7; s=a+b+c} END {printf "%.2f %.2f %.2f %.2f", s, a, b, c}' `
        echo $i $avg >> $scorefile
        #echo $i $sum $scores >> $scorefile
done

normalize=$DistanceFoldingHome/Helpers/Normalize.py

if [ -s $scorefile ]; then
	processID=$$	
	sort -k2,2 -g $scorefile > $scorefile.$processID
	python $normalize $scorefile.$processID $scorefile
	if [ $? -ne 0 ]; then
		echo "ERROR: failed to run python $normalize $scorefile.$processID $scorefile"
		exit 1
	fi
	rm -f $scorefile.$processID
	echo "All model scores are saved to $scorefile"
fi