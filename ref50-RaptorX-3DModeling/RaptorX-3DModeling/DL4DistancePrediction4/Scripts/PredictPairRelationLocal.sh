#!/bin/sh

if [[ -z "${ModelingHome}" ]]; then
	echo "ERROR: please set environmental variable ModelingHome to the install folder of RaptorX-3DModeling"
	exit 1
fi

if [[ -z "${DL4DistancePredHome}" ]]; then
        echo "ERROR: please set environmental variable DL4DistancePredHome to the install folder of DL4DistancePrediction4"
        exit 1
fi


DeepModelFile=$DL4DistancePredHome/params/ModelFile4PairwisePred.txt
#ModelName=EC47C31C16CL99LargerS35V2020CbCbTwoRModels
#ModelName=EC47C37C19CL99S35V2020MidModels
DefaultModel4FM=EC47C37C19CL99S35V2020MidModels
DefaultModel4HHP=HHEC47C37C19CL99S35PDB70Models
DefaultModel4NDT=NDTEC47C37C19CL99S35BC40Models
DefaultModel4HA=HAHHEC47C37C19CL99S35NewPDB70Models

GPU=-1

alignmentType=0
ModelName=""

function Usage {
        echo $0 "[  -T alignmentType | -f DeepModelFile | -m ModelName | -d ResultDir | -g gpu ] proteinName inputFolder"
        echo "	This script predicts distance and orientation from MSAs and optionally templates using a GPU on a local machine"
	echo "	proteinName: the potein name"
        echo "	inputFolder: a folder generated by BuildFeatures.sh, e.g., 1pazA_OUT/, that contains all needed feature folders and files, e.g., proteinName_contact, proteinName_thread and proteinName.seq"
	echo " "
	echo "	-T: indicate which query-template alignments to be used: 0 for none (default), 1 for alignments generated by HHpred and 2 for alignments generated by RaptorX threading"
        echo "		When -T is 0, predict distance/orientation from only sequence info such as MSAs"
        echo "		When not 0, predict distance/orientation matrices from both MSAs and query-template alignments"
	echo "		The query-template alignment files shall be ready at inputFolder/HHP and inputFolder/DeepThreadr, respectively"
	echo "	-f: a file containing a set of deep model names, default $DeepModelFile"
        echo "	-m: a model name defined in DeepModelFile representing a set of deep learning models, default empty"
        echo "		When it is empty, the deep model will be chosen by alignmentType as follows"
        echo "		0: $DefaultModel4FM, 1: $DefaultModel4HHP, 2:$DefaultModel4NDT, 3:$DefaultModel4HA"
	echo " "
        echo "	-d: the folder for result saving, default current work directory"
        echo "	-g: -1 (default), 0-3. If -1, automatically select a GPU"
	echo "		Users shall make sure that at least one GPU has enough memory for the prediction job. Otherwise it may crash itself or other jobs"
}

while getopts ":f:m:d:r:g:T:" opt; do
        case ${opt} in
                f )
                  DeepModelFile=$OPTARG
                  ;;
                m )
                  ModelName=$OPTARG
                  ;;
                d )
                  ResultDir=$OPTARG
                  ;;
                r )
                  RemoteAccount=$OPTARG
                  ;;
                g )
                  GPU=$OPTARG
                  ;;
		T )
		  alignmentType=$OPTARG
		  ;;
                \? )
                  echo "Invalid Option: -$OPTARG" 1>&2
                  exit 1
                  ;;
                : )
                  echo "Invalid Option: -$OPTARG requires an argument" 1>&2
                  exit 1
                  ;;
        esac
done
shift $((OPTIND -1))

if [ $# -ne 2 ]; then
        Usage
        exit 1
fi

proteinName=$1
rootDir=$2
if [ ! -d $rootDir ]; then
	echo "ERROR: the folder for features does not exist: $rootDir"
	exit 1
fi

if [ ! -f $DeepModelFile ]; then
        echo "ERROR: invalid file for deep model path information: $DeepModelFile"
        exit 1
fi
. $DeepModelFile

if [ -z "$ModelName" ]; then
	ModelName=$DefaultModel4FM
	if [ $alignmentType -eq 1 ]; then
		ModelName=$DefaultModel4HHP

	elif [ $alignmentType -eq 2 ]; then
		ModelName=$DefaultModel4NDT

	elif [ $alignmentType -eq 3 ]; then
		ModelName=$DefaultModel4HA
	fi
fi

ModelFiles=`eval echo '$'${ModelName}`
if [ $ModelFiles == "" ]; then
	echo "ERROR: ModelFiles is empty !"
	exit 1
fi
#echo ModelFiles=$ModelFiles

seqFile=$rootDir/$proteinName.seq

contactDir=$rootDir/${proteinName}_contact/
if [ ! -d $contactDir ]; then
	echo "ERROR: invalid folder for contact/distance features: $contactDir"
	exit 1
fi

for method in uce3 uce5 ure3 ure5 user
do
        featureFolder=$contactDir/feat_${proteinName}_${method}_meta
        if [ ! -d $featureFolder ]; then
                featureFolder=$contactDir/feat_${proteinName}_${method}
        fi
        if [ ! -d $featureFolder ]; then
                continue
        fi
        if [ -z $inputFolders ]; then
                inputFolders=$featureFolder
        else
                inputFolders=$inputFolders';'$featureFolder
        fi
done

if [ -z $inputFolders ]; then
        echo "ERROR: there are no feature folders for $proteinName in $rootDir"
        exit 1
fi

#echo 'inputFolders= $inputFolders'

program=$DL4DistancePredHome/RunPairwisePredictor.py
if [ ! -f $program ]; then
	echo "ERROR: the main program does not exist: $program "
	exit 1
fi

if [ ! -d $ResultDir ]; then
	mkdir -p $ResultDir
fi

arguments="-m $ModelFiles -p $proteinName -i $inputFolders -d $ResultDir "

if [ $alignmentType -ne 0 ]; then
	if [ -z "$PDB70TPL" ]; then
		echo "ERROR: please set the environmental variable PDB70TPL to the TPLPKL folder of PDB70"
		exit 1
	fi
	if [ ! -d $PDB70TPL ]; then
		echo "ERROR: invalid folder for the tpl.pkl files of PDB70: $PDB70TPL"
		exit 1
	fi
fi

if [ $alignmentType -eq 1 -o $alignmentType -eq 3 ]; then
	if [ ! -d $rootDir/HHP ]; then
		echo "ERROR: you specify to predict distance/orientation matrices from seq-template alignments generated by HHpred, but they cannot be found in $rootDir/HHP/"
		exit 1
	fi
	arguments=$arguments" -a $rootDir/HHP -t $PDB70TPL "

elif [ $alignmentType -eq 2 ]; then
	if [ ! -d $rootDir/DeepThreader ]; then
		echo "ERROR: you specify to predict distance/orientation matrices from seq-template alignments generated by RaptorX threading, but they cannot be found in $rootDir/DeepThreader/"
		exit 1
	fi
	arguments=$arguments" -a $rootDir/DeepThreader -t $PDB70TPL "
fi

if [[ -z "${CUDA_ROOT}" ]]; then
        echo "ERROR: please set environmental variable CUDA_ROOT to the install folder of cuda and CUDNN, e.g., /usr/local/cuda/"
        exit 1
fi

if [ $GPU == "-1" ]; then
        neededRAM=`$DL4DistancePredHome/Scripts/EstimateGPURAM4DistPred.sh $seqFile`
        GPU=`$ModelingHome/Utils/FindOneGPUByMemory.sh $neededRAM 30`
fi

if [ $GPU == "-1" ]; then
        echo "ERROR: cannot find an appropriate GPU to predict distance/orientation for $seqFile !"
	exit 1
else
        GPU=cuda$GPU
fi

THEANO_FLAGS=blas.ldflags=,device=$GPU,floatX=float32,dnn.include_path=${CUDA_ROOT}/include,dnn.library_path=${CUDA_ROOT}/lib64 python $program $arguments
if [ $? -ne 0 ]; then
        echo "ERROR: failed to run python $program $arguments"
        exit 1
fi