import os
import sys
import numpy as np
import datetime

import cPickle
import msgpack

def Usage():
	print 'python CCMpredUtils.py target.mpk [outDir]'
	print '	this script converts the original CCMpred precision matrix from float32 to float16 and produces a smaller summary matrix with dimension L*L*43'
	print ' the input is an MsgPack file generated by CCMpred; outDir is the folder for result saving (current work directory by default)'
	print ' the result will be saved in cPickle format as a dict() with 5 keys: seqLen, Fnorm, FnormZ, rawCCM and sumCCM'
	print '  Fnorm is a L*L matrix formed by the Fro norm of all 20*20 blocks. FnormZ is the normalized Fnorm without considering the diagonal line'
	print '	 rawCCM is the original precision matrix and sumCCM is the reducted precision matrix. '
	print '  rawCCM and sumCCM are dict() with (i, j) as key where i and j are redidue indices starting from 0'
	print '  Each value in rawCCM and sumCCM is a vector of 441 and 43 entries, respectively'

## the value of precision matrix is usually small
## here we use a scale constant before converting float32 to float16, which may maintain precision
scaleConstant = 10

## normalize a matrix without considering the diagonal
## the diagonal of m is assumed to be 0
def Normalize(m):
	## caluclate average, adjust it by n/(n-1) to remove the impact of diagonal
	n = m.shape[0]
	avg = np.mean(m) * n/(n-1)
	tmp = np.square( m - avg )
	np.fill_diagonal(tmp, 0)
	std = np.sqrt( np.mean(tmp) * n/(n-1) )
	if std == 0:
		res = np.zeros_like(m, dtype=np.float16)
		return res

	res = (m-avg)/std
	np.fill_diagonal(res, 0)

	return res.astype(np.float16)

# apply APC to a matrix
def APC(m):
	avg = np.mean(m)
	rowAvg = np.mean(m, axis=1)
	colAvg = np.mean(m, axis=0)
	res = m - ( np.outer(rowAvg, colAvg) / avg)
	np.fill_diagonal(res, 0)

	return res


## W is a list or vector of 441 entries	
## here we assume that the index 20 corresponds to a gap
def SummarizeOneBlock(W, m):
	a = np.array(W).reshape( (21, 21) ) - m
	a =  a*scaleConstant

	## calculate different norms of a
	Fnorm = np.linalg.norm(a[:20,:20], ord='fro')
	#Nnorm = np.linalg.norm(a[:20,:20], ord='nuc')

	lastE = np.absolute(a[20, 20])
	a = np.square(a)
	s1 = np.sum(a[:, :20], axis=1)
	s2 = np.sum(a[:20, :], axis=0)

	s1 = np.sqrt(s1)
	s2 = np.sqrt(s2)

	s1 = s1.astype(np.float16)
	s2 = s2.astype(np.float16)
	lastE = np.float16(lastE)

	return [s1, s2, lastE], Fnorm


## summarize the whole precision matrix
## W represents the original content in the msgpack file. It has 4keys: x_single, ncol, x_pair and format.
## W['ncol'] is sequence length and W['x_pair'] has the info for precision matrix
## W['x_pair'] is a dict() with keys like '0/1', '3/7'...
## Each value in W['x_pair'] has three keys: i, j and x where x is a list of 441 entries for the precision values at positions i and j
## Since the precision matrix is symmetric, only i<j is stored in W['x_pair']
def Reduce(W):

	seqLen = W['ncol']
	xPair = W['x_pair']

	rawMatrix = dict()
	
	## calculate average of the whole precision matrix
	avgpool = []
	for k, v in xPair.iteritems():
		## v is a dict() with three keys: i, j, and x where i and j are for indices and x is a list of real values
		i = v['i']
		j = v['j']
		if i>=j:
			continue
		rawMatrix[(i, j)] = (np.array(v['x']) * scaleConstant).astype(np.float16)

		avg = np.mean(v['x'])
		avgpool.append(avg)

	finalavg = np.mean(avgpool)
	
	normMatrix = np.zeros((seqLen, seqLen), dtype=np.float16)
	
	reducedMatrix = dict()
	## calculate the final result
	for k, v in xPair.iteritems():
		## v is a dict() with three keys: i, j, and x where i and j are for indices and x is a list of real values
		i = v['i']
		j = v['j']
		if i>=j:
			continue

		res, norm = SummarizeOneBlock(v['x'], finalavg)
		reducedMatrix[(i, j)] = res
		normMatrix[i, j] = norm
		normMatrix[j, i] = norm

	apcNormMatrix = APC(normMatrix)
	

	result = dict()
	result['seqLen'] = seqLen
	result['rawCCM'] = rawMatrix
	result['sumCCM'] = reducedMatrix
	result['Fnorm'] = apcNormMatrix.astype(np.float16)
	## calculate Zscore of apcNorm
	result['FnormZ'] = Normalize(apcNormMatrix)
		
	return result

	
def ParseCCMMsgPackFile(inputFile):
	with open(inputFile, 'rb') as fh:
		W = msgpack.load(fh)
		return Reduce(W)

def LoadCCMMatrix(file):
    	allECs = []
	with open(file, 'r') as fh:
    		for line in list(fh):
        		ECs = [ np.float32(x) for x in line.split() ]
        		allECs.append(ECs)

    	CCM = np.array(allECs) 
	if np.isnan(CCM).any():
        	print 'ERROR: there is at least one NaN in ', file
        	exit(1)

    	return CCM

## input is the upper triangle of a matrix in Python dict() with (i, j) as the key.
## i<j and they are the residue indices
## this function converts input to a regular matrix with dimension L*L*n_out where n_out=441 or 43

## bounds = [top, left, bottom, right]
def ExpandMatrix(input, seqLen, bounds=None):
	if not bool(input):
		return None

	v = input.values()[0]
	if len(v) == 441:
		n_out = 441
	else:
		n_out = len(v[0]) + len(v[1]) + 1

	out = np.zeros((seqLen, seqLen, n_out), dtype=np.float16)
	for k, v in input.iteritems():
		i, j = k

		assert (i<j)

		ijInBounds = True
		jiInBounds = True

		if bounds is not None:
			top, left, bottom, right = bounds
			ijInBounds = (i>=top and i<=bottom) and (j>=left and j<=right) 
			jiInBounds = (j>=top and j<=bottom) and (i>=left and i<=right)
		
		if len(v) == 441:
			if ijInBounds:
				out[i, j] = v
			if jiInBounds:
				tmp = v.reshape(21, 21)
				tmp2 = np.transpose(tmp)
				out[j, i] = tmp2.flatten()
		elif len(v)==3:
			if ijInBounds:
				out[i, j] = np.concatenate( (v[0], v[1], [ v[2] ]) )
			if jiInBounds:
				out[j, i] = np.concatenate( (v[1], v[0], [ v[2] ]) )

		else:
			print 'ERROR: unsupported length for the value in a matrix dict()'
			exit(1)

	return out

## verify the result. sumCCM is a dict() represents the reduced CCM matrix with dimension L*L*43
## uCCM is the unnormalized matrix generated by CCMpred with dimension L*L. Each entry in uCCM represents interaction strength between two residues
def CheckConsistency(Norm, NormZ):

	print Norm[10]
	print NormZ[10]


def main(argv):
	if len(argv) < 1:
		Usage()
		exit(1)

	inputFile = argv[0]
	outDir = './'

	if len(argv) >=2:
		outDir = argv[1]

	res = ParseCCMMsgPackFile(inputFile)

	#CheckConsistency(res['Fnorm'], res['FnormZ'])

	## save it
	if not os.path.isdir(outDir):
		os.mkdir(outDir)

	savefile = os.path.basename(inputFile).split('.')[0] + '.extraCCM.pkl'
	savefile = os.path.join(outDir, savefile)
	with open(savefile, 'w') as fh:
		cPickle.dump(res, fh, protocol=cPickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
        main(sys.argv[1:])
