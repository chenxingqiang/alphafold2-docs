/* PSICOV - Protein Sparse Inverse COVariance analysis program */

/* by David T. Jones August 2011 - Copyright (C) 2011 University College London */

/* This code is licensed under the terms of GNU General Public License v2 or later */

/* Version 2.4 - Last Edit 28/12/16 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define FALSE 0
#define TRUE 1

#define SQR(x) ((x)*(x))
#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)<(y)?(x):(y))

#define MAXSEQLEN 5000
#define MINEFSEQS (seqlen)

/* Dump a rude message to standard error and exit */
void fail(char *fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt) ;
    fprintf(stderr, "*** ");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    
    exit(-1);
}

/* Convert AA letter to numeric code (0-21) */
int
                aanum(int ch)
{
    const static int aacvs[] =
    {
	999, 0, 3, 4, 3, 6, 13, 7, 8, 9, 21, 11, 10, 12, 2,
	21, 14, 5, 1, 15, 16, 21, 19, 17, 21, 18, 6
    };

    return (isalpha(ch) ? aacvs[ch & 31] : 20);
}

/* Allocate matrix */
void           *allocmat(int rows, int columns, int size)
{
    int             i;
    void          **p, *rp;

    rp = malloc(rows * sizeof(void *) + sizeof(int));

    if (rp == NULL)
	fail("allocmat: malloc [] failed!");

    *((int *)rp) = rows;

    p = rp + sizeof(int);

    for (i = 0; i < rows; i++)
	if ((p[i] = calloc(columns, size)) == NULL)
	    fail("allocmat: malloc [][] failed!");

    return p;
}

/* Allocate vector */
void           *allocvec(int columns, int size)
{
    void          *p;

    p = calloc(columns, size);

    if (p == NULL)
	fail("allocvec: calloc failed!");

    return p;
}

/*  This subroutine computes the L1 regularized covariance matrix estimate
    using the algorithm described in the paper:
    J. Friedman, T. Hastie, R. Tibshirani:
    Sparse inverse covariance estimation with the graphical lasso
    Biostatistics, 9(3):432-441, July 2008.
    This code is adapted from the Fortran code described in the following report:
    M. A. Sustik & B. Calderhead:
    GLASSOFAST: An efficient GLASSO implementation
    Technical Report TR-12-29, University of Texas at Austin

    NOTE: that when multiple threads are used, we gain a huge time saving by
    avoiding full thread synchronisation when updating elements of the W (covariance)
    matrix. In multithreaded mode, the order of updates to the W matrix at each iteration
    will depend on the order in which threads complete. In practice, this hardly matters,
    because the algorithm is iterative, and in testing still converges to within 6 d.p.
    of the non-threaded code. If a very small degree of non-deterministic behaviour really
    worries you, then set the maximum number of threads to 1 (or compile without OpenMP).
*/

#define EPS (1.1e-15)
#define BIG (1e9)

int glassofast(const int n, double **S, double **L, const double thr, const int maxit, int approxflg, int warm, double **X, double **W)
{
    int i, j, ii, iter, jj;
    double a, b, c, delta, dlx, dw, shr, sum, thrlasso, tmp, wd[MAXSEQLEN*21], wxj[MAXSEQLEN*21];

    for (shr=ii=0; ii<n; ii++)
	for (jj=0; jj<n; jj++)
	    shr += fabs(S[ii][jj]);
    
    for (i=0; i<n; i++)
	shr -= fabs(S[i][i]);

    if (shr == 0.0)
    {
        /* S is diagonal. */
	for (ii=0; ii<n; ii++)
	    for (jj=0; jj<n; jj++)
		W[ii][jj] = X[ii][jj] = 0.0;
	
	for (i=0; i<n; i++)
	    W[i][i] = W[i][i] + L[i][i];
	
	for (ii=0; ii<n; ii++)
	    for (jj=0; jj<n; jj++)
		X[ii][jj] = 0.0;
	
	for (i=0; i<n; i++)
	    X[i][i] = 1.0 / MAX(W[i][i], EPS);

	return 0;
    }
    
    shr *= thr/(n-1);
    thrlasso = shr/n;
    if (thrlasso < 2*EPS)
	thrlasso = 2*EPS;
    
    if (!warm)
    {
	for (ii=0; ii<n; ii++)
	    for (jj=0; jj<n; jj++)
	    {
		W[ii][jj] = S[ii][jj];
		X[ii][jj] = 0.0;
	    }
    }
    else
    {
	for (i=0; i<n; i++)
	{
	    for (ii=0; ii<n; ii++)
		X[i][ii] = -X[i][ii]/X[i][i];
	    X[i][i] = 0.0;
	}
    }
    
    for (i=0; i<n; i++)
    {
	wd[i] = S[i][i] + L[i][i];
	W[i][i] = wd[i];
    }
    
    for (iter = 1; iter<=maxit; iter++)
    {
	dw = 0.0;

#pragma omp parallel for default(shared) private(i,j,ii,wxj,a,b,c,dlx,delta,sum)
	for (j=0; j<n; j++)
	{
	    for (ii=0; ii<n; ii++)
		wxj[ii] = 0.0;

	    for (i=0; i<n; i++)
		if (X[j][i] != 0.0)
		    for (ii=0; ii<n; ii++)
			wxj[ii] += W[i][ii] * X[j][i];

	    for (;;)
	    {
		dlx = 0.0;
		
		for (i=0; i<n; i++)
		{
		    if (i != j && L[j][i] < BIG)
		    {
			a = S[j][i] - wxj[i] + wd[i] * X[j][i];
			b = fabs(a) - L[j][i];
			if (b <= 0.0)
			    c = 0.0;
			else if (a >= 0.0)
			    c = b / wd[i];
			else
			    c = -b / wd[i];

			delta = c - X[j][i];
			if (delta != 0.0 && (!approxflg || fabs(delta) > 1e-6))
			{
			    X[j][i] = c;
			
			    for (ii=0; ii<n; ii++)
				wxj[ii] += W[i][ii] * delta;
			    
			    if (fabs(delta) > dlx)
				dlx = fabs(delta);
			}
		    }
		}
		
		if (dlx < thrlasso)
		    break;
	    }
	    
	    wxj[j] = wd[j];
	    
	    for (sum=ii=0; ii<n; ii++)
		sum += fabs(wxj[ii] - W[j][ii]);

#pragma omp critical
	    if (sum > dw)
		dw = sum;

	    for (ii=0; ii<n; ii++)
		W[j][ii] = wxj[ii];
	    for (ii=0; ii<n; ii++)
		W[ii][j] = wxj[ii];
	}
	
	if (dw <= shr)
	    break;
    }

    for (i=0; i<n; i++)
    {
	for (sum=ii=0; ii<n; ii++)
	    sum += X[i][ii] * W[i][ii];
	
	tmp = 1.0 / (wd[i] - sum);
	
	for (ii=0; ii<n; ii++)
	    X[i][ii] = -tmp * X[i][ii];
	X[i][i] = tmp;
    }
    
    for (i=0; i<n-1; i++)
    {
	for (ii=i+1; ii<n; ii++)
	{
	    X[i][ii] = 0.5 * (X[i][ii] + X[ii][i]);
	    X[ii][i] = X[i][ii];
	}
    }
    
    return iter;
}


/* Test Cholesky decomposition on matrix */
int test_cholesky(double **a, const int n) 
{
    int i, j, k;
    double sum, diag;

    for (i=0; i<n; i++)
    {
	sum = a[i][i];

	for (k=i-1; k >= 0; k--)
	    sum -= a[i][k]*a[i][k];
	    
	if (sum <= 0.0)
	    return TRUE;
	
	diag = sqrt(sum);

        #pragma omp parallel for
	for (j=i+1; j<n; j++)
	{
	    double sum = a[i][j];
	    
	    for (k=0; k<i; k++)
		sum -= a[i][k]*a[j][k];

	    a[j][i] = sum / diag;
	}
    }
    
    return FALSE;
}


struct sc_entry
{
    double sc;
    int i, j;
} *sclist;

/* Sort descending */
int cmpfn(const void *a, const void *b)
{
    if (((struct sc_entry *)a)->sc == ((struct sc_entry *)b)->sc)
	return 0;

    if (((struct sc_entry *)a)->sc < ((struct sc_entry *)b)->sc)
	return 1;

    return -1;
}
    

int             main(int argc, char **argv)
{
    int             a, b, i, j, k, seqlen, nids, s, nseqs, ncon, opt, ndim, filtflg=0, approxflg=0, initflg=0, apcflg=1, maxit=10000, npair, nnzero, niter, jerr, shrinkflg=1, rawscflg = 1, pseudoc = 1, minseqsep = 5, overrideflg=0, ntries;
    unsigned int *wtcount, ccount[MAXSEQLEN];
    double thresh=1e-4, del, **pcmat, *pcsum, pcmean, pc, trialrho, rhodefault = -1.0;
    double sum, score, **pa, wtsum, lambda, low_lambda, high_lambda, smean, fnzero, lastfnzero, rfact, r2, targfnzero = 0.0, scsum, scsumsq, mean, sd, zscore, ppv;    
    double *weight, idthresh = -1.0, maxgapf = 0.9, besttd = 1.0, bestrho = 0.001;
    char            buf[4096], seq[MAXSEQLEN], *blockfn = NULL, **aln;
    FILE *ifp;

    while ((opt = getopt(argc, argv, "aflnopr:b:i:t:c:g:d:j:z:")) >= 0)
	switch (opt)
	{
	case 'a':
	    approxflg = 1;
	    break;
	case 'n':
	    shrinkflg = 0;
	    break;
	case 'o':
	    overrideflg = 1;
	    break;
	case 'p':
	    rawscflg = 0;
	    break;
	case 'f':
	    filtflg = 1;
	    break;
	case 'l':
	    apcflg = 0;
	    break;
	case 'r':
	    rhodefault = atof(optarg);
	    break;
	case 'd':
	    targfnzero = atof(optarg);
	    if (targfnzero < 5e-5 || targfnzero >= 1.0)
		fail("Target density value must be in range 5e-5 >= d < 1!");
	    break;
	case 't':
	    thresh = atof(optarg);
	    break;
	case 'i':
	    idthresh = 1.0 - atof(optarg)/100.0;
	    break;
	case 'c':
	    pseudoc = atoi(optarg);
	    break;
	case 'j':
	    minseqsep = atoi(optarg);
	    break;
	case 'b':
	    blockfn = strdup(optarg);
	    break;
	case 'g':
	    maxgapf = atof(optarg);
	    break;
	case 'z':
#ifdef _OPENMP
	    omp_set_num_threads(atoi(optarg));
#endif
	    break;
	case '?':
	    exit(-1);
	}

    if (optind >= argc)
	fail("PSICOV V2.4 Usage: psicov [options] alnfile\n\nOptions:\n-a\t: use approximate Lasso algorithm\n-n\t: don't pre-shrink the sample covariance matrix\n-f\t: filter low-scoring contacts\n-p\t: output PPV estimates rather than raw scores\n-l\t: don't apply APC to Lasso output\n-r nnn\t: set initial rho parameter\n-d nnn\t: set target precision matrix sparsity (default 0 = not specified)\n-t nnn\t: set Lasso convergence threshold (default 1e-4)\n-i nnn\t: select BLOSUM-like weighting with given identity threshold (default selects threshold automatically)\n-c nnn\t: set pseudocount value (default 1)\n-j nnn\t: set minimum sequence separation (default 5)\n-g nnn\t: maximum fraction of gaps (default 0.9)\n-z nnn\t: set maximum no. of threads\n-b file\t: read rho parameter file\n");

    ifp = fopen(argv[optind], "r");
    if (!ifp)
	fail("Unable to open alignment file!");

    for (nseqs=0;; nseqs++)
	if (!fgets(seq, MAXSEQLEN, ifp))
	    break;

    aln = allocvec(nseqs, sizeof(char *));
    
    weight = allocvec(nseqs, sizeof(double));

    wtcount = allocvec(nseqs, sizeof(unsigned int));
    
    rewind(ifp);
    
    if (!fgets(seq, MAXSEQLEN, ifp))
	fail("Bad alignment file!");
    
    seqlen = strlen(seq)-1;

    if (!(aln[0] = malloc(seqlen)))
	fail("Out of memory!");

    for (j=0; j<seqlen; j++)
	aln[0][j] = aanum(seq[j]);
    
    for (i=1; i<nseqs; i++)
    {
	if (!fgets(seq, MAXSEQLEN, ifp))
	    break;
	
	if (seqlen != strlen(seq)-1)
	    fail("Length mismatch in alignment file!");
	
	if (!(aln[i] = malloc(seqlen)))
	    fail("Out of memory!");
	
	for (j=0; j<seqlen; j++)
	    aln[i][j] = aanum(seq[j]);
    }


    /* Calculate sequence weights (use openMP/pthreads if available) */

    if (idthresh < 0.0)
    {
	double meanfracid = 0.0;
	
#pragma omp parallel for default(shared) private(j,k) reduction(+:meanfracid)
	for (i=0; i<nseqs; i++)
	    for (j=i+1; j<nseqs; j++)
	    {
		int nids;
		double fracid;

		for (nids=k=0; k<seqlen; k++)
		    nids += (aln[i][k] == aln[j][k]);
		
		fracid = (double)nids / seqlen;
		
		meanfracid += fracid;
	    }
	
	meanfracid /= 0.5 * nseqs * (nseqs - 1.0);

	idthresh = MIN(0.6, 0.38 * 0.32 / meanfracid);

//	printf("idthresh = %f  meanfracid = %f\n", idthresh, meanfracid);
    }


#pragma omp parallel for default(shared) private(j,k)
    for (i=0; i<nseqs; i++)
	for (j=i+1; j<nseqs; j++)
	{
	    int nthresh = seqlen * idthresh;

	    for (k=0; nthresh > 0 && k<seqlen; k++)
		nthresh -= (aln[i][k] != aln[j][k]);
	    
	    if (nthresh > 0)
	    {
#pragma omp critical 
		wtcount[i]++;
		wtcount[j]++;
	    }
	}

    for (wtsum=i=0; i<nseqs; i++)
	wtsum += (weight[i] = 1.0 / (1 + wtcount[i]));

//    printf("wtsum = %f\n", wtsum);
    
    if (wtsum < MINEFSEQS && !overrideflg)
	fail("Sorry - not enough sequences or sequence diversity to proceed!\nNeff (%f) < MINEFSEQS (%d)\nIf you want to force a calculation at your own risk, adjust MINEFSEQS or use -o to override.\n", wtsum, MINEFSEQS);
    
    pa = allocmat(seqlen, 21, sizeof(double));

    /* Calculate singlet frequencies with pseudocount */
    for (i=0; i<seqlen; i++)
    {
	for (a=0; a<21; a++)
	    pa[i][a] = pseudoc;
	
	for (k=0; k<nseqs; k++)
	{
	    a = aln[k][i];
	    if (a < 21)
		pa[i][a] += weight[k];
	}
	
	for (a=0; a<21; a++)
	    pa[i][a] /= pseudoc * 21.0 + wtsum;
    }

    double **cmat, **rho, **ww, **wwi, **tempmat;

    ndim = seqlen * 21;

    cmat = allocmat(ndim, ndim, sizeof(double));
    tempmat = allocmat(ndim, ndim, sizeof(double));

    /* Form the covariance matrix */
#pragma omp parallel for default(shared) private(j,k,a,b)
    for (i=0; i<seqlen; i++)
	for (j=i; j<seqlen; j++)
	{
	    double pab[21][21];

	    for (a=0; a<21; a++)
		for (b=0; b<21; b++)
		    if (i == j)
			pab[a][b] = (a == b) ? pa[i][a] : 0.0;
		    else
			pab[a][b] = pseudoc / 21.0;
	    
	    if (i != j)
	    {
		for (k=0; k<nseqs; k++)
		{
		    a = aln[k][i];
		    b = aln[k][j];
		    if (a < 21 && b < 21)
			pab[a][b] += weight[k];
		}
	    
		for (a=0; a<21; a++)
		    for (b=0; b<21; b++)
			pab[a][b] /= pseudoc * 21.0 + wtsum;
	    }
	    
	    for (a=0; a<21; a++)
		for (b=0; b<21; b++)
		    if (i != j || a == b)
			cmat[i*21+a][j*21+b] = cmat[j*21+b][i*21+a] = pab[a][b] - pa[i][a] * pa[j][b];
	}

    /* Shrink sample covariance matrix towards shrinkage target F = Diag(1,1,1,...,1) * smean */

    if (shrinkflg)
    {
	for (smean=i=0; i<ndim; i++)
	    smean += cmat[i][i];
	
	smean /= (double)ndim;

	high_lambda = 1.0;
	low_lambda = 0.0;
	lambda = 0.5;
	
	for (;;)
	{
#pragma omp parallel for default(shared) private(j,a,b)
	    for (i=0; i<seqlen; i++)
		for (j=0; j<seqlen; j++)
		    for (a=0; a<21; a++)
			for (b=0; b<21; b++)
			    if (i != j)
				tempmat[i*21+a][j*21+b] = cmat[i*21+a][j*21+b] * (1.0 - lambda);
			    else if (a == b)
				tempmat[i*21+a][j*21+b] = smean * lambda + (1.0 - lambda) * cmat[i*21+a][j*21+b];

	    /* Test if positive definite using Cholesky decomposition */
	    if (!test_cholesky(tempmat, ndim))
	    {
		if (high_lambda - low_lambda < 0.01)
		    break;
		
		high_lambda = lambda;
		lambda = 0.5 * (lambda + low_lambda);
	    }
	    else
	    {
		low_lambda = lambda;
		lambda = 0.5 * (lambda + high_lambda);
	    }
	}

	for (i=0; i<seqlen; i++)
	    for (j=0; j<seqlen; j++)
		for (a=0; a<21; a++)
		    for (b=0; b<21; b++)
			if (i != j)
			    cmat[i*21+a][j*21+b] *= (1.0 - lambda);
			else if (a == b)
			    cmat[i*21+a][j*21+b] = smean * lambda + (1.0 - lambda) * cmat[i*21+a][j*21+b];
    }

    rho = allocmat(ndim, ndim, sizeof(double));
    ww = allocmat(ndim, ndim, sizeof(double));
    wwi = allocmat(ndim, ndim, sizeof(double));

    lastfnzero=0.0;

    /* Guess at a reasonable starting rho value if undefined */
    if (rhodefault < 0.0)
	trialrho = 0.001;
    else
	trialrho = rhodefault;

    rfact = 0.0;

    ntries = 1;

    for (;;)
    {
	double targdiff;

	if (trialrho <= 0.0 || trialrho >= 1.0 || ntries++ > 10)
	{
	    /* Give up search - recalculate with best rho found so far and exit */
	    trialrho = bestrho;
	    targfnzero = 0.0;
	}
	    
	for (i=0; i<ndim; i++)
	    for (j=0; j<ndim; j++)
		rho[i][j] = trialrho;
	
	for (i=0; i<seqlen; i++)
	    for (j=0; j<seqlen; j++)
		for (a=0; a<21; a++)
		    for (b=0; b<21; b++)
			if ((a != b && i == j) || pa[i][20] > maxgapf || pa[j][20] > maxgapf)
			    rho[i*21+a][j*21+b] = BIG;
	
	/* Mask out regions if block-out list provided */
	if (blockfn != NULL)
	{
	    ifp = fopen(blockfn, "r");
	    
	    for (;;)
	    {
		if (fscanf(ifp, "%d %d %lf", &i, &j, &score) != 3)
		    break;
		
		for (a=0; a<21; a++)
		    for (b=0; b<21; b++)
		    {
			rho[(i-1)*21+a][(j-1)*21+b] = score;
			rho[(j-1)*21+b][(i-1)*21+a] = score;
		    }
	    }
	    
	    fclose(ifp);
	}

	glassofast(ndim, cmat, rho, thresh, maxit, approxflg, initflg, wwi, ww);

	/* Quit with optimum rho value so far */
	if (targfnzero <= 0.0)
	    break;
	
	for (npair=nnzero=i=0; i<ndim; i++)
	    for (j=i+1; j<ndim; j++,npair++)
		if (wwi[i][j] != 0.0)
		    nnzero++;

	fnzero = (double) nnzero / npair;

	/* Stop iterating if we have achieved the target sparsity level */

	targdiff = fabs(fnzero - targfnzero)/targfnzero;

	//printf("rho=%f fnzero = %f targdiff = %f\n", trialrho, fnzero, targdiff);

	if (targdiff < 0.02)
	    break;

	if (targdiff < besttd)
	{
	    besttd = targdiff;
	    bestrho = trialrho;
	}
	
	if (fnzero == 0.0)
	{
	    /* As we have guessed far too high, halve rho and try again */
	    trialrho *= 0.5;
	    continue;
	}
	
	if (lastfnzero > 0.0 && fnzero != lastfnzero)
	{
//	    printf("fnzero=%f lastfnzero=%f trialrho=%f oldtrialrho=%f\n", fnzero, lastfnzero, trialrho, trialrho/rfact);
	    
	    rfact = pow(rfact, log(targfnzero / fnzero) / log(fnzero / lastfnzero));

//	    printf("New rfact = %f\n", rfact);
	}

	lastfnzero = fnzero;

	/* Make a small trial step in the appropriate direction */

	if (rfact == 0.0)
	    rfact = (fnzero < targfnzero) ? 0.9 : 1.1;
	
	trialrho *= rfact;
    }

    /* Calculate background corrected scores using average product correction */

    pcmat = allocmat(seqlen, seqlen, sizeof(double));
    pcsum = allocvec(seqlen, sizeof(double));
    
    pcmean = 0.0;
    
    for (i=0; i<seqlen; i++)
	for (j=i+1; j<seqlen; j++)
	{	
	    for (pc=a=0; a<20; a++)
		for (b=0; b<20; b++)
		    pc += fabs(wwi[i*21+a][j*21+b]);

	    pcmat[i][j] = pcmat[j][i] = pc;
	    pcsum[i] += pc;
	    pcsum[j] += pc;

	    pcmean += pc;
	}

    pcmean /= seqlen * (seqlen - 1) * 0.5;

    /* Build final list of predicted contacts */

    sclist = allocvec(seqlen * (seqlen - 1) / 2, sizeof(struct sc_entry));

    for (scsum=scsumsq=ncon=i=0; i<seqlen; i++)
	for (j=i+minseqsep; j<seqlen; j++)
	    if (pcmat[i][j] > 0.0)
	    {
		/* Calculate APC score */
		if (apcflg)
		    sclist[ncon].sc = pcmat[i][j] - pcsum[i] * pcsum[j] / SQR(seqlen - 1.0) / pcmean;
		else
		    sclist[ncon].sc = pcmat[i][j];
		scsum += sclist[ncon].sc;
		scsumsq += SQR(sclist[ncon].sc);
		sclist[ncon].i = i;
		sclist[ncon++].j = j;
	    }

    qsort(sclist, ncon, sizeof(struct sc_entry), cmpfn);

    mean = scsum / ncon;
    sd = 1.25 * sqrt(scsumsq / ncon - SQR(mean)); /* Corrected for extreme-value bias */

    for (i=0; i<seqlen; i++)
	ccount[i] = 0;

    /* Print output in CASP RR format with optional PPV estimated from final Z-score */
    if (rawscflg)
	for (i=0; i<ncon; i++)
	    printf("%d %d 0 8 %f\n", sclist[i].i+1, sclist[i].j+1, sclist[i].sc);
    else
	for (i=0; i<ncon; i++)
	{
	    zscore = (sclist[i].sc - mean) / sd;
	    ppv = 0.904 / (1.0 + 16.61 * exp(-0.8105 * zscore));
	    if (ppv >= 0.5 || (!ccount[sclist[i].i] || !ccount[sclist[i].j]) || !filtflg)
	    {
		printf("%d %d 0 8 %f\n", sclist[i].i+1, sclist[i].j+1, ppv);
		ccount[sclist[i].i]++;
		ccount[sclist[i].j]++;
	    }
	}
    
    return 0;
}
