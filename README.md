# ref-Alphafold-Code

![proteins](img/proteins.jpeg)
## ToDo Lists
### 1. AlphaFold2 Source Code Review

  #### **alphafold**

common
- [ ] confidence.py
- [ ] protein.py
- [ ] protein_test.py
- [ ] residue_constants.py
- [ ] residue_constants_test.py

data
- [ ] mmcif_parsing.py
- [ ] parsers.py
- [ ] pipeline.py
- [ ] templates.py

tools
- [ ] hhblits.py
- [ ] hhsearch.py
- [ ] hmmbuild.py
- [ ] hmmsearch.py
- [ ] jackhmmer.py
- [ ] kalign.py
- [ ] utils.py

model
 - [ ] all_atom.py
 - [ ] all_atom_test.py
 - [ ] common_modules.py
 - [ ] config.py
 - [ ] data.py
 - [ ] features.py
 - [ ] folding.py
 - [ ] layer_stack.py
 - [ ] layer_stack_test.py
 - [ ] lddt.py
 - [ ] lddt_test.py
 - [ ] mapping.py
 - [ ] model.py
 - [ ] modules.py
 - [ ] prng.py
 - [ ] prng_test.py
 - [ ] quat_affine.py
 - [ ] quat_affine_test.py
 - [ ] r3.py
 - [ ] utils.py

tf
- [ ] data_transforms.py
- [ ] input_pipeline.py
- [ ] protein_features.py
- [ ] protein_features_test.py
- [ ] proteins_dataset.py
- [ ] shape_helpers.py
- [ ] shape_helpers_test.py
- [ ] shape_placeholders.py
- [ ] utils.py
  

relax
- [ ] amber_minimize.py
- [ ] amber_minimize_test.py
- [ ] cleanup.py
- [ ] cleanup_test.py
- [ ] relax.py
- [ ] relax_test.py

----------------------------------------------------------------------------------------------------------------------

#### 2. Create Unit Test for 32 Algorithms

|   Function Name   (Algorithms)                 	|   Instruction                                                          	| CheckBox 	|
|------------------------------------	|------------------------------------------------------------------------	|----------	|
|   MSABlockDeletion                 	|     MSA block deletion                                                  	|  [ ]    	|
|   Inference                        	|    AlphaFold Model Inference                                           	|  [ ]    	|
|   InputEmbedder                    	|    Embeddings for initial representations                              	|  [ ]    	|
|   relpos                           	|    Relative position encoding                                          	|  [ ]    	|
|   one_hot                          	|    One-hot encoding with nearest bin                                   	|  [ ]    	|
|   EvoformerStack                   	|    Evoformer stack                                                     	|  [ ]    	|
|   MSARowAttentionWithPairBias      	|    MSA row-wise gated self-attention with pair bias                    	|  [ ]    	|
|   MSAColumnAttention               	|    MSA column-wise gated self-attention                                	|   [ ]     	|
|   MSATransition                    	|    Transition layer in the MSA stack                                   	|   [ ]     	|
|   OuterProductMean                 	|    Outer product mean                                                  	|    [ ]     	|
|   TriangleMultiplicationOutgoing   	|    Triangular multiplicative update using “outgoing” edges             	|    [ ]     	|
|   TriangleMultiplicationIncoming   	|    Triangular multiplicative update using “incoming” edges             	|    [ ]     	|
|   TriangleAttentionStartingNode    	|    Triangular gated self-attention around starting node                	|    [ ]     	|
|   TriangleAttentionEndingNode      	|    Triangular gated self-attention around ending node                  	|    [ ]     	|
|   PairTransition                   	|    Transition layer in the pair stack                                  	|    [ ]     	|
|   TemplatePairStack                	|    Template pair stack                                                 	|    [ ]     	|
|   TemplatePointwiseAttention       	|    Template pointwise attention                                        	|    [ ]     	|
|   ExtraMsaStack                    	|    Extra MSA stack                                                     	|    [ ]     	|
|   MSAColumnGlobalAttention         	|    MSA global column-wise gated self-attention                         	|    [ ]     	|
|   StructureModule                  	|    Structure module                                                    	|    [ ]     	|
|   rigidFrom3Points                 	|   Rigid from 3 points using the Gram–Schmidt process                   	|    [ ]     	|
|   InvariantPointAttention          	|    Invariant point attention(IPA)                                      	|    [ ]     	|
|   BackboneUpdate                   	|    Backbone update                                                     	|    [ ]     	|
|   computeAllAtomCoordinates        	|    Compute all atom coordinates                                        	|    [ ]     	|
|   makeRotX                         	|   Make a transformation that rotates around the x-axis                 	|    [ ]     	|
|   renameSymmetricGroundTruthAtoms  	|    Rename symmetric ground truth atoms                                 	|    [ ]     	|
|   torsionAngleLoss                 	|   Side chain and backbone torsion angle loss                           	|    [ ]     	|
|   computeFAPE                      	|   Compute the Frame aligned point error                                	|    [ ]     	|
|   predictPerResidueLDDT            	|   Predict model confidence pLDDT                                       	|    [ ]     	|
|   RecyclingInference               	|    Generic recycling inference procedure                               	|    [ ]     	|
|   RecyclingTraining                	|    Generic recycling training procedure                                	|    [ ]     	|
|   RecyclingEmbedder                	|    Embedding of Evoformer and Structure module outputs for recycling   	|    [ ]     	|

## Reference Papers List 

## Code and programmings availability
### Source code
 for the AlphaFold model, trained weights, and an inference script is available under an open-source license at https://github.com/deepmind/alphafold. 

### Neural networks
 Neural networks were developed with 
- TensorFlow v1 (https://github.com/tensorflow/tensorflow), 
- Sonnet v1 (https://github.com/deepmind/sonnet),
- JAX v0.1.69 (https://github.com/google/jax/), 
- Haiku v0.0.4 (https://github.com/deepmind/dm-haiku).

### MSA search
For MSA search on 
- UniRef90, MGnify clusters, 
and reduced BFD we used jackhmmer and for template search on the PDB SEQRES we used 
- hmmsearch, both from HMMER v3.3 (http://eddylab.org/soft-ware/hmmer/).

For template search against PDB70, we used HHsearch from HH-suite v3.0-beta.3 14/07/2017 (https://github.com/soedinglab/hh-suite). 
For constrained relaxation of structures, we used OpenMM v7.3.1 (https://github.com/openmm/openmm) with the Amber99sb force field.


### Docking analysis
 Docking analysis on DGAT used 
 - P2Rank v2.1 (https://github.com/rdk/p2rank), 
 - MGLTools v1.5.6 (https://ccsb.scripps.edu/mgltools/) 
 - and AutoDockVina v1.1.2 (http://vina.scripps.edu/download/) on a workstation running Debian GNU/Linux rodete 5.10.40-1rodete1-amd64 x86_64.

### Data analysis 
Data analysis used 
- Python v3.6 (https://www.python.org/), 
- NumPy v1.16.4 (https://github.com/numpy/numpy), 
- SciPy v1.2.1 (https://www.scipy.org/), 
- seaborn v0.11.1 (https://github.com/mwaskom/seaborn), 
- scikit-learn v0.24.0 (https://github.com/scikit-learn/), 
- Matplotlib v3.3.4 (https://github.com/matplotlib/matplotlib), 
- pandas v1.1.5 (https://github.com/pandas-dev/pandas), 
- and Colab (https://research.google.com/colaboratory). 
- TM-align v20190822 (https://zhanglab.dcmb.med.umich.edu/TM-align) was used for computing TM-scores.

 ### Structure analysis  
 Structure analysis used Pymol v2.3.0 (https://github.com/schrodinger/pymol-open-source).



## Public conference 
Alphafold2 paper Reading mp4 
https://drive.google.com/drive/folders/1lMmrYRO4fHBcDE31YTKS-tHRDADDORpa?usp=sharing 
Alphafold2 
paper Reading PPT 初版下载链接：
https://drive.google.com/file/d/1Py6jXKTUyCvCJF2Q3WWq-8KYNv6BI28X/view?usp=sharing
