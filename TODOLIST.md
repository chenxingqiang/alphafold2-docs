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