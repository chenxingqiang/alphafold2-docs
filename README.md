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
- [x] 1. Thompson, M. C., Yeates, T. O. & Rodriguez, J. A. Advances in methods for atomic resolution macromolecular structure determination. F1000Res. 9, (2020).

- [x] 2. Bai, X.-C., McMullan, G. & Scheres, S. H. W. How cryo-EM is revolutionizing structural biology. Trends Biochem. Sci. 40, 49–57 (2015).

- [x] 3. Jaskolski, M., Dauter, Z. & Wlodawer, A. A brief history of macromolecular crystallography, illustrated by a family tree and its Nobel fruits. FEBS J. 281, 3985–4009 (2014).

- [x] 4. Wüthrich, K. The way to NMR structures of proteins. Nat. Struct. Biol. 8, 923–925 (2001).

- [x] 5. wwPDB Consortium. Protein Data Bank: the single global archive for 3D macromolecular structure data. Nucleic Acids Res. 47, D520–D528 (2018).

- [x] 6. Mitchell, A. L. et al. MGnify: the microbiome analysis resource in 2020. Nucleic Acids Res. 48, D570–D578 (2019).

- [x] 7. Steinegger, M., Mirdita, M. & Söding, J. Protein-level assembly increases protein sequence recovery from metagenomic samples manyfold. Nat. Methods 16, 603–606 (2019).

- [x] 8. Dill, K. A., Ozkan, S. B., Shell, M. S. & Weikl, T. R. The protein folding problem. Annu. Rev. Biophys. 37, 289–316 (2008).

- [x] 9. Anfinsen, C. B. Principles that Govern the Folding of Protein Chains. Science 181, 223–230 (1973).

- [x] 10. Senior, A. W. et al. Improved protein structure prediction using potentials from deep learning. Nature 577, 706–710 (2020).

- [x] 11. Wang, S., Sun, S., Li, Z., Zhang, R. & Xu, J. Accurate De Novo Prediction of Protein Contact Map by Ultra-Deep Learning Model. PLoS Comput. Biol. 13, e1005324 (2017).

- [x] 12. Zheng, W. et al. Deep-learning contact-map guided protein structure prediction in CASP13. Proteins: Struct. Funct. Bioinf. 87, 1149–1164 (2019).

- [x] 13. Abriata, L. A., Tamò, G. E. & Dal Peraro, M. A further leap of improvement in tertiary structure prediction in CASP13 prompts new routes for future assessments. Proteins: Struct. Funct. Bioinf. 87, 1100–1112 (2019).

- [x] 14. Pearce, R. & Zhang, Y. Deep learning techniques have significantly impacted protein structure prediction and protein design. Curr. Opin. Struct. Biol. 68, 194–207 (2021). 

- [ ] 15. Moult, J., Fidelis, K., Kryshtafovych, A., Schwede, T. & Topf, M. Critical Assessment of Techniques for Protein Structure Prediction, Fourteenth Round: Abstract Book. (2020). 

- [x] 16. Brini, E., Simmerling, C. & Dill, K. Protein storytelling through physics. Science 370,
- [x] (2020).

- [x] 17. Sippl, M. J. Calculation of conformational ensembles from potentials of mena force: an approach to the knowledge-based prediction of local structures in globular proteins.J. Mol. Biol. 213, 859–883 (1990).

- [x] 18. Šali, A. & Blundell, T. L. Comparative protein modelling by satisfaction of spatial restraints. J. Mol. Biol. 234, 779–815 (1993).

- [x] 19. Roy, A., Kucukural, A. & Zhang, Y. I-TASSER: a unified platform for automated protein structure and function prediction. Nat. Protoc. 5, 725–738 (2010).

- [x] 20. Altschuh, D., Lesk, A. M., Bloomer, A. C. & Klug, A. Correlation of co-ordinated amino acid substitutions with function in viruses related to tobacco mosaic virus. Journal of Molecular Biology 193, 693–707 (1987).

- [x] 21. Shindyalov, I. N., Kolchanov, N. A. & Sander, C. Can three-dimensional contacts in protein structures be predicted by analysis of correlated mutations? Protein Eng. 7, 349–358 (1994).

- [x] 22. Weigt, M., White, R. A., Szurmant, H., Hoch, J. A. & Hwa, T. Identification of direct residue contacts in protein-protein interaction by message passing. Proceedings of the National Academy of Sciences 106, 67–72 (2009).

- [x] 23. Marks, D. S. et al. Protein 3D structure computed from evolutionary sequence variation. PLoS One 6, e28766 (2011).

- [x] 24. Jones, D. T., Buchan, D. W. A., Cozzetto, D. & Pontil, M. PSICOV: precise structural contact prediction using sparse inverse covariance estimation on large multiple sequence alignments. Bioinformatics 28, 184–190 (2012).

- [x] 25. Moult, J., Pedersen, J. T., Judson, R. & Fidelis, K. A large-scale experiment to assess protein structure prediction methods. Proteins: Structure, Function, and Genetics 23,(1995).

- [x] 26. Kryshtafovych, A., Schwede, T., Topf, M., Fidelis, K. & Moult, J. Critical assessment of methods of protein structure prediction (CASP)—Round XIII. Proteins: Struct. Funct.Bioinf. 87, 1011–1020 (2019).

- [x] 27. Zhang, Y. & Skolnick, J. Scoring function for automated assessment of protein structure template quality. Proteins: Struct. Funct. Bioinf. 57, 702–710 (2004).

- [x] 28. Tu, Z. & Bai, X. Auto-context and its application to high-level vision tasks and 3d brain image segmentation. IEEE Trans. Pattern Anal. Mach. Intell. 32, 1744–1757 (2009).

- [x] 29. Carreira, J., Agrawal, P., Fragkiadaki, K. & Malik, J. Human pose estimation with iterative error feedback. in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition 4733–4742 (2016).

- [x] 30. Mirabello, C. & Wallner, B. rawMSA: End-to-end Deep Learning using raw Multiple Sequence Alignments. PLoS One 14, e0220182 (2019).

- [x] 31. Huang, Z. et al. CCNet: Criss-Cross Attention for Semantic Segmentation. in Proceedings of the IEEE/CVF International Conference on Computer Vision 603–612 (2019). 

- [x] 32. Hornak, V. et al. Comparison of multiple Amber force fields and development of improved protein backbone parameters. Proteins: Struct. Funct. Bioinf. 65, 712–725(2006).

- [x] 33. Zemla, A. LGA – a Method for Finding 3D Similarities in Protein Structures. Nucleic Acids Res. 31, 3370–3374 (2003).

- [x] 34. Mariani, V., Biasini, M., Barbato, A. & Schwede, T. lDDT: A local superposition-free score for comparing protein structures and models using distance difference tests. Bioinformatics 29, 2722–2728 (2013).

- [x] 35. Xie, Q., Luong, M.-T., Hovy, E. & Le, Q. V. Self-training with noisy student improves magenet classification. in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 10687–10698 (2020).

- [x] 36. Mirdita, M. et al. Uniclust databases of clustered and deeply annotated protein sequences and alignments. Nucleic Acids Res. 45, D170–D176 (2017).

- [x] 37. Devlin, J., Chang, M.-W., Lee, K. & Toutanova, K. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies 1, 4171–4186 (2019).

- [x] 38. Rao, R. et al. MSA Transformer. biorXiv preprint 2021. 02. 12. 430858 (2021).

- [x] 39. Kuhlman, B. & Bradley, P. Advances in protein structure prediction and design. Nat. Rev.Mol. Cell Biol. 20, 681–697 (2019).

- [x] 40. Marks, D. S., Hopf, T. A. & Sander, C. Protein structure prediction from sequence variation.
- [x] Nat. Biotechnol. 30, 1072–1080 (2012).

- [x] 41. Qian, N. & Sejnowski, T. J. Predicting the secondary structure of globular proteins using neural network models. J. Mol. Biol. 202, 865–884 (1988).

- [x] 42. Fariselli, P., Olmea, O., Valencia, A. & Casadio, R. Prediction of contact maps with neural
- [x] networks and correlated mutations. Protein Eng. 14, 835–843 (2001).

- [x] 43. Yang, J. et al. Improved protein structure prediction using predicted interresidue orientations. Proc. Natl. Acad. Sci. U. S. A. 117, 1496–1503 (2020).

- [x] 44. Li, Y. et al. Deducing high-accuracy protein contact-maps from a triplet of coevolutionary matrices through deep residual convolutional networks. PLoS Comput. Biol. 17,e1008865 (2021).

- [x] 45. He, K., Zhang, X., Ren, S. & Sun, J. Deep residual learning for image recognition. in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition 770–778(2016).

- [x] 46. AlQuraishi, M. End-to-End Differentiable Learning of Protein Structure. Cell Systems 8,292–301.e3 (2019).

- [x] 47. Senior, A. W. et al. Protein structure prediction using multiple deep neural networks in the 13th Critical Assessment of Protein Structure Prediction (CASP13). Proteins: Struct. Funct. Bioinf. 87, 1141–1148 (2019).

- [x] 48. Ingraham, J., Riesselman, A. J., Sander, C. & Marks, D. S. Learning Protein Structure with a Differentiable Simulator. in Proceedings of the International Conference on Learning Representations (2019).

- [x] 49. Li, J. Universal Transforming Geometric Network. arXiv preprint arXiv:1908. 00723 (2019).

- [x] 50. Xu, J., Mcpartlon, M. & Li, J. Improved protein structure prediction by deep learning irrespective of co-evolution information. Nature Machine Intelligence (2021).

- [x] 51. Vaswani, A. et al. Attention Is All You Need. in Advances in Neural Information Processing Systems 5998–6008 (2017).

- [x] 52. Wang, H. et al. Axial-deeplab: Stand-alone axial-attention for panoptic segmentation. in European Conference on Computer Vision 108–126 (Springer, 2020).

- [x] 53. Alley, E. C., Khimulya, G., Biswas, S., AlQuraishi, M. & Church, G. M. Unified rational protein engineering with sequence-based deep representation learning. Nat. Methods 16, 1315–1322 (2019).

- [x] 54. Heinzinger, M. et al. Modeling aspects of the language of life through transfer-learning protein sequences. BMC Bioinformatics 20, 1–17 (2019).

- [x] 55. Rives, A. et al. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. Proceedings of the National Academy of Sciences 118, (2021).

- [x] 56. Pereira, J. et al. High-accuracy protein structure prediction in CASP14. Proteins (2021) https://doi.org/10.1002/prot.26171.

- [x] 57. Gupta, M. et al. CryoEM and AI reveal a structure of SARS-CoV-2 Nsp2, a multifunctional protein involved in key host processes. bioRxiv (2021) https://doi.org/10.1101/ 2021.05.10.443524.


- [x] 58. Ingraham, J., Garg, V. K., Barzilay, R. & Jaakkola, T. Generative models for graph-based protein design. in Proceedings of the 33rd Conference on Neural Information Processing Systems (2019).

- [x] 59. Johnson, L. S., Eddy, S. R. & Portugaly, E. Hidden Markov model speed heuristic and iterative HMM search procedure. BMC Bioinformatics 11, 1–8 (2010).

- [x] 60. Remmert, M., Biegert, A., Hauser, A. & Söding, J. HHblits: lightning-fast iterative protein sequence searching by HMM-HMM alignment. Nat. Methods 9, 173–175 (2012).

- [x] 61. Bateman, A. et al. UniProt: the universal protein knowledgebase in 2021. Nucleic Acids Res. (2020).

- [x] 62. Steinegger, M. & Söding, J. Clustering huge protein sequence sets in linear time. Nat. Commun. 9, 1–8 (2018).

- [x] 63. Steinegger, M. & Söding, J. MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. Nat. Biotechnol. 35, 1026–1028 (2017).

- [x] 64. Deorowicz, S., Debudaj-Grabysz, A. & Gudyś, A. FAMSA: Fast and accurate multiple sequence alignment of huge protein families. Sci. Rep. 6, 1–13 (2016).

- [x] 65. Steinegger, M. et al. HH-suite3 for fast remote homology detection and deep protein annotation. BMC Bioinformatics 20, 1–15 (2019).

- [x] 66. Suzek, B. E. et al. UniRef clusters: a comprehensive and scalable alternative for improving sequence similarity searches. Bioinformatics 31, 926–932 (2015).

- [x] 67. Eddy, S. R. Accelerated profile HMM searches. PLoS Comput. Biol. 7, e1002195 (2011).

- [x] 68. Eastman, P. et al. OpenMM 7: Rapid development of high performance algorithms for molecular dynamics. PLoS Comput. Biol. 13, 1–17 (2017).

- [x] 69. Ashish, A. M. A. et al. TensorFlow: Large-Scale Machine Learning on Heterogeneous
- [x] Systems. arXiv preprint arXiv:1603. 04467 (2015).

- [x] 70. Reynolds, M. et al. Open sourcing Sonnet - a new library for constructing neural networks. https://deepmind.com/blog/open-sourcing-sonnet/ (2017).

- [x] 71. Harris, C. R. et al. Array programming with NumPy. Nature 585, 357–362 (2020).

- [x] 72. Van Rossum, G. & Drake, F. L. Python 3 Reference Manual. (CreateSpace, 2009).

- [x] 73. Bisong, E. Google Colaboratory. in Building Machine Learning and Deep Learning Models
- [x] on Google Cloud Platform: A Comprehensive Guide for Beginners 59–64 (Apress,2019).

- [x] 74. XLA: Optimizing Compiler for TensorFlow. https://www.tensorflow.org/xla.

- [x] 75. Wu, T., Hou, J., Adhikari, B. & Cheng, J. Analysis of several key factors influencing deep learning-based inter-residue contact prediction. Bioinformatics 36, 1091–1098 (2020).

- [x] 76. Jiang, W. et al. MrpH, a new class of metal-binding adhesin, requires zinc to mediate biofilm formation. PLoS Pathog. 16, e1008707 (2020).

- [x] 77. Dunne, M., Ernst, P., Sobieraj, A., Pluckthun, A. & Loessner, M. J. The M23 peptidase domain of the Staphylococcal phage 2638A endolysin. https://doi.org/10.2210/pdb6YJ1/ pdb (2020).

- [x] 78. Drobysheva, A. V. et al. Structure and function of virion RNA polymerase of a crAss-like phage. Nature 589, 306–309 (2021).

- [x] 79. Flaugnatti, N. et al. Structural basis for loading and inhibition of a bacterial T6SS phospholipase effector by the VgrG spike. EMBO J. 39, e104129 (2020).

- [x] 80. ElGamacy, M. et al. An Interface-Driven Design Strategy Yields a Novel, Corrugated Protein Architecture. ACS Synth. Biol. 7, 2226–2235 (2018).

- [x] 81. Lim, C. J. et al. The structure of human CST reveals a decameric assembly bound to telomeric DNA. Science 368, 1081–1085 (2020).

- [x] 82. Debruycker, V. et al. An embedded lipid in the multidrug transporter LmrP suggests a mechanism for polyspecificity. Nat. Struct. Mol. Biol. 27, 829–835 (2020).

- [x] 83. Flower, T. G. et al. Structure of SARS-CoV-2 ORF8, a rapidly evolving immune evasion protein. Proc. Natl. Acad. Sci. U. S. A. 118, (2021).

------------------------------------------------------------------------------------------
## Data availability
All input data are freely available from public sources.

Structures from the PDB were used for training and as templates (https://www.wwpdb.org/ftp/pdb-ftp-sites; for the associated sequence data and 40% sequence clustering see also https://ftp.wwpdb. org/pub/pdb/derived_data/ and https://cdn.rcsb.org/resources/ sequence/clusters/bc-40.out).

 Training used a version of the PDB downloaded 28/08/2019, while CASP14 template search used a version downloaded 14/05/2020. Template search also used the PDB70 data- base, downloaded 13/05/2020 (https://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/).

We show experimental structures from the PDB with accessions
6Y4F<sup>76</sup>, 6YJ1<sup>77</sup>, 6VR4<sup>78</sup>, 6SK0<sup>79</sup>, 6FES<sup>80</sup>, 6W6W<sup>81</sup>, 6T1Z<sup>82</sup>, and 7JTL<sup>83</sup>. 

For MSA lookup at both training and prediction time, 

we used UniRef90 v2020_01 (https://ftp.ebi.ac.uk/pub/databases/uniprot/previous_releases/release-2020_01/uniref/), 

BFD (https://bfd.mmseqs.com), Uniclust30 v2018_08 (https://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/), 

and MGnify clusters v2018_12 (https://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/2018_12/). Uniclust30 v2018_08 was further used as input for constructing a distillation structure dataset.


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
