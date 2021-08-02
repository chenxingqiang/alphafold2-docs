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

- [ ] 1. SWISS-MODEL Homo sapiens (Human). https://swissmodel.expasy.org/repository/species/9606 (2021).
- [ ] 2. Jumper, J. et al. Highly accurate protein structure prediction with AlphaFold. Nature https://doi.org/10.1038/s41586-021-03819-2 (2021).
- [ ] 3. Lander, E. S. et al. Initial sequencing and analysis of the human genome. Nature 409, 860–921 (2001).
- [ ] 4. Venter, J. C. et al. The sequence of the human genome. Science 291, 1304–1351 (2001).
- [ ] 5. wwPDB Consortium. Protein Data Bank: the single global archive for 3D macromolecular
structure data. Nucleic Acids Res. 47, D520–D528 (2018).
- [ ] 6. Bateman, A. et al. UniProt: The Universal Protein Knowledgebase in 2021. Nucleic Acids
Res. (2020).
- [ ] 7. Slabinski, L. et al. The challenge of protein structure determination—lessons from
structural genomics. Protein Sci. 16, 2472–2482 (2007).
- [ ] 8. Elmlund, D., Le, S. N. & Elmlund, H. High-resolution cryo-EM: the nuts and bolts. Curr.
Opin. Struct. Biol. 46, 1–6 (2017).
- [ ] 9. Yang, J. et al. Improved protein structure prediction using predicted interresidue
orientations. Proc. Natl Acad. Sci. USA 117, 1496–1503 (2020).
- [ ] 10. Greener, J. G., Kandathil, S. M. & Jones, D. T. Deep learning extends de novo protein
modelling coverage of genomes using iteratively predicted structural constraints. Nat.
Commun. 10, 1–13 (2019).
- [ ] 11. Michel, M., Menéndez Hurtado, D., Uziela, K. & Elofsson, A. Large-scale structure
prediction by improved contact predictions and model quality assessment.
Bioinformatics 33, i23–i29 (2017).
- [ ] 12. Ovchinnikov, S. et al. Large-scale determination of previously unsolved protein structures
using evolutionary information. Elife 4, e09248 (2015).
- [ ] 13. Zhang, J., Yang, J., Jang, R. & Zhang, Y. GPCR-I-TASSER: a hybrid approach to G
protein-coupled receptor structure modeling and the application to the human genome.
Structure 23, 1538–1549 (2015).
- [ ] 14. Bender, B. J., Marlow, B. & Meiler, J. Improving homology modeling from low-sequence identity
templates in Rosetta: A case study in GPCRs. PLoS Comput. Biol. 16, e1007597 (2020).
- [ ] 15. Drew, K. et al. The Proteome Folding Project: proteome-scale prediction of structure and
function. Genome Res. 21, 1981–1994 (2011).
- [ ] 16. Xu, D. & Zhang, Y. Ab initio structure prediction for Escherichia coli: towards genome-wide
protein structure modeling and fold assignment. Sci. Rep. 3, 1–11 (2013).
- [ ] 17. Waterhouse, A. et al. SWISS-MODEL: Homology modelling of protein structures and
complexes. Nucleic Acids Res. 46, W296–W303 (2018).
- [ ] 18. Sillitoe, I. et al. Genome3D: Integrating a collaborative data pipeline to expand the depth
and breadth of consensus protein structure annotation. Nucleic Acids Res. 48,
D314–D319 (2020).
- [ ] 19. Pieper, U. et al. MODBASE: A database of annotated comparative protein structure
models and associated resources. Nucleic Acids Res. 42, D336–D346 (2014).
- [ ]  20. Huang, P.-S., Boyken, S. E. & Baker, D. The coming of age of de novo protein design.
Nature 537, 320–327 (2016).
- [ ] 21. Kuhlman, B. & Bradley, P. Advances in protein structure prediction and design. Nat. Rev.
Mol. Cell Biol. 20, 681–697 (2019).
- [ ] 22. Consortium, G. O. The gene ontology resource: 20 years and still GOing strong. Nucleic
Acids Res. 47, D330–D338 (2019).
- [ ] 23. Zhou, N. et al. The CAFA challenge reports improved protein function prediction and new
functional annotations for hundreds of genes through experimental screens. Genome
Biol. 20, 1–23 (2019).
- [ ] 24. Gligorijević, V. et al. Structure-based protein function prediction using graph
convolutional networks. Nat. Commun. 12, 1–14 (2021).
- [ ] 25. Necci, M., Piovesan, D. & Tosatto, S. C. E. Critical assessment of protein intrinsic disorder
prediction. Nat. Methods 1–10 (2021).
- [ ] 26. Sillitoe, I. et al. CATH: expanding the horizons of structure-based functional annotations
for genome sequences. Nucleic Acids Res. 47, D280–D284 (2019).
- [x] 27. Andreeva, A., Kulesha, E., Gough, J. & Murzin, A. G. The SCOP database in 2020:
expanded classification of representative family and superfamily domains of known
protein structures. Nucleic Acids Res. 48, D376–D382 (2020).
- [x] 28. Mistry, J. et al. Pfam: The protein families database in 2021. Nucleic Acids Res. 49,
D412–D419 (2021).
- [x] 29. Kryshtafovych, A., Schwede, T., Topf, M., Fidelis, K. & Moult, J. Critical assessment of
methods of protein structure prediction (CASP)—Round XIII. Proteins: Struct. Funct.
Bioinf. 87, 1011–1020 (2019).
- [x] 30. Lupas, A. N., Pereira, J. & Hartmann, M. D. High Accuracy Assessment in CASP14. https://
predictioncenter.org/casp14/doc/presentations/2020_11_30_HighAccuracy_assessmentLupas-Pereira-Hartmann.pdf (2020).
- [x] 31. Senior, A. W. et al. Improved protein structure prediction using potentials from deep
learning. Nature 577, 706–710 (2020).
- [x] 32. Zhang, Y. Protein structure prediction: when is it useful? Curr. Opin. Struct. Biol. 19,
145–155 (2009).

- [x] 33. Flower, T. G. & Hurley, J. H. Crystallographic molecular replacement using an in silico-generated search model of SARS-CoV-2 ORF8. Protein Sci. 30, 728–734 (2021).
- [x] 34. Egbert, M. & Vajda, S. CASP14 Functional Assessment: Conservation of Binding Properties in Modeled Proteins. https://predictioncenter.org/casp14/doc/presentations/2020_12_03_Function_Assessment_VajdaLab_KozakovLab.pdf (2020).
- [x] 35. Mariani, V., Biasini, M., Barbato, A. & Schwede, T. lDDT: A local superposition-free score for comparing protein structures and models using distance difference tests. Bioinformatics 29, 2722–2728 (2013).
- [x] 36. Ashburner, M. et al. Gene Ontology: tool for the unification of biology. Nat. Genet. 25, 25–29 (2000).
- [x] 37. Carbon, S. et al. The Gene Ontology resource: enriching a GOld mine. Nucleic Acids Res. 49, D325–D334 (2021).
- [x] 38. Hopkins, A. L. & Groom, C. R. The druggable genome. Nat. Rev. Drug Discov. 1, 727–730 (2002).
- [x] 39. Haas, J. et al. Introducing ‘best single template’ models as reference baseline for the Continuous Automated Model Evaluation (CAMEO). Proteins: Struct. Funct. Bioinf. 87, 1378–1387 (2019).
- [x] 40. Haas, J. et al. Continuous Automated Model EvaluatiOn (CAMEO) complementing the critical assessment of structure prediction in CASP12. Proteins: Struct. Funct. Bioinf. 86, 387–398 (2018).
- [x] 41. Schaeffer, R. D., Kinch, L. & Grishin, N. CASP14: InterDomain Performance. https://predictioncenter.org/casp14/doc/presentations/2020_12_02_Interdomain_assessment1_Schaeffer.pdf (2020).
- [x] 42. Zhang, Y. & Skolnick, J. Scoring function for automated assessment of protein structure template quality. Proteins: Struct. Funct. Bioinf. 57, 702–710 (2004).
- [x] 43. Grinter, R. et al. Protease-associated import systems are widespread in Gram-negative bacteria. PLoS Genet. 15, e1008435 (2019).
- [x] 44. Pan, C.-J., Lei, K.-J., Annabi, B., Hemrika, W. & Chou, J. Y. Transmembrane topology of glucose-6-phosphatase. J. Biol. Chem. 273, 6144–6148 (1998).
- [x] 45. Van Schaftingen, E. & Gerin, I. The glucose-6-phosphatase system. Biochem. J. 362, 513–532 (2002).
- [x] 46. Messerschmidt, A., Prade, L. & Wever, R. Implications for the catalytic mechanism of the vanadium-containing enzyme chloroperoxidase from the fungus Curvularia inaequalis by X-ray structures of the native and peroxide form. Biol. Chem. 378, 309–315 (1997).
- [x] 47. Amin, N. B. et al. Targeting diacylglycerol acyltransferase 2 for the treatment of nonalcoholic steatohepatitis. Sci. Transl. Med. 11, (2019).
- [x] 48. Futatsugi, K. et al. Discovery and optimization of imidazopyridine-based inhibitors of diacylglycerol acyltransferase 2 (DGAT2). J. Med. Chem. 58, 7173–7185 (2015).
- [x] 49. Birch, A. M. et al. Discovery of a potent, selective, and orally efficacious pyrimidinooxazinyl bicyclooctaneacetic acid diacylglycerol acyltransferase-1 inhibitor. J. Med. Chem. 52, 1558–1568 (2009).
- [x] 50. Cao, H. Structure-function analysis of diacylglycerol acyltransferase sequences from 70 organisms. BMC Res. Notes 4, 1–24 (2011).
- [x] 51. Wang, L. et al. Structure and mechanism of human diacylglycerol O-acyltransferase 1. Nature 581, 329–332 (2020).
- [x] 52. Stone, S. J., Levin, M. C. & Farese, R. V., Jr Membrane topology and identification of key functional amino acid residues of murine acyl-CoA: diacylglycerol acyltransferase-2.
J. Biol. Chem. 281, 40273–40282 (2006).
- [x] 53. Rigoli, L., Lombardo, F. & Di Bella, C. Wolfram syndrome and WFS1 gene. Clin. Genet. 79,
103–117 (2011).
- [x] 54. Urano, F. Wolfram syndrome: diagnosis, management, and treatment. Curr. Diab. Rep. 16,6 (2016).
- [x] 55. Schäffer, D. E., Iyer, L. M., Burroughs, A. M. & Aravind, L. Functional innovation in the
evolution of the calcium-dependent system of the eukaryotic endoplasmic reticulum.
Front. Genet. 11, 34 (2020).
- [x] 56. Guardino, K. M., Sheftic, S. R., Slattery, R. E. & Alexandrescu, A. T. Relative stabilities of
conserved and non-conserved structures in the OB-fold superfamily. Int. J. Mol. Sci. 10,
2412–2430 (2009).
- [x] 57. Zhang, Y. & Skolnick, J. TM-align: a protein structure alignment algorithm based on the
TM-score. Nucleic Acids Res. 33, 2302–2309 (2005).
- [x] 58. Das, D. et al. The structure of KPN03535 (gi|152972051), a novel putative lipoprotein from
Klebsiella pneumoniae, reveals an OB-fold. Acta Crystallogr. Sect. F Struct. Biol. Cryst.
Commun. 66, 1254–1260 (2010).
- [x] 59. Fass, D. & Thorpe, C. Chemistry and enzymology of disulfide cross-linking in proteins.
Chem. Rev. 118, 1169–1198 (2018).
- [x] 60. Basile, W., Salvatore, M., Bassot, C. & Elofsson, A. Why do eukaryotic proteins contain
more intrinsically disordered regions? PLoS Comput. Biol. 15, e1007186 (2019).
- [x] 61. Bhowmick, A. et al. Finding our way in the dark proteome. J. Am. Chem. Soc. 138,
9730–9742 (2016).
- [x] 62. Oates, M. E. et al. D2P2: database of disordered protein predictions. Nucleic Acids Res. 41,
D508–D516 (2012).
- [x] 63. Hanson, J., Paliwal, K. K., Litfin, T. & Zhou, Y. SPOT-Disorder2: Improved Protein Intrinsic
Disorder Prediction by Ensembled Deep Learning. Genomics Proteomics Bioinformatics
17, 645–656 (2019).
- [x] 64. Dunne, M., Ernst, P., Sobieraj, A., Pluckthun, A. & Loessner, M. J. The M23 peptidase
domain of the Staphylococcal phage 2638A endolysin. https://doi.org/10.2210/pdb6YJ1/pdb (2020).
- [x] 65. Krivák, R. & Hoksza, D. P2Rank: machine learning based tool for rapid and accurate
prediction of ligand binding sites from protein structure. J. Cheminform. 10, 1–12 (2018).
- [x] 66. Li, Y.-C. et al. Structure and noncanonical Cdk8 activation mechanism within an
Argonaute-containing Mediator kinase module. Sci. Adv. 7, eabd4484 (2021).

- [x] 67. Eddy, S. R. A new generation of homology search tools based on probabilistic inference. Genome Informatics 2009: Genome Informatics Series 23, 205–211 (World Scientific, 2009).
- [x] 68. Steinegger, M., Mirdita, M. & Söding, J. Protein-level assembly increases protein sequence recovery from metagenomic samples manyfold. Nat. Methods 16, 603–606 (2019).
- [ ] 69. Schrödinger. The PyMOL Molecular Graphics System, Version 1.8. (2015).
- [ ] 70. Morris, G. M. et al. AutoDock4 and AutoDockTools4: automated docking with selective
receptor flexibility. J. Comput. Chem. 30, 2785–2791 (2009).
- [ ] 71. Trott, O. & Olson, A. J. AutoDock Vina: Improving the speed and accuracy of docking with
a new scoring function, efficient optimization, and multithreading. J. Comput. Chem. 31,
455–461 (2010).
- [ ] 72. Stein, P. E. et al. The crystal structure of pertussis toxin. Structure 2, 45–57 (1994).
- [ ] 73. Necci, M., Piovesan, D., Clementel, D., Dosztányi, Z. & Tosatto, S. C. E. MobiDB-lite 3.0:
fast consensus annotation of intrinsic disorder flavors in proteins. Bioinformatics
(2020).
- [ ] 74. Dyson, H. J. Roles of intrinsic disorder in protein-nucleic acid interactions. Mol. Biosyst. 8,
97–104 (2012).
- [ ] 75. Dunbrack, R. L., Jr & Karplus, M. Backbone-dependent rotamer library for proteins
application to side-chain prediction. J. Mol. Biol. 230, 543–574 (1993).
- [ ] 76. Fischer, A., Smiesko, M., Sellner, M. & Lill, M. A. Decision making in structure-based drug
discovery: visual inspection of docking results. J. Med. Chem. 64, 2489-2500 (2021).


## Data availability
AlphaFold structure predictions for the human proteome are available under a CC-BY-4.0 license at https://alphafold.ebi.ac.uk/.

All input data are freely available from public sources. The human reference proteome together with its xml annotations was obtained from UniProt 2021_02 (https://ftp.ebi.ac.uk/pub/databases/uniprot/previous_releases/release-2021_02/knowledgebase/).

At prediction time, MSA search was performed against UniRef90 2020_03 (https://ftp.ebi.ac.uk/pub/databases/uniprot/previous_releases/release-2020_03/uniref/), 

MGnify clusters 2018_12 (https://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/2018_12/), 

and a reduced version of BFD (produced from as outlined in the Methods from BFD https://bfd.mmseqs.com/). 

Template structures, the SEQRES fasta file, and the 40% sequence clustering were taken from a copy of the PDB downloaded 15/2/2021 (https://www.wwpdb.org/ftp/pdb-ftp-sites; see also https://ftp.wwpdb.org/pub/pdb/derived_data/ and https://cdn.rcsb.org/resources/sequence/clusters/bc-40.out for sequence data). 

Experimental structures are drawn from the same copy of the PDB; we show structures with accessions 6YJ164, 6OFS43,46, 1IDQ46, 1PRT72, 3F1Z58, 7KPX66 and 6VP051. Template search used PDB70, downloaded 10/02/2021 (http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/). 

The CAID dataset was downloaded from (https://idpcentral.org/caid/data/1/reference/disprot-disorder-pdb-atleast.txt). 

CAMEO data was accessed 17/03/2021 from (https://www.cameo3d.org/static/down-loads/modeling/1-year/raw_targets-1-year.public.tar.gz). 

A copy of the current Gene Ontology was downloaded 29/04/2021 from (http://current.geneontology.org/ontology/go.obo).


## Data availability
AlphaFold structure predictions for the human proteome are available under a CC-BY-4.0 license at https://alphafold.ebi.ac.uk/.

All input data are freely available from public sources. The human reference proteome together with its xml annotations was obtained from UniProt 2021_02 (https://ftp.ebi.ac.uk/pub/databases/uniprot/previous_releases/release-2021_02/knowledgebase/).

At prediction time, MSA search was performed against UniRef90 2020_03 (https://ftp.ebi.ac.uk/pub/databases/uniprot/previous_releases/release-2020_03/uniref/), 

MGnify clusters 2018_12 (https://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/2018_12/), 

and a reduced version of BFD (produced from as outlined in the Methods from BFD https://bfd.mmseqs.com/). 

Template structures, the SEQRES fasta file, and the 40% sequence clustering were taken from a copy of the PDB downloaded 15/2/2021 (https://www.wwpdb.org/ftp/pdb-ftp-sites; see also https://ftp.wwpdb.org/pub/pdb/derived_data/ and https://cdn.rcsb.org/resources/sequence/clusters/bc-40.out for sequence data). 

Experimental structures are drawn from the same copy of the PDB; we show structures with accessions 6YJ164, 6OFS43,46, 1IDQ46, 1PRT72, 3F1Z58, 7KPX66 and 6VP051. Template search used PDB70, downloaded 10/02/2021 (http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/). 

The CAID dataset was downloaded from (https://idpcentral.org/caid/data/1/reference/disprot-disorder-pdb-atleast.txt). 

CAMEO data was accessed 17/03/2021 from (https://www.cameo3d.org/static/down-loads/modeling/1-year/raw_targets-1-year.public.tar.gz). 

A copy of the current Gene Ontology was downloaded 29/04/2021 from (http://current.geneontology.org/ontology/go.obo).
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
