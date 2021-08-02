"""Configuration classes for building and running.

The model configuration class contains attributes on:
    input/outputs
    computing
    initialization
    optimization
    queuing
    ciriculum
    architecture
    regularization
    loss

The running configuration class contains attributes on:
    names
    input/outputs
    computing
    optimization
    queuing
    evaluation
    loss

TODO: realign the dictionaries
TODO: rename all the config inputs
"""

from ast import literal_eval

# functions for reading in parsed config file
flt_or_none = lambda x: float(x) if x is not None else None
int_or_none = lambda x: int(x) if x is not None else None
str_or_none = lambda x: None if isinstance(x, str) and x == 'none' else x
str_or_bool = lambda x: (x == 'true' or x == 'True') if isinstance(x, str) else x
eval_if_str = lambda x: literal_eval(x) if isinstance(x, str) else x


class Config(object):
    """Class for configuration settings.
    
    The elements in config take higher precedence than that from the file.

    Attributes:
        file: config file name
        config: dictionary that holds parameters
    """

    def __init__(self, file=None, config={}):
        """Loads configurations from file."""

        if file is not None:
            file_config = self.dict_import(file)
            file_config.update(config)
            config = file_config

        # assign config values
        self._create_config(config)

    def dict_import(self, file):
        """Parse the config file into a dictionary."""

        vars_ = {}
        with open(file) as f:
            for line in f:
                if line[0] != '#':
                    name, var = line.partition(' ')[::2]
                    vars_[name.strip()] = var.strip()
        return vars_

    def _create_config(self, config):
        raise NotImplementedError('Abstract method')

class RGNConfig(Config):
    """Encapsulates parameters for RGN model.

       Options marked with HO indicate that they're completely dependent on higher-order layers being enabled.
       Options marked with pHO indicate that their behavior is partially dependent on higher-order layers.

    TODO: make the names match:
    example: data_files and dataFiles
    """

    def _create_config(self, config):
        self.io = {
            'name':                              config.get('name',                  None), # name to give the model
            'num_edge_residues':             int(config.get('num_edge_residues',       2)), # end point residues to ignore because gold standard end point coordinates tend to be inaccurate
            'num_evo_entries':               int(config.get('num_evo_entries',         20)),
            'data_files':                        config.get('data_files',             None), # a list of file names, used by default (??? what files are we talking about?)
            'data_files_glob':                   config.get('dataFilesGlob',         None), # a glob, used if no data_files are supplied
            'evaluation_sub_groups': eval_if_str(config.get('evaluation_sub_groups',   [])),  # subgroups for evaluation as seen in ProteinNet
            'alphabet_file':                     config.get('alphabetFile',          None), # Alphabet file name. If passed this overrides alphabet_init
            'checkpoints_directory':             config.get('checkpointsDirectory',  None), # Directory to retrieve or save model.
            'logs_directory':                    config.get('logsDirectory',         None), # Directory to log summaries
            'log_model_summaries':   str_or_bool(config.get('logModelSummaries',     True)), # whether to log model summaries
            'log_alphabet':          str_or_bool(config.get('logAlphabet',           False)), # whether to log alphabet to diagnostics
            'detailed_logs':         str_or_bool(config.get('detailedLogs',          True)), # Entails creating histograms.
            'max_checkpoints':       int_or_none(config.get('maxCheckpoints',        None)), # maximum checkpoints to keep when saving model
            'checkpoint_every_n_hours':      int(config.get('checkpointEveryNHours', 24))} # this is in addition to the max_checkpoints. Additional checkpoint after n hours.

        self.computing = {
            'num_cpus':                             int(config.get('numCPUs',                        4)), # number of CPUs to use
            'num_recurrent_parallel_iters':         int(config.get('numRecurrentParallelIters',      32)), # number of iterations to run in parallel for RNN
            'default_device':                           config.get('defaultDevice',                  ''), # default device to place operations
            'functions_on_devices':         eval_if_str(config.get('functionsOnDevices',             {'/cpu:0': ['point_to_coordinate']})), # which functions to place in which device
            'gpu_fraction':                       float(config.get('gpuFraction',                    1)), # Fill only the fraction of GPU memory.
            'allow_gpu_growth':             str_or_bool(config.get('allowGPUGrowth',                 False)), # whether to allow GPU memory to grow as memory usage increases. Not needed if GPU fraction is 1.
            'fill_gpu':                     str_or_bool(config.get('fillGPU',                        False)), # whether to fill up the GPU with allocated memory.
            'num_reconstruction_fragments': int_or_none(config.get('numReconstructionFragments',     6)), # Number of fragments to reconstruct in parallel. Used for creating the 3D structure.
            'num_reconstruction_parallel_iters':    int(config.get('numReconstructionParallelIters', 4)) # Number of parallel iterations for creating the 3D structure.
        }

        self.initialization = {
            'graph_seed':                        int_or_none(config.get('randSeed',                      None)),
            'angle_shift':                       eval_if_str(config.get('angleShift',                    [0., 0., 0.])), # The angles by which to shift the predicted dihedral angles.
            'recurrent_forget_bias':                   float(config.get('recurrentForgetBias',           1)),                               
            'recurrent_init':                    eval_if_str(config.get('recurrentInit',                 None)), # can be list if HO
            'recurrent_seed':                    int_or_none(config.get('recurrentSeed',                 None)), # seed
            'recurrent_out_proj_init':           eval_if_str(config.get('recurrentOutProjInit',          {'base': {}, 'bias': {}})),
            'recurrent_out_proj_seed':           int_or_none(config.get('recurrentOutProjSeed',          None)), # seed
            'recurrent_nonlinear_out_proj_init': eval_if_str(config.get('recurrentNonlinearOutProjInit', {'base': {}, 'bias': {}})),
            'recurrent_nonlinear_out_proj_seed': int_or_none(config.get('recurrentNonlinearOutProjSeed', None)), # seed
            'alphabet_init':                     eval_if_str(config.get('alphabetInit',                  {})),
            'alphabet_seed':                     int_or_none(config.get('alphabetSeed',                  None)), # seed
            'queue_seed':                        int_or_none(config.get('queueSeed',                     None)), # seed for queuing and shuffling
            'dropout_seed':                      int_or_none(config.get('dropoutSeed',                   None)), # seed for all the dropouts
            'zoneout_seed':                      int_or_none(config.get('zoneoutSeed',                   None)), # seed for zone out wrapper
            'evolutionary_multiplier':                 float(config.get('evolutionaryMultiplier',        1)) # Number to multiply the PSSM inputs.
        }

        self.optimization = {
            'optimizer':                       config.get('optimiser',            'steepest'),
            'learning_rate':             float(config.get('learnRate',            0.001)), # all optimizers
            'momentum':                  float(config.get('momentum',             0)),     # momentum, rmsprop, has no analog in autograd
            'beta1':                     float(config.get('beta1',                0.9)),   # adam, momentum in autograd
            'beta2':                     float(config.get('beta2',                0.999)), # adam, hoMomentum in autograd
            'epsilon':                   float(config.get('epsilon',              10e-8)), # adam, rmsprop, adadelta. this should really be 1e-8
            'decay':                     float(config.get('decay',                0.9)),   # rmsprop, adadelta (rho), momentum in autograd
            'initial_accumulator_value': float(config.get('initAccumulatorValue', 0.1)),   # adagrad
            'rescale_behavior':    str_or_none(config.get('rescaleBehavior',      None)),
            'gradient_threshold':        float(config.get('gradientThreshold',    'inf')),
            'recurrent_threshold': flt_or_none(config.get('recurrentThreshold',   None)),  # only TF-based RNNs
            'alphabet_temperature':      float(config.get('alphabetTemperature',  1.0)),
            'batch_size':                  int(config.get('batchSize',            256)),   # batch size
            'num_steps':                   int(config.get('maxSeqLength',         500)),
            'num_epochs':          int_or_none(config.get('numEpochs',            None))
        }

        self.queueing = {
            'file_queue_capacity':        int(config.get('fileQueueCapacity',        1000)),  # Defaults make sense if each file has ~100 sequences. Capacity when queuing.
            'batch_queue_capacity':       int(config.get('batchQueueCapacity',       10000)), # Batch capacity.
            'min_after_dequeue':          int(config.get('minAfterDequeue',          500)), # Minimum  number of elements that will remain in the queue after a dequeue
            'shuffle':            str_or_bool(config.get('shuffle',                  True)), # whether to shuffle or not
            'bucket_boundaries':  eval_if_str(config.get('bucketBoundaries',         None)),
            'num_evaluation_invocations': int(config.get('numEvaluationInvocations', 1)) # must be 1 for training
        }

        self.curriculum = {
            'mode':                str_or_none(config.get('currMode',            None)), # can be 'loss', 'length'
            'behavior':            str_or_none(config.get('currBehavior',        None)), # can be 'fixed_rate', 'loss_threshold', 'loss_change'. If none, no cirriculum updates.
            'slope':                     float(config.get('currSlope',           1.0)), # influences the weights for drmsd
            'base':                      float(config.get('currBase',            4.0)), # influences the weights for drmsd
            'rate':                      float(config.get('currRate',            0.002)), # rate of curr update.
            'threshold':                 float(config.get('currThreshold',       5.0)), # use if 'loss_threshold'
            'change_num_iterations':       int(config.get('currChangeNumIters',  5)),
            'sharpness':                 float(config.get('currSharpness',       20.)), # use for 'loss_change'
            'update_loss_history': str_or_bool(config.get('updateLossHistory',   False)), # whether to update loss history
            'loss_history_subgroup':           config.get('lossHistorySubgroup', 'all')
        }

        self.architecture = {
            'alphabet_size':                            eval_if_str(config.get('alphabetSize',                         None)), # pHO
            'alphabet_trainable':                       str_or_bool(config.get('alphabetTrainable',                    True)),
            'include_primary':                          str_or_bool(config.get('includePrimary',                       True)),
            'include_evolutionary':                     str_or_bool(config.get('includeEvolutionary',                  False)),
            'recurrent_nonlinear_out_proj_size':        eval_if_str(config.get('recurrentNonlinearOutputProjSize',     None)),
            'recurrent_nonlinear_out_proj_function':                config.get('recurrentNonlinearOutputProjFunction', 'tanh'),
            'tertiary_output':                                      config.get('tertiaryOutput',                       'linear'),
            'internal_representation':                                      config.get('internal_representation',                       'transformer'),

            # RNN Parameters
            'recurrent_unit':                                       config.get('recurrentUnit',                        'LSTM'),
            'recurrent_peepholes':                      str_or_bool(config.get('recurrentPeepholes',                   True)), # LSTM
            'bidirectional':                            str_or_bool(config.get('bidirectional',                        False)), # pHO
            'include_recurrent_outputs_between_layers': str_or_bool(config.get('includeRecurrentOutputsBetweenLayers', True)), # HO
            'recurrent_layer_size':                     eval_if_str(config.get('recurrentSize',                        [20])),
            'higher_order_layers':                      str_or_bool(config.get('higherOrderLayers',                    False)),
            'residual_connections_every_n_layers':      int_or_none(config.get('residualConnectionsEveryNLayers',      0)), # HO
            'first_residual_connection_from_nth_layer': int_or_none(config.get('firstResidualConnectionFromNthLayer',  1)), # HO
            'recurrent_to_output_skip_connections':     str_or_bool(config.get('recurrentToOutputSkipConnections',     False)), # HO
            'input_to_recurrent_skip_connections':      str_or_bool(config.get('inputToRecurrentSkipConnections',      False)), # HO
            'all_to_recurrent_skip_connections':        str_or_bool(config.get('allToRecurrentSkipConnections',        False)), # HO
        
            # Transformer Parameters
            'transformer_layers':                     int_or_none(config.get('transformer_layers',                  6)),
            'transformer_heads':                     int_or_none(config.get('transformer_heads',                  8)),
            'transformer_ff_dims':                     int_or_none(config.get('transformer_ff_dims',                  512)),
            'transformer_dense_input_dim':                     int_or_none(config.get('transformer_dense_input_dim',                  256)),
            'transformer_type':                                       config.get('transformer_type',                        'vanilla'),
            'act_max_steps':                     int_or_none(config.get('act_max_steps',                  10)),
            'act_threshold':      float(config.get('act_threshold',  0.5)),
            'transition_function':                                       config.get('transition_function',                        'feed_forward'),
            'seperable_kernel_size':                     int_or_none(config.get('seperable_kernel_size',                  3)),
            'include_pos_encodings':        str_or_bool(config.get('include_pos_encodings',        False)), # HO
            

        }

        self.regularization = {
            'recurrent_input_keep_probability':           eval_if_str(config.get('recurInKeepProb',                    1.0)),
            'recurrent_output_keep_probability':          eval_if_str(config.get('recurOutKeepProb',                   1.0)),
            'recurrent_keep_probability':                 eval_if_str(config.get('recurKeepProb',                      1.0)),
            'recurrent_state_zonein_probability':         eval_if_str(config.get('recurStateZoneinProb',               1.0)),
            'recurrent_memory_zonein_probability':        eval_if_str(config.get('recurMemoryZoneinProb',              1.0)),
            'alphabet_keep_probability':                  eval_if_str(config.get('alphabetKeepProb',                   1.0)), # pHO
            'alphabet_normalization':                     str_or_none(config.get('alphabetNormalization',              None)), # pHO
            'recurrent_nonlinear_out_proj_normalization': str_or_none(config.get('recurNonlinearOutProjNormalization', None)),
            'recurrent_layer_normalization':              str_or_bool(config.get('recurLayerNormalization',            False)), # LNLSTM
            'recurrent_variational_dropout':              str_or_bool(config.get('recurVariationalDropout',            False)),
            'transformer_keep_prob':                     eval_if_str(config.get('transformer_keep_prob',                  1.0)),
        }

        self.loss = {
            'include':                       str_or_bool(config.get('includeLoss',                 True)), # whether to perform loss operation
            'tertiary_weight':                     float(config.get('tertiaryWeight',              1.0)), # weights for tertiary loss. Should be > 0.
            'tertiary_normalization':                    config.get('tertiaryNormalization',       'zeroth'), # influences the loss factor
            'batch_dependent_normalization': str_or_bool(config.get('batchDependentNormalization', True)), # (???)
            'atoms':                                     config.get('lossAtoms',                   'c_alpha') # which atoms to calc 3D structure
        }

class RunConfig(Config):
    """Contains parameters for an entire run.
    
    The run consists of possibly multiple models. (???)
    """

    def _create_config(self, config):
        self.names = {
            'run':                      config.get('runName'),
            'dataset':                  config.get('datasetName'),
            'alphabet':                 config.get('alphabetName', None) # Alphabet name
        }

        self.io = {
            'full_training_glob':       config.get('fullTrainingGlob',     '*'), # path
            'sample_training_glob':     config.get('sampleTrainingGlob',   '*'), # path
            'full_validation_glob':     config.get('fullValidationGlob',   '*'), # path
            'sample_validation_glob':   config.get('sampleValidationGlob', '*'), # path
            'full_testing_glob':        config.get('fullTestingGlob',      '*'), # path
            'sample_testing_glob':      config.get('sampleTestingGlob',    '*'), # path
            'evaluation_frequency': int(config.get('evaluationFrequency',  10)),
            'prediction_frequency': int(config.get('predictionFrequency',  100)),
            'checkpoint_frequency': int(config.get('checkpointFrequency',  10000))
        }

        self.computing = {
            'training_device':   config.get('trainingDevice',   'GPU'), # where to put training operations
            'evaluation_device': config.get('evaluationDevice', 'GPU') # where to put evaluation operations
        }

        # validationMilestone:
        # 'Milestone that the model must achieve or it will be restarted. ' \
        #       + 'Milestones must be of the form step:loss. ' \
        #       + 'Multiple milestones can be set.'
        self.optimization = {
            'validation_milestone': eval_if_str(config.get('validationMilestone', {})), 
            'validation_reference':             config.get('validationReference', 'weighted') # '[un]weighted', used for milestones, curricula, and predictions
        } 

        self.queueing = {
            'training_file_queue_capacity':    int(config.get('trainingFileQueueCapacity',    1000)),
            'evaluation_file_queue_capacity':  int(config.get('evaluationFileQueueCapacity',  10)),
            'training_batch_queue_capacity':   int(config.get('trainingBatchQueueCapacity',   10000)),
            'evaluation_batch_queue_capacity': int(config.get('evaluationBatchQueueCapacity', 300)),
            'training_min_after_dequeue':      int(config.get('trainingMinAfterDequeue',      500)),
            'evaluation_min_after_dequeue':    int(config.get('evaluationMinAfterDequeue',    10)),
            'training_shuffle':        str_or_bool(config.get('trainingShuffle',              True)),
            'evaluation_shuffle':      str_or_bool(config.get('evaluationShuffle',            False))
        }

        self.evaluation = {
            'num_training_samples':                  int(config.get('num_training_samples',                    98)), # train batch size
            'num_validation_samples':                int(config.get('numValidationSamples',                  100)), # validation batch size
            'num_testing_samples':                   int(config.get('numTestingSamples',                     100)), # testing batch size
            'num_training_invocations':              int(config.get('numTrainingInvocations',                1)),     # evaluation (! actual training)
            'num_validation_invocations':            int(config.get('numValidationInvocations',              1)),
            'num_testing_invocations':               int(config.get('numTestingInvocations',                 1)),
            'include_weighted_training':     str_or_bool(config.get('includeWeightedTraining',               False)),  # introduce weights TODO: this causes the training to stop early
            'include_weighted_validation':   str_or_bool(config.get('includeWeightedValidation',             False)),  # introduce weights
            'include_weighted_testing':      str_or_bool(config.get('includeWeightedTesting',                False)),  # introduce weights
            'include_unweighted_training':   str_or_bool(config.get('includeUnweightedTraining',             False)),  
            'include_unweighted_validation': str_or_bool(config.get('includeUnweightedValidation',           False)), 
            'include_unweighted_testing':    str_or_bool(config.get('includeUnweightedTesting',              False)),
            'include_diagnostics':           str_or_bool(config.get('includeDiagnostics',                    True))  # whether to run diagnostics TODO: diagnostics doesn't work for UT because None gradients are introduced
        }

        self.loss = {
            'training_tertiary_normalization':          config.get('trainingTertiaryNormalization',         'first'), # not even used ???
            'evaluation_tertiary_normalization':        config.get('evaluationTertiaryNormalization',       'first'), # not even used ???
            'training_batch_dependent_normalization':   config.get('trainingBatchDependentNormalization',   True), # not even used ???
            'evaluation_batch_dependent_normalization': config.get('evaluationBatchDependentNormalization', True) # not even used ???
        }
