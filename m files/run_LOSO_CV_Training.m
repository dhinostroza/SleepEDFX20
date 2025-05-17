% run_LOSO_CV_Training_Stage1_Sequential.m
% Trains CNN feature extractor using LOSO CV and arrayDatastore.
% *** Uses transform datastore for input. Fixes output_dir error. ***
% Runs SEQUENTIALLY. Saves each fold's network periodically.

clear;
clc; close all; tic;

fprintf('--- Stage 1: LOSO CV Training for CNN Feature Extractor (Sequential) ---\n');

% =========================================================================
% Load Data and Group by Subject
% =========================================================================
fprintf('Loading and Grouping Data...\n');
processed_data_dir = fullfile(pwd, 'processed_data');
% *** CHOOSE WHICH DATASET TO LOAD ***
% processed_filename = 'SleepEDFX8_processed_parallel.mat';
processed_filename = 'SleepEDFX_SC40_processed_parallel.mat'; % Using the 40 recordings
% ************************************
processed_filepath = fullfile(processed_data_dir, processed_filename);
if ~exist(processed_filepath, 'file'), error('Processed data file not found: %s', processed_filepath); end
loaded_data = load(processed_filepath, 'all_spectrograms', 'all_labels', 'processed_subject_info'); fprintf('Data loaded successfully.\n');
if ~isfield(loaded_data,'all_spectrograms') || ~isfield(loaded_data,'all_labels') || ~isfield(loaded_data,'processed_subject_info'), error('Loaded .mat file does not contain the expected variables.'); end
fprintf('Extracting subject IDs and grouping data...\n');
num_recordings = numel(loaded_data.processed_subject_info); epoch_subject_ids = strings(size(loaded_data.all_labels)); current_epoch_idx = 1;
subject_epoch_counts = zeros(num_recordings, 1); subject_ids_list = cell(num_recordings, 1);
for i = 1:num_recordings, rec_info = loaded_data.processed_subject_info{i}; if isempty(rec_info) || ~isfield(rec_info, 'psg_file') || ~isfield(rec_info, 'num_valid_epochs'), continue; end; num_epochs_this_rec = rec_info.num_valid_epochs; subject_epoch_counts(i) = num_epochs_this_rec; [~, psg_name, ~] = fileparts(rec_info.psg_file); base_subject_id = psg_name(1:5); subject_ids_list{i} = base_subject_id; end_epoch_idx = current_epoch_idx + num_epochs_this_rec - 1; if end_epoch_idx > length(epoch_subject_ids), end_epoch_idx = length(epoch_subject_ids); num_epochs_this_rec = max(0, end_epoch_idx - current_epoch_idx + 1); subject_epoch_counts(i) = num_epochs_this_rec; end; if num_epochs_this_rec > 0, epoch_subject_ids(current_epoch_idx : end_epoch_idx) = base_subject_id; end; current_epoch_idx = end_epoch_idx + 1; end
if sum(subject_epoch_counts) ~= size(loaded_data.all_spectrograms, 1), warning('Total epoch count mismatch!'); end
unique_subject_ids = unique(epoch_subject_ids(strlength(epoch_subject_ids) > 0)); num_unique_subjects = numel(unique_subject_ids); fprintf('Found %d unique subject IDs.\n', num_unique_subjects);
grouped_data = struct();
for i = 1:num_unique_subjects, subj_id = unique_subject_ids(i); subject_indices = find(epoch_subject_ids == subj_id); if isempty(subject_indices), continue; end; valid_field_name = matlab.lang.makeValidName(subj_id); grouped_data.(valid_field_name).Spectrograms = loaded_data.all_spectrograms(subject_indices, :, :, :); grouped_data.(valid_field_name).Labels = loaded_data.all_labels(subject_indices); grouped_data.(valid_field_name).SubjectID = subj_id; grouped_data.(valid_field_name).NumEpochs = length(subject_indices); end
clear loaded_data epoch_subject_ids subject_epoch_counts subject_ids_list num_recordings rec_info psg_name base_subject_id end_epoch_idx num_epochs_this_rec current_epoch_idx subject_indices valid_field_name subj_id i; % Clear intermediate vars
fprintf('Data grouping complete.\n');

% =========================================================================
% Define CNN Feature Extractor Model
% =========================================================================
if exist('defineEEGSNet.m','file')
    fprintf('Defining EEGSNet architectures...\n');
    lgraph_full = defineEEGSNet();
    fprintf('Modifying graph for CNN feature extractor pre-training...\n');
    layersToRemove = {'flatten_gap','bilstm1','bilstm2','main_fc','main_softmax','main_output'};
    lgraph_cnn_feat = removeLayers(lgraph_full, layersToRemove);
    numClasses = 5;
    newLayers = [fullyConnectedLayer(numClasses, 'Name', 'pretrain_fc'), softmaxLayer('Name', 'pretrain_softmax'), classificationLayer('Name', 'pretrain_output')];
    lgraph_cnn_feat = addLayers(lgraph_cnn_feat, newLayers);
    lgraph_cnn_feat = connectLayers(lgraph_cnn_feat, 'gap', 'pretrain_fc');
    fprintf('CNN Feature Extractor Graph Created.\n');
    analyzeNetwork(lgraph_cnn_feat);
else
    error('`defineEEGSNet.m` function not found.');
end

% =========================================================================
% Training Setup
% =========================================================================
results_cnn = struct();
cnn_output_dir = fullfile(pwd, 'trained_cnn_folds');
if ~exist(cnn_output_dir, 'dir'), mkdir(cnn_output_dir); end

options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'VerboseFrequency', 50, ...
    'ValidationData', [], ...
    'ExecutionEnvironment', 'auto');

fprintf('\n--- Checking Execution Environment ---\n'); try gpuCount = gpuDeviceCount("available"); catch, gpuCount = 0; end; trainEnv = options.ExecutionEnvironment; fprintf('Training Option "ExecutionEnvironment" set to: %s\n', trainEnv);
if gpuCount > 0, fprintf('Compatible GPU(s) detected (%d device(s)).\n', gpuCount); if strcmpi(trainEnv, 'auto'), fprintf('Training will likely utilize the GPU.\n'); elseif strcmpi(trainEnv, 'gpu'), fprintf('Training will attempt to use the GPU.\n'); elseif strcmpi(trainEnv, 'cpu'), fprintf('Training is explicitly set to use the CPU.\n'); end
else, fprintf('No compatible (NVIDIA CUDA-enabled) GPU detected by MATLAB.\n'); if strcmpi(trainEnv, 'auto'), fprintf('Training will proceed on the CPU.\n'); elseif strcmpi(trainEnv, 'gpu'), fprintf('WARNING: ExecutionEnvironment set to ''gpu'', but no compatible GPU detected. Training will run on the CPU instead.\n'); options.ExecutionEnvironment = 'cpu'; fprintf('         Automatically changed ExecutionEnvironment to ''cpu''.\n'); elseif strcmpi(trainEnv, 'cpu'), fprintf('Training will proceed on the CPU as specified.\n'); end; end
fprintf('---------------------------------------\n');

% =========================================================================
% LOSO CV Loop for CNN Training (Using standard FOR loop)
% =========================================================================
fprintf('\n--- Starting Sequential LOSO CV Loop for CNN Feature Extractor Training ---\n');
results_cnn = repmat(struct('SubjectID',[],'TrainInfo',[],'SavedNetFile',[],'Error',[]), num_unique_subjects, 1);

for k = 1:num_unique_subjects
    test_subject_id = unique_subject_ids(k);
    test_subject_fieldname = matlab.lang.makeValidName(test_subject_id);
    fprintf('\n===== Starting Fold %d/%d: Testing on Subject %s =====\n', k, num_unique_subjects, test_subject_id);

    cnn_fold_filename = fullfile(cnn_output_dir, sprintf('cnn_fold_%d_subject_%s.mat', k, test_subject_id));
    if exist(cnn_fold_filename, 'file')
        fprintf(' Fold %d: Output file already exists (%s). Skipping training.\n', k, cnn_fold_filename);
        load(cnn_fold_filename, 'test_subject_id'); % Load just the ID
        results_cnn(k).SubjectID = test_subject_id; results_cnn(k).SavedNetFile = cnn_fold_filename;
        results_cnn(k).TrainInfo = []; results_cnn(k).Error = 'Skipped - File Exists';
        continue;
    end

    fprintf(' Fold %d: Preparing training data...\n', k);
    num_train_subjects = num_unique_subjects - 1; trainSpectrogramsList = cell(num_train_subjects, 1); trainLabelsList = cell(num_train_subjects, 1); train_idx = 0;
    for j = 1:num_unique_subjects
        if k == j, continue; end
        train_subj_id = unique_subject_ids(j); train_subj_fieldname = matlab.lang.makeValidName(train_subj_id);
        if isfield(grouped_data, train_subj_fieldname), train_idx = train_idx + 1; trainSpectrogramsList{train_idx} = grouped_data.(train_subj_fieldname).Spectrograms; trainLabelsList{train_idx} = grouped_data.(train_subj_fieldname).Labels; end
    end
    if train_idx ~= num_train_subjects, fprintf('Fold %d: WARNING - Expected %d training subjects, found %d.\n', k, num_train_subjects, train_idx); trainSpectrogramsList = trainSpectrogramsList(1:train_idx); trainLabelsList = trainLabelsList(1:train_idx); end
    if isempty(trainSpectrogramsList), fprintf('Fold %d: No training data found. Skipping fold.\n', k); results_cnn(k).SubjectID = test_subject_id; results_cnn(k).Error='No training data'; continue; end
    trainSpectrogramsEpochs = cat(1, trainSpectrogramsList{:});
    trainLabelsEpochs = vertcat(trainLabelsList{:});

    % --- Create Datastores using transform ---
    trainDataCNN = permute(trainSpectrogramsEpochs, [2 3 4 1]); % HxWxCxN
    trainLabelsCategorical = categorical(trainLabelsEpochs);
    dsTrainLabels = arrayDatastore(trainLabelsCategorical);
    dsTrainInput = arrayDatastore(trainDataCNN, 'IterationDimension', 4); % Iterate over 4th dim (N)

    % *** FIX: Use transform datastore ***
    dsTrain = transform(combine(dsTrainInput, dsTrainLabels), @preprocessTrainData);
    fprintf(' Fold %d: Created transform datastore for %d training epochs.\n', k, size(trainDataCNN, 4));

    % --- Train the CNN Network ---
    fprintf(' Fold %d: Training CNN feature extractor...\n', k);
    try
        current_cnn_lgraph = lgraph_cnn_feat;
        fold_options = options;
        fold_options.Plots = 'training-progress';
        fold_options.Verbose = true;

        [netCNN_Fold, trainInfo] = trainNetwork(dsTrain, current_cnn_lgraph, fold_options);

        fprintf(' Fold %d: Saving trained CNN to %s\n', k, cnn_fold_filename);
        save(cnn_fold_filename, 'netCNN_Fold', 'trainInfo', 'test_subject_id', '-v7.3');

        results_cnn(k).SubjectID = test_subject_id;
        results_cnn(k).TrainInfo = trainInfo;
        results_cnn(k).SavedNetFile = cnn_fold_filename;
        fprintf(' Fold %d CNN training finished successfully.\n', k);

    catch ME_train
        fprintf('\n!!! Fold %d: ERROR during CNN training (Subject %s) !!!\n', k, test_subject_id);
        fprintf('   Error Message: %s\n', ME_train.message);
        if ~isempty(ME_train.stack), fprintf('   Error occurred in file: %s, line: %d\n', ME_train.stack(1).file, ME_train.stack(1).line); end
        results_cnn(k).SubjectID = test_subject_id;
        results_cnn(k).Error = ME_train;
    end

    clear trainSpectrogramsEpochs trainLabelsEpochs trainDataCNN trainLabelsCategorical ...
          dsTrain* netCNN_Fold trainInfo trainSpectrogramsList trainLabelsList;

end % End FOR loop

% =========================================================================
% Save Stage 1 Summary
% =========================================================================
% *** FIX: Define output_dir and processed_filename again before saving ***
output_dir = fullfile(pwd, 'processed_data');
processed_filename = 'SleepEDFX_SC40_processed_parallel.mat'; % Or the one you loaded
% *********************************************************************
results_cnn_filename = sprintf('Stage1_CNN_Training_Results_%s_Sequential.mat', processed_filename);
results_cnn_filepath = fullfile(output_dir, results_cnn_filename);
fprintf('\n--- Saving Stage 1 CNN Training Summary ---\n');
fprintf(' Summary saved to: %s\n', results_cnn_filepath);
save(results_cnn_filepath, 'results_cnn');

% Clean up temporary image directory base if it exists and is empty
temp_imds_base_dir = fullfile(pwd, 'temp_imds_data_par');
if exist(temp_imds_base_dir, 'dir') && numel(dir(temp_imds_base_dir)) == 2
    try rmdir(temp_imds_base_dir); catch; end
end

toc;
fprintf('\n--- Stage 1 CNN Training Script Finished (Sequential) ---\n');

% =========================================================================
% Helper Function for transform datastore
% =========================================================================
function dataOut = preprocessTrainData(data)
% Formats data read from combined arrayDatastore for trainNetwork
% Input 'data' is a 1x2 cell: {InputDataBatch, ResponseBatch}
% InputDataBatch is HxWxCxMiniBatchSize
% ResponseBatch is MiniBatchSize x 1 categorical
% Output 'dataOut' should be a 1x2 cell: {InputForNetwork, ResponseForNetwork}
    inputData = data{1};
    response = data{2};
    % For standard image input, return data as is in a cell
    dataOut = {inputData, response};
end