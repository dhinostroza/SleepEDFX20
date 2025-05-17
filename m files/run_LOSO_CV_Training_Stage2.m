% run_LOSO_CV_Training_Stage2.m
% Loads pre-trained CNNs, extracts features, trains BiLSTM part sequentially.
% *** Passes cell array of sequences directly to trainNetwork. ***

clear;
clc; close all; tic;

fprintf('--- Stage 2: LOSO CV Training for BiLSTM Part (Sequential) ---\n');

% =========================================================================
% Configuration & Load Previous Results
% =========================================================================
fprintf('Loading Grouped Data and Stage 1 Results...\n');
processed_data_dir = fullfile(pwd, 'processed_data');
cnn_output_dir = fullfile(pwd, 'trained_cnn_folds');
lstm_output_dir = fullfile(pwd, 'trained_lstm_folds');
if ~exist(lstm_output_dir, 'dir'), mkdir(lstm_output_dir); end
stage1_results_filename_base = 'Stage1_CNN_Training_Results_';
processed_filename = 'SleepEDFX_SC40_processed_parallel.mat';

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
clear loaded_data epoch_subject_ids subject_epoch_counts subject_ids_list num_recordings rec_info psg_name base_subject_id end_epoch_idx num_epochs_this_rec current_epoch_idx subject_indices valid_field_name subj_id i;
fprintf('Data grouping complete.\n');

results_cnn_filename = sprintf('%s%s_Sequential.mat', stage1_results_filename_base, processed_filename);
results_cnn_filepath = fullfile(processed_data_dir, results_cnn_filename);
if ~exist(results_cnn_filepath, 'file'), error('Stage 1 results file not found: %s', results_cnn_filepath); end
fprintf('Loading Stage 1 results summary from: %s\n', results_cnn_filepath);
load(results_cnn_filepath, 'results_cnn');
if ~exist('results_cnn', 'var'), error('Variable `results_cnn` not found in Stage 1 results file.'); end

% =========================================================================
% Define LSTM Network Part
% =========================================================================
fprintf('Defining LSTM network part...\n');
numFeatures = 64; numHiddenUnitsLSTM = 128; numClasses = 5;
layersLSTM = [sequenceInputLayer(numFeatures, 'Name', 'seq_input', 'Normalization', 'none'), bilstmLayer(numHiddenUnitsLSTM, 'OutputMode', 'sequence', 'Name', 'bilstm1'), bilstmLayer(numHiddenUnitsLSTM, 'OutputMode', 'last', 'Name', 'bilstm2'), fullyConnectedLayer(numClasses, 'Name', 'main_fc'), softmaxLayer('Name', 'main_softmax'), classificationLayer('Name', 'main_output')];
lgraphLSTM = layerGraph(layersLSTM); fprintf('LSTM Network Part Defined.\n');
analyzeNetwork(lgraphLSTM);

% =========================================================================
% Training Setup for Stage 2
% =========================================================================
results_stage2 = struct(); sequenceLength = 10;
optionsLSTM = trainingOptions('adam', 'InitialLearnRate', 0.0005, 'MaxEpochs', 40, 'MiniBatchSize', 64, 'Shuffle', 'every-epoch', 'Plots', 'training-progress', 'Verbose', true, 'VerboseFrequency', 20, 'GradientThreshold', 1, 'ExecutionEnvironment', 'auto');
fprintf('\n--- Checking Execution Environment for LSTM Training ---\n'); try gpuCount = gpuDeviceCount("available"); catch, gpuCount = 0; end; trainEnv = optionsLSTM.ExecutionEnvironment; fprintf('Training Option "ExecutionEnvironment" set to: %s\n', trainEnv);
if gpuCount > 0, fprintf('Compatible GPU(s) detected (%d device(s)).\n', gpuCount); if strcmpi(trainEnv, 'auto'), fprintf('Training will likely utilize the GPU.\n'); elseif strcmpi(trainEnv, 'gpu'), fprintf('Training will attempt to use the GPU.\n'); elseif strcmpi(trainEnv, 'cpu'), fprintf('Training is explicitly set to use the CPU.\n'); end
else, fprintf('No compatible (NVIDIA CUDA-enabled) GPU detected by MATLAB.\n'); if strcmpi(trainEnv, 'auto'), fprintf('Training will proceed on the CPU.\n'); elseif strcmpi(trainEnv, 'gpu'), fprintf('WARNING: ExecutionEnvironment set to ''gpu'', but no compatible GPU detected. Training will run on the CPU instead.\n'); optionsLSTM.ExecutionEnvironment = 'cpu'; fprintf('         Automatically changed ExecutionEnvironment to ''cpu''.\n'); elseif strcmpi(trainEnv, 'cpu'), fprintf('Training will proceed on the CPU as specified.\n'); end; end
fprintf('---------------------------------------------------------\n');

% =========================================================================
% LOSO CV Loop for LSTM Training (Sequential)
% =========================================================================
fprintf('\n--- Starting Sequential LOSO CV Loop for LSTM Training ---\n');
allFoldPredictions_S2 = cell(num_unique_subjects, 1); allFoldTrueLabels_S2 = cell(num_unique_subjects, 1);
results_stage2 = repmat(struct('SubjectID',[],'TrueLabels',[], 'PredictedLabels', [], 'ConfMat',[], 'Accuracy', [], 'TrainInfoLSTM',[], 'SavedLSTMNetFile',[], 'Error',[]), num_unique_subjects, 1);

for k = 1:num_unique_subjects
    test_subject_id = unique_subject_ids(k);
    test_subject_fieldname = matlab.lang.makeValidName(test_subject_id);
    fprintf('\n===== Starting Stage 2 - Fold %d/%d: Testing on Subject %s =====\n', k, num_unique_subjects, test_subject_id);

    lstm_fold_filename = fullfile(lstm_output_dir, sprintf('lstm_fold_%d_subject_%s.mat', k, test_subject_id));
    if exist(lstm_fold_filename, 'file')
        fprintf(' Fold %d: LSTM output file already exists (%s). Loading results and skipping training/evaluation.\n', k, lstm_fold_filename);
        fold_data = load(lstm_fold_filename, 'netLSTM_Fold', 'trainInfoLSTM', 'test_subject_id', 'testLabelsCategorical_fold', 'predictionsCategorical_fold');
        results_stage2(k).SubjectID = fold_data.test_subject_id; results_stage2(k).TrueLabels = fold_data.testLabelsCategorical_fold; results_stage2(k).PredictedLabels = fold_data.predictionsCategorical_fold;
        results_stage2(k).ConfMat = confusionmat(fold_data.testLabelsCategorical_fold, fold_data.predictionsCategorical_fold); results_stage2(k).Accuracy = sum(fold_data.predictionsCategorical_fold == fold_data.testLabelsCategorical_fold) / numel(fold_data.testLabelsCategorical_fold);
        results_stage2(k).TrainInfoLSTM = fold_data.trainInfoLSTM; results_stage2(k).SavedLSTMNetFile = lstm_fold_filename; results_stage2(k).Error = 'Skipped - File Exists';
        allFoldTrueLabels_S2{k} = fold_data.testLabelsCategorical_fold; allFoldPredictions_S2{k} = fold_data.predictionsCategorical_fold;
        fprintf(' Fold %d loaded. Accuracy: %.4f\n', k, results_stage2(k).Accuracy); continue;
    end

    if k > numel(results_cnn) || (isfield(results_cnn(k), 'Error') && ~isempty(results_cnn(k).Error) && ~strcmp(results_cnn(k).Error,'Skipped - File Exists')), fprintf(' Fold %d: Skipping Stage 2 because Stage 1 failed or results missing.\n', k); results_stage2(k).SubjectID = test_subject_id; results_stage2(k).Error = 'Skipped - Stage 1 Failed'; continue; end
    if ~isfield(results_cnn(k), 'SavedNetFile') || isempty(results_cnn(k).SavedNetFile) || ~exist(results_cnn(k).SavedNetFile, 'file'), fprintf(' Fold %d: Skipping Stage 2 because Stage 1 network file not found: %s\n', k, results_cnn(k).SavedNetFile); results_stage2(k).SubjectID = test_subject_id; results_stage2(k).Error = 'Skipped - Stage 1 Net Missing'; continue; end

    fprintf(' Fold %d: Loading pre-trained CNN from %s...\n', k, results_cnn(k).SavedNetFile);
    cnnData = load(results_cnn(k).SavedNetFile);
    if ~isfield(cnnData, 'netCNN_Fold'), fprintf(' Fold %d: ERROR - Variable netCNN_Fold not found. Skipping fold.\n', k); results_stage2(k).SubjectID = test_subject_id; results_stage2(k).Error = 'Stage 1 Net Variable Missing'; continue; end
    netCNN_Fold = cnnData.netCNN_Fold; clear cnnData;

    fprintf(' Fold %d: Preparing training and test data...\n', k);
    if ~isfield(grouped_data, test_subject_fieldname), warning('Test subject %s not found. Skipping fold.', test_subject_id); results_stage2(k).Error='Test subject not found'; continue; end
    testSpectrogramsEpochs = grouped_data.(test_subject_fieldname).Spectrograms; testLabelsEpochs = grouped_data.(test_subject_fieldname).Labels;
    num_train_subjects = num_unique_subjects - 1; trainSpectrogramsList = cell(num_train_subjects, 1); trainLabelsList = cell(num_train_subjects, 1); train_idx = 0;
    for j = 1:num_unique_subjects
        if k == j, continue; end
        train_subj_id = unique_subject_ids(j); train_subj_fieldname = matlab.lang.makeValidName(train_subj_id);
        if isfield(grouped_data, train_subj_fieldname), train_idx = train_idx + 1; trainSpectrogramsList{train_idx} = grouped_data.(train_subj_fieldname).Spectrograms; trainLabelsList{train_idx} = grouped_data.(train_subj_fieldname).Labels; end
    end
    if train_idx ~= num_train_subjects, fprintf('Fold %d: WARNING - Expected %d training subjects, found %d.\n', k, num_train_subjects, train_idx); trainSpectrogramsList = trainSpectrogramsList(1:train_idx); trainLabelsList = trainLabelsList(1:train_idx); end
    if isempty(trainSpectrogramsList), fprintf('Fold %d: No training data found. Skipping fold.\n', k); results_stage2(k).SubjectID = test_subject_id; results_stage2(k).Error='No training data'; continue; end
    trainSpectrogramsEpochs = cat(1, trainSpectrogramsList{:}); trainLabelsEpochs = vertcat(trainLabelsList{:});
    clear trainSpectrogramsList trainLabelsList;

    featureLayer = 'gap';
    fprintf(' Fold %d: Extracting features from layer ''%s''...\n', k, featureLayer);
    fprintf('  Extracting training features (%d epochs)...\n', size(trainSpectrogramsEpochs, 1));
    miniBatchSizeActivations = optionsLSTM.MiniBatchSize * 4;
    trainFeatures = extractFeaturesInBatches(netCNN_Fold, trainSpectrogramsEpochs, featureLayer, miniBatchSizeActivations);
    fprintf('  Extracting testing features (%d epochs)...\n', size(testSpectrogramsEpochs, 1));
    testFeatures = extractFeaturesInBatches(netCNN_Fold, testSpectrogramsEpochs, featureLayer, miniBatchSizeActivations);
    clear trainSpectrogramsEpochs testSpectrogramsEpochs netCNN_Fold;
    fprintf(' Fold %d: Feature extraction complete.\n', k);

    fprintf(' Fold %d: Converting features to sequences (length %d)...\n', k, sequenceLength);
    [trainFeatureSequences, trainSequenceLabels] = createFeatureSequences(trainFeatures, trainLabelsEpochs, sequenceLength);
    [testFeatureSequences, testSequenceLabels] = createFeatureSequences(testFeatures, testLabelsEpochs, sequenceLength);
    if isempty(trainFeatureSequences) || isempty(testFeatureSequences), warning('Fold %d: Could not create feature sequences. Skipping fold.', k); results_stage2(k).Error='Feature sequence creation failed'; continue; end
    fprintf('  -> Created %d training sequences, %d testing sequences.\n', numel(trainFeatureSequences), numel(testFeatureSequences));
    clear trainFeatures testFeatures trainLabelsEpochs testLabelsEpochs;

    % --- Prepare data for trainNetwork (no explicit datastore needed for cell array input) ---
    trainLabelsCategorical = categorical(trainSequenceLabels);
    testLabelsCategorical_fold = categorical(testSequenceLabels);
    fprintf(' Fold %d: Prepared data for direct input to trainNetwork.\n', k);

    fprintf(' Fold %d: Training LSTM network part...\n', k);
    try
        current_lstm_lgraph = lgraphLSTM;
        fold_lstm_options = optionsLSTM;
        fold_lstm_options.Plots = 'training-progress'; fold_lstm_options.Verbose = true;

        % *** Pass cell array of sequences and categorical labels directly ***
        [netLSTM_Fold, trainInfoLSTM] = trainNetwork(trainFeatureSequences, trainLabelsCategorical, current_lstm_lgraph, fold_lstm_options);

        fprintf(' Fold %d: Evaluating LSTM network on test sequences...\n', k);
        predictionsCategorical_fold = classify(netLSTM_Fold, testFeatureSequences, 'MiniBatchSize', fold_lstm_options.MiniBatchSize, 'SequenceLength', 'longest');

        fprintf(' Fold %d: Saving trained LSTM and results to %s\n', k, lstm_fold_filename);
        save(lstm_fold_filename, 'netLSTM_Fold', 'trainInfoLSTM', 'test_subject_id', 'testLabelsCategorical_fold', 'predictionsCategorical_fold', '-v7.3');

        results_stage2(k).SubjectID = test_subject_id; results_stage2(k).TrueLabels = testLabelsCategorical_fold;
        results_stage2(k).PredictedLabels = predictionsCategorical_fold; results_stage2(k).ConfMat = confusionmat(testLabelsCategorical_fold, predictionsCategorical_fold);
        results_stage2(k).Accuracy = sum(predictionsCategorical_fold == testLabelsCategorical_fold) / numel(testLabelsCategorical_fold);
        results_stage2(k).TrainInfoLSTM = trainInfoLSTM; results_stage2(k).SavedLSTMNetFile = lstm_fold_filename;
        fprintf(' Fold %d LSTM Accuracy: %.4f\n', k, results_stage2(k).Accuracy);
        allFoldPredictions_S2{k} = predictionsCategorical_fold; allFoldTrueLabels_S2{k} = testLabelsCategorical_fold;

    catch ME_train_lstm
        fprintf('\n!!! Fold %d: ERROR during LSTM training/evaluation (Subject %s) !!!\n', k, test_subject_id);
        fprintf('   Error Message: %s\n', ME_train_lstm.message);
        if ~isempty(ME_train_lstm.stack), fprintf('   Error occurred in file: %s, line: %d\n', ME_train_lstm.stack(1).file, ME_train_lstm.stack(1).line); end
        results_stage2(k).SubjectID = test_subject_id; results_stage2(k).Error = ME_train_lstm;
    end

    clear trainFeatureSequences trainSequenceLabels testFeatureSequences testSequenceLabels netLSTM_Fold trainInfoLSTM predictionsCategorical_fold testLabelsCategorical_fold;

end % End FOR loop

% =========================================================================
% Aggregate and Report Final Results for Stage 2
% =========================================================================
% ... (Keep the results aggregation and saving logic as before) ...
fprintf('\n--- Stage 2 Cross-Validation Finished ---\n');
valid_fold_indices_s2 = find(arrayfun(@(x) isempty(x.Error) || (ischar(x.Error) && contains(x.Error, 'Skipped')), results_stage2));
if isempty(valid_fold_indices_s2)
     fprintf('No successful folds in Stage 2 to aggregate results from.\n');
else
    valid_true_labels_cells = allFoldTrueLabels_S2(valid_fold_indices_s2);
    valid_predictions_cells = allFoldPredictions_S2(valid_fold_indices_s2);
    if all(cellfun('isempty', valid_true_labels_cells)) || all(cellfun('isempty', valid_predictions_cells))
         fprintf('No valid predictions collected from successful/loaded folds.\n');
    else
        allTrue_S2 = vertcat(valid_true_labels_cells{:});
        allPred_S2 = vertcat(valid_predictions_cells{:});
        if isempty(allTrue_S2) || isempty(allPred_S2), fprintf('Concatenated predictions or labels are empty.\n');
        else
            fprintf('Aggregating results across %d successful/loaded folds...\n', numel(valid_fold_indices_s2));
            overallConfMat_S2 = confusionmat(allTrue_S2, allPred_S2); disp('Overall Confusion Matrix (Stage 2):'); disp(overallConfMat_S2);
            overallAccuracy_S2 = sum(diag(overallConfMat_S2)) / sum(overallConfMat_S2(:)); fprintf('Overall Accuracy (Stage 2): %.4f\n', overallAccuracy_S2);
            numClassesActual = size(overallConfMat_S2, 1); classNames = categories(allTrue_S2);
            if isempty(classNames) || numel(classNames) ~= numClassesActual, classNames = string(0:numClassesActual-1); end
            precision = zeros(numClassesActual, 1); recall = zeros(numClassesActual, 1); f1 = zeros(numClassesActual, 1);
            for c = 1:numClassesActual
                tp = overallConfMat_S2(c, c); fp = sum(overallConfMat_S2(:, c)) - tp; fn = sum(overallConfMat_S2(c, :)) - tp;
                precision(c) = tp / (tp + fp); recall(c) = tp / (tp + fn); f1(c) = 2 * (precision(c) * recall(c)) / (precision(c) + recall(c));
                if isnan(precision(c)), precision(c) = 0; end; if isnan(recall(c)), recall(c) = 0; end; if isnan(f1(c)), f1(c) = 0; end
                fprintf('Class %s (Label %s): Precision=%.4f, Recall=%.4f, F1=%.4f\n', classNames{c}, classNames{c}, precision(c), recall(c), f1(c));
            end
            macroF1_S2 = mean(f1); fprintf('Macro-F1 Score (MF1) (Stage 2): %.4f\n', macroF1_S2);
            kappaValue_S2 = cohensKappa(overallConfMat_S2); fprintf('Cohen''s Kappa (Stage 2): %.4f\n', kappaValue_S2);
            resultsOverall_S2.ConfMat = overallConfMat_S2; resultsOverall_S2.Accuracy = overallAccuracy_S2; resultsOverall_S2.Precision = precision; resultsOverall_S2.Recall = recall; resultsOverall_S2.F1 = f1; resultsOverall_S2.MacroF1 = macroF1_S2; resultsOverall_S2.Kappa = kappaValue_S2; resultsOverall_S2.ClassNames = classNames;
            results_s2_filename = sprintf('Stage2_LSTM_Training_Results_%s.mat', processed_filename);
            results_s2_filepath = fullfile(processed_data_dir, results_s2_filename);
            fprintf('Saving detailed fold results and overall metrics for Stage 2 to: %s\n', results_s2_filepath);
            save(results_s2_filepath, 'results_stage2', 'resultsOverall_S2');
        end
    end
end
toc;
fprintf('\n--- Stage 2 LSTM Training Script Finished (Sequential) ---\n');

% =========================================================================
% Helper Functions (extractFeaturesInBatches, createFeatureSequences, cohensKappa)
% =========================================================================
% ... (Keep the helper functions as defined in the previous message) ...
function features = extractFeaturesInBatches(net, dataSpectrograms, layerName, miniBatchSize)
    numSamples = size(dataSpectrograms, 1); numBatches = ceil(numSamples / miniBatchSize); features = [];
    fprintf('    Starting feature extraction in %d batches...\n', numBatches);
    for i = 1:numBatches
        startIdx = (i-1) * miniBatchSize + 1; endIdx = min(i * miniBatchSize, numSamples);
        batchData = permute(dataSpectrograms(startIdx:endIdx, :, :, :), [2 3 4 1]);
        batchFeatures = activations(net, batchData, layerName, 'OutputAs', 'rows', 'ExecutionEnvironment', 'auto');
        if isempty(features), numFeatures = size(batchFeatures, 2); features = zeros(numSamples, numFeatures, 'like', batchFeatures); end
        features(startIdx:endIdx, :) = batchFeatures;
    end
end

function [sequences, sequenceLabels] = createFeatureSequences(epochFeatures, epochLabels, sequenceLength)
    numEpochs = size(epochFeatures, 1);
    if numEpochs < sequenceLength, sequences = {}; sequenceLabels = []; warning('Not enough epochs (%d) for sequence length %d.', numEpochs, sequenceLength); return; end
    numSequences = numEpochs - sequenceLength + 1; sequences = cell(numSequences, 1); sequenceLabels = zeros(numSequences, 1, class(epochLabels));
    for i = 1:numSequences
        idx_end = i + sequenceLength - 1;
        sequences{i} = epochFeatures(i:idx_end, :)'; % [NumFeatures x sequenceLength]
        sequenceLabels(i) = epochLabels(idx_end);
    end
end

function kappaValue = cohensKappa(confMat)
    N = sum(confMat(:)); if N == 0, kappaValue = 0; return; end
    po = sum(diag(confMat)) / N; sumRows = sum(confMat, 2); sumCols = sum(confMat, 1);
    pe = sum(sumRows .* sumCols') / (N * N);
    if abs(1 - pe) < eps, kappaValue = 1; else, kappaValue = (po - pe) / (1 - pe); end
    if isnan(kappaValue), kappaValue = 0; end
end