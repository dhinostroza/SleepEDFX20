% run_preprocess_sleepedfx_sc40.m
% Main script to preprocess the 40 SC recordings (20 subjects, 2 nights)
% from the Sleep-EDFX 'sleep-cassette' directory.
% NOTE: This differs from the paper's "78/153" count, which requires a specific list.
% Uses Parallel Computing Toolbox.

clear;
clc;
close all;

tic; % Start timer for the whole process

fprintf('--- Starting Preprocessing for Sleep-EDFX SC40 Recordings (Parallel) ---\n');
fprintf('*** NOTE: Processing all 40 SC recordings (20 subjects x 2 nights) from sleep-cassette. ***\n');
fprintf('*** This differs from the "78/153" count mentioned in Li et al. 2022. ***\n');

% =========================================================================
% Configuration
% =========================================================================
% !!! USER: SET THIS PATH CORRECTLY !!!
dataset_parent_path = '/Users/dhinostroza/Documents/MATLAB/2025-04-16 Practica EEG Deep Learning/Exposicion/Bases_datos/sleep-edf-database-expanded-1.0.0';
output_dir = fullfile(pwd, 'processed_data');
output_filename = 'SleepEDFX_SC40_processed_parallel.mat'; % Reflects 40 recordings processed

% --- Define Sleep-EDFX SC Subjects (20 subjects, nights 1 & 2) ---
subject_psg_suffixes = cell(40, 1); % 20 subjects * 2 nights
idx = 1;
for k = 0:19 % Subject index 00 to 19
    % Night 1
    subject_psg_suffixes{idx} = sprintf('SC4%02d1E0', k);
    idx = idx + 1;
    % Night 2 (Usually SC4xx2E0 in expanded dataset)
    subject_psg_suffixes{idx} = sprintf('SC4%02d2E0', k);
    idx = idx + 1;
end
num_subjects = length(subject_psg_suffixes); % This is now number of recordings
fprintf('Defined %d recordings for Sleep-EDFX SC40 set.\n', num_subjects);
disp('First few recordings defined:');
disp(subject_psg_suffixes(1:4));

% --- Define Shared Configuration Parameters (Same as before) ---
configParams = struct();
configParams.target_eeg_channel = 'EEG Fpz-Cz';
configParams.epoch_duration_sec = 30;
configParams.fs_target = 64;
configParams.max_freq = 32;
configParams.image_dpi = 100;
configParams.image_width_in = 0.8;
configParams.image_height_in = 1.0;
configParams.crop_rows = 14:90;
configParams.crop_cols = 11:71;
configParams.final_img_size_paper = [76, 60];
configParams.window_sec = 2;
configParams.overlap_perc = 0.5;
configParams.nfft = 256;
configParams.label_map = containers.Map(...
    {'Wake', 'N1', 'N2', 'N3', 'REM'}, {0, 1, 2, 3, 4});
configParams.temp_dir_base = fullfile(pwd, 'temp_spectrograms_subjects_par');

% --- Create directories ---
if ~exist(output_dir, 'dir'), mkdir(output_dir); fprintf('Created output directory: %s\n', output_dir); end
if ~exist(configParams.temp_dir_base, 'dir'), mkdir(configParams.temp_dir_base); end

% =========================================================================
% START PARALLEL POOL
% =========================================================================
fprintf('\n--- Starting Parallel Pool ---\n');
poolobj = gcp('nocreate');
if isempty(poolobj), parpool();
else, fprintf('Parallel pool already running with %d workers.\n', poolobj.NumWorkers); end

% =========================================================================
% Preprocessing Loop (using parfor over recordings)
% =========================================================================
results_spectrograms = cell(num_subjects, 1);
results_labels = cell(num_subjects, 1);
results_info = cell(num_subjects, 1);
results_success = false(num_subjects, 1);

fprintf('\n--- Starting Parallel Recording Loop (%d recordings) ---\n', num_subjects);

parfor i = 1:num_subjects
    psg_suffix = subject_psg_suffixes{i};
    subject_folder = 'sleep-cassette'; % All SC subjects are here
    current_dataset_path = fullfile(dataset_parent_path, subject_folder);

    psg_filename = [psg_suffix, '-PSG.edf'];
    psg_filepath = fullfile(current_dataset_path, psg_filename);

    % Dynamically find the corresponding Hypnogram file using '*' wildcard
    hyp_pattern = [psg_suffix(1:end-2), '*-Hypnogram.edf']; % Pattern like SC4xx*-Hypnogram.edf
    hyp_files_found = dir(fullfile(current_dataset_path, hyp_pattern));

    hyp_filepath = ''; % Initialize empty
    if isempty(hyp_files_found)
        fprintf('Worker processing Recording %d/%d: %s - ERROR: No hypnogram file found matching pattern "%s"\n', i, num_subjects, psg_suffix, hyp_pattern);
        results_success(i) = false;
        continue; % Skip to the next iteration
    elseif length(hyp_files_found) > 1
        fprintf('Worker processing Recording %d/%d: %s - WARNING: Multiple hypnogram files found matching pattern "%s". Using first one: %s\n', i, num_subjects, psg_suffix, hyp_pattern, hyp_files_found(1).name);
        hyp_filepath = fullfile(current_dataset_path, hyp_files_found(1).name);
    else
        hyp_filepath = fullfile(current_dataset_path, hyp_files_found(1).name);
        fprintf('Worker processing Recording %d/%d: %s (using hypnogram: %s)\n', i, num_subjects, psg_suffix, hyp_files_found(1).name);
    end

    % Call the preprocessing function
    [subj_spectrograms, subj_labels, subj_info, success] = preprocessSubjectEDFX(psg_filepath, hyp_filepath, configParams);

    % Store results for this iteration
    results_spectrograms{i} = subj_spectrograms;
    results_labels{i} = subj_labels;
    results_info{i} = subj_info;
    results_success(i) = success;

    if success
         fprintf('Worker finished Recording %s successfully.\n', psg_suffix);
    else
         fprintf('Worker FAILED Recording %s.\n', psg_suffix);
    end

end % End parfor loop

fprintf('\n--- Parallel Recording Loop Finished ---\n');

% =========================================================================
% Aggregate and Save Results (after parfor)
% =========================================================================
fprintf('\n--- Aggregating results from workers ---\n');

successful_indices = find(results_success);
recordings_processed_count = length(successful_indices); % Changed variable name
failed_recordings_indices = find(~results_success);
failed_recordings = {}; % Changed variable name
if ~isempty(failed_recordings_indices)
    for k = 1:length(failed_recordings_indices)
        failed_recordings{end+1} = subject_psg_suffixes{failed_recordings_indices(k)};
    end
end

fprintf('Successfully processed %d out of %d recordings.\n', recordings_processed_count, num_subjects);
if ~isempty(failed_recordings)
    fprintf('Failed recordings: %s\n', strjoin(failed_recordings, ', '));
end

if recordings_processed_count > 0
    all_subject_spectrograms = results_spectrograms(successful_indices);
    all_subject_labels = results_labels(successful_indices);
    processed_subject_info = results_info(successful_indices); % Renamed for clarity

    total_epochs_processed = 0;
    for k = 1:recordings_processed_count
        if ~isempty(processed_subject_info{k}) && isfield(processed_subject_info{k}, 'num_valid_epochs')
             total_epochs_processed = total_epochs_processed + processed_subject_info{k}.num_valid_epochs;
        end
    end
     fprintf('Total valid epochs aggregated: %d\n', total_epochs_processed);

    if total_epochs_processed > 0
        fprintf('Concatenating spectrograms...\n');
        all_spectrograms = cat(1, all_subject_spectrograms{:});
        fprintf('Concatenating labels...\n');
        all_labels = vertcat(all_subject_labels{:});

        final_spec_size = size(all_spectrograms);
        final_label_size = size(all_labels);
        fprintf('Final spectrogram array size: [%s]\n', sprintf('%d ', final_spec_size));
        fprintf('Final labels array size: [%s]\n', sprintf('%d ', final_label_size));
        if final_spec_size(1) ~= total_epochs_processed || final_label_size(1) ~= total_epochs_processed
            warning('Mismatch between calculated total epochs and final array size!');
        end

        output_path = fullfile(output_dir, output_filename);
        fprintf('Saving aggregated data to: %s\n', output_path);
        try
            save(output_path, 'all_spectrograms', 'all_labels', 'processed_subject_info', 'configParams', '-v7.3');
            fprintf('Data saved successfully.\n');
        catch ME_save
            fprintf('ERROR saving data: %s\n', ME_save.message);
            fprintf('Aggregated data is available in workspace variables `all_spectrograms` and `all_labels`.\n');
        end
    else
         fprintf('\n--- No valid epochs found in successfully processed recordings. Nothing to save. ---\n');
    end
else
    fprintf('\n--- No recordings processed successfully. Nothing to save. ---\n');
end

if exist(configParams.temp_dir_base, 'dir') && numel(dir(configParams.temp_dir_base)) == 2
    try rmdir(configParams.temp_dir_base); catch; end
end

toc;
fprintf('\n--- Preprocessing for Sleep-EDFX SC40 finished ---\n');

% Optional: Shut down parallel pool
% delete(gcp('nocreate'));

% MATLAB Script to Prepare Processed Data for LOSO Cross-Validation
% Loads the aggregated data and groups it by subject ID.

clear;
clc;
close all;

fprintf('--- Part 5: Load Processed Data and Group by Subject ---\n');

% --- Configuration ---
processed_data_dir = fullfile(pwd, 'processed_data');
processed_filename = 'SleepEDFX_SC40_processed_parallel.mat'; % File saved from previous step
processed_filepath = fullfile(processed_data_dir, processed_filename);

% --- Load Data ---
if ~exist(processed_filepath, 'file')
    error('Processed data file not found: %s', processed_filepath);
end
fprintf('Loading processed data from: %s\n', processed_filepath);
load(processed_filepath, 'all_spectrograms', 'all_labels', 'processed_subject_info');
fprintf('Data loaded successfully.\n');

% Verify loaded variables
if ~exist('all_spectrograms', 'var') || ~exist('all_labels', 'var') || ~exist('processed_subject_info', 'var')
    error('Loaded .mat file does not contain the expected variables (all_spectrograms, all_labels, processed_subject_info).');
end

fprintf('Total epochs loaded: %d\n', size(all_spectrograms, 1));
fprintf('Spectrogram size: %d x %d x %d\n', size(all_spectrograms, 2), size(all_spectrograms, 3), size(all_spectrograms, 4));
fprintf('Number of recording info entries: %d\n', numel(processed_subject_info));

% --- Extract Subject IDs for Each Epoch ---
fprintf('Extracting subject IDs for each epoch...\n');

num_recordings = numel(processed_subject_info);
epoch_subject_ids = strings(size(all_labels)); % String array to store subject ID for each epoch
current_epoch_idx = 1;
subject_epoch_counts = zeros(num_recordings, 1); % Store epoch count per recording

subject_ids_list = cell(num_recordings, 1); % Store the base subject ID for each recording

for i = 1:num_recordings
    rec_info = processed_subject_info{i};
    if isempty(rec_info) || ~isfield(rec_info, 'psg_file') || ~isfield(rec_info, 'num_valid_epochs')
        warning('Skipping empty or incomplete info entry for recording index %d.', i);
        continue;
    end

    num_epochs_this_rec = rec_info.num_valid_epochs;
    subject_epoch_counts(i) = num_epochs_this_rec;

    % Extract base subject ID (e.g., 'SC400' from 'SC4001E0-PSG.edf')
    [~, psg_name, ~] = fileparts(rec_info.psg_file);
    % Assuming format SC4xxYE0-PSG where Y is night number
    base_subject_id = psg_name(1:5); % Extract 'SC4xx' part
    subject_ids_list{i} = base_subject_id;

    % Assign this subject ID to the corresponding epochs
    end_epoch_idx = current_epoch_idx + num_epochs_this_rec - 1;
    if end_epoch_idx > length(epoch_subject_ids)
         warning('Epoch index mismatch for recording %d (%s). Adjusting.', i, psg_name);
         end_epoch_idx = length(epoch_subject_ids); % Prevent out-of-bounds
         num_epochs_this_rec = end_epoch_idx - current_epoch_idx + 1;
         if num_epochs_this_rec < 0, num_epochs_this_rec = 0; end % Handle edge case
         subject_epoch_counts(i) = num_epochs_this_rec; % Update count
    end

    if num_epochs_this_rec > 0
        epoch_subject_ids(current_epoch_idx : end_epoch_idx) = base_subject_id;
    end

    current_epoch_idx = end_epoch_idx + 1;
end

% Verify total epochs match
if sum(subject_epoch_counts) ~= size(all_spectrograms, 1)
    warning('Total epoch count from subject info (%d) does not match spectrogram array size (%d). Check preprocessing aggregation.', ...
            sum(subject_epoch_counts), size(all_spectrograms, 1));
end

unique_subject_ids = unique(epoch_subject_ids);
% Filter out any potential empty strings if warnings occurred
unique_subject_ids = unique_subject_ids(strlength(unique_subject_ids) > 0);
num_unique_subjects = numel(unique_subject_ids);
fprintf('Found %d unique subject IDs.\n', num_unique_subjects);
disp('Unique Subject IDs:');
disp(unique_subject_ids');

if num_unique_subjects ~= 20
    warning('Expected 20 unique subjects for SC40 dataset, but found %d.', num_unique_subjects);
end

% --- Group Data by Subject ---
fprintf('Grouping data by subject ID...\n');
grouped_data = struct(); % Structure to hold data for each subject

for i = 1:num_unique_subjects
    subj_id = unique_subject_ids(i);
    fprintf(' Grouping data for subject: %s\n', subj_id);

    % Find all epochs belonging to this subject
    subject_indices = find(epoch_subject_ids == subj_id);

    if isempty(subject_indices)
        warning('No epochs found for subject %s, skipping.', subj_id);
        continue;
    end

    % Store the spectrograms and labels for this subject
    % Use makeValidName in case subject IDs have characters invalid for struct field names
    valid_field_name = matlab.lang.makeValidName(subj_id);
    grouped_data.(valid_field_name).Spectrograms = all_spectrograms(subject_indices, :, :, :);
    grouped_data.(valid_field_name).Labels = all_labels(subject_indices);
    grouped_data.(valid_field_name).SubjectID = subj_id; % Store original ID too
    grouped_data.(valid_field_name).NumEpochs = length(subject_indices);

    fprintf('  -> Found %d epochs.\n', grouped_data.(valid_field_name).NumEpochs);
end

fprintf('Data grouping complete.\n');

% --- Outline LOSO CV Loop ---
fprintf('\n--- Ready for Leave-One-Subject-Out Cross-Validation ---\n');
fprintf('Total unique subjects for folds: %d\n', num_unique_subjects);

% Example structure for the LOSO loop (actual training code goes inside)
for k = 1:num_unique_subjects
    test_subject_id = unique_subject_ids(k);
    test_subject_fieldname = matlab.lang.makeValidName(test_subject_id);
    fprintf('\n===== Fold %d/%d: Testing on Subject %s =====\n', k, num_unique_subjects, test_subject_id);

    % --- Get Test Data ---
    if isfield(grouped_data, test_subject_fieldname)
        testSpectrograms = grouped_data.(test_subject_fieldname).Spectrograms;
        testLabels = grouped_data.(test_subject_fieldname).Labels;
        fprintf(' Test Set: %d epochs.\n', size(testSpectrograms, 1));
    else
        warning('Test subject %s not found in grouped data for fold %d. Skipping fold.', test_subject_id, k);
        continue;
    end

    % --- Get Training Data (Combine data from all other subjects) ---
    trainSpectrogramsList = {};
    trainLabelsList = {};
    train_epoch_count = 0;
    for j = 1:num_unique_subjects
        if i == j % Skip the test subject
            continue;
        end
        train_subj_id = unique_subject_ids(j);
        train_subj_fieldname = matlab.lang.makeValidName(train_subj_id);

        if isfield(grouped_data, train_subj_fieldname)
            trainSpectrogramsList{end+1} = grouped_data.(train_subj_fieldname).Spectrograms;
            trainLabelsList{end+1} = grouped_data.(train_subj_fieldname).Labels;
            train_epoch_count = train_epoch_count + grouped_data.(train_subj_fieldname).NumEpochs;
        else
             warning('Training subject %s not found in grouped data for fold %d.', train_subj_id, k);
        end
    end

    if isempty(trainSpectrogramsList)
        warning('No training data found for fold %d. Skipping fold.', k);
        continue;
    end

    % Concatenate training data
    trainSpectrograms = cat(1, trainSpectrogramsList{:});
    trainLabels = vertcat(trainLabelsList{:});
    fprintf(' Training Set: %d epochs from %d subjects.\n', size(trainSpectrograms, 1), num_unique_subjects - 1);

    % --- TODO: Implement Model Training and Evaluation Here ---
    % 1. (Optional) Split trainSpectrograms/trainLabels further into training/validation sets.
    % 2. Define/load your deep learning model architecture (CNN+BiLSTM).
    % 3. Set training options (optimizer, learning rate, epochs, etc.).
    % 4. Create datastores (e.g., arrayDatastore, custom sequence datastore) for training and testing.
    % 5. Train the model using trainNetwork on the training datastore.
    % 6. Evaluate the trained model on testSpectrograms/testLabels.
    % 7. Store performance metrics for this fold.
    % Example placeholder:
    fprintf('   (Placeholder: Train and evaluate model for fold %d...)\n', k);
    % metrics(k) = trainAndEvaluateModel(trainSpectrograms, trainLabels, testSpectrograms, testLabels, ...);

    clear testSpectrograms testLabels trainSpectrograms trainLabels trainSpectrogramsList trainLabelsList; % Clear large fold data

end % End LOSO loop

% --- TODO: Aggregate and report cross-validation results ---
% fprintf('\n--- Cross-Validation Finished ---\n');
% reportCrossValidationResults(metrics);

fprintf('\n--- Data preparation script finished ---\n');
disp('Data grouped by subject is available in the `grouped_data` structure.');
disp('Next steps involve implementing the training loop using this structure.');