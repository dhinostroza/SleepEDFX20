function [processedSpectrograms, processedLabels, subjectInfo, success] = preprocessSubjectEDFX(psgFilePath, hypFilePath, configParams)
%preprocessSubjectEDFX Preprocesses a single subject's EDF/EDF+ files for sleep staging.
%   Follows the pipeline described in Li et al., IJERPH 2022, 19, 6322.
%   Handles reading, resampling, epoching, label mapping, spectrogram
%   generation/cropping, normalization, and label encoding.
%
%   Inputs:
%       psgFilePath     - Full path to the subject's PSG EDF file.
%       hypFilePath     - Full path to the subject's Hypnogram EDF+ file.
%       configParams    - Structure containing configuration parameters.
%
%   Outputs:
%       processedSpectrograms - [N x H x W x C] single array of normalized spectrograms [0,1].
%       processedLabels       - [N x 1] int32 array of numerical labels.
%       subjectInfo           - Structure with info like fs, num_epochs etc.
%       success               - Logical true if preprocessing succeeded, false otherwise.

% Initialize outputs
processedSpectrograms = [];
processedLabels = [];
subjectInfo = struct();
success = false;

fprintf('\n--- Processing Subject ---\n PSG: %s\n HYP: %s\n', psgFilePath, hypFilePath);

try
    % =========================================================================
    % PART 1: Data Reading
    % =========================================================================
    fprintf('--- Part 1: Data Reading ---\n');
    subject_data = struct(); % Temporary structure for this subject

    if ~exist(psgFilePath, 'file'), error('PSG file not found.'); end
    if ~exist(hypFilePath, 'file'), error('Hypnogram file not found.'); end

    % --- Read PSG Data ---
    fprintf('Reading PSG file header info using edfinfo...\n');
    signalInfo = edfinfo(psgFilePath);
    fprintf('Successfully ran edfinfo.\n');
    fprintf('Attempting to extract details from object...\n');

    if ~isprop(signalInfo, 'SignalLabels'), error('Could not find SignalLabels property.'); end
    channel_labels = string(signalInfo.SignalLabels);
    eeg_channel_index = find(strcmpi(channel_labels, configParams.target_eeg_channel), 1);
    if isempty(eeg_channel_index), fprintf('Available channels:\n'); disp(channel_labels); error('Target EEG channel "%s" not found.', configParams.target_eeg_channel); end
    fprintf('Found target channel "%s" at index %d.\n', configParams.target_eeg_channel, eeg_channel_index);

    hasNumSamples = isprop(signalInfo, 'NumSamples');
    hasDuration = isprop(signalInfo, 'DataRecordDuration');
    hasSampleRate = isprop(signalInfo, 'SampleRate');
    if hasNumSamples && hasDuration && signalInfo.DataRecordDuration > 0
        record_duration_seconds = seconds(signalInfo.DataRecordDuration);
        fs_original = signalInfo.NumSamples(eeg_channel_index) / record_duration_seconds;
    elseif hasSampleRate, fs_original = signalInfo.SampleRate(eeg_channel_index);
    else, error('Could not determine sample rate.'); end
    fprintf('Original Sampling Frequency: %.2f Hz\n', fs_original);

    fprintf('Reading signal data for channel "%s" using edfread...\n', configParams.target_eeg_channel);
    [signalDataTable, ~] = edfread(psgFilePath, 'SelectedSignals', configParams.target_eeg_channel);
    valid_channel_name = matlab.lang.makeValidName(configParams.target_eeg_channel);
    fprintf('Attempting to access table column using generated name: "%s"\n', valid_channel_name);
    if ~ismember(valid_channel_name, signalDataTable.Properties.VariableNames), fprintf('Actual table variable names:\n'); disp(signalDataTable.Properties.VariableNames); error('Cannot find expected column "%s".', valid_channel_name); end
    eeg_signal = signalDataTable{:, valid_channel_name};
    fprintf('Successfully read %d records/cells for channel "%s".\n', numel(eeg_signal));

    % --- Read Hypnogram Data ---
    fprintf('Reading Hypnogram file annotations...\n');
    [~, annotationTimetable] = edfread(hypFilePath);
    if isempty(annotationTimetable), error('No annotation timetable found.'); end
    if ~ismember('Duration', annotationTimetable.Properties.VariableNames), error('"Duration" column missing.'); end
    if ~ismember('Annotations', annotationTimetable.Properties.VariableNames), error('"Annotations" column missing.'); end
    annotation_onset_duration = annotationTimetable.Properties.RowTimes;
    annotation_dur_duration = annotationTimetable.Duration;
    annotation_labels_raw = string(annotationTimetable.Annotations);
    annotation_onset_sec = seconds(annotation_onset_duration);
    annotation_dur_sec = seconds(annotation_dur_duration);
    fprintf('Successfully read %d block annotations from timetable.\n', height(annotationTimetable));

    % --- Store Raw/Block Data ---
    subject_data.eeg_signal_raw_chunks = eeg_signal;
    subject_data.fs_original = fs_original;
    subject_data.block_annotations.onset_sec = annotation_onset_sec;
    subject_data.block_annotations.dur_sec = annotation_dur_sec;
    subject_data.block_annotations.labels_raw = annotation_labels_raw;
    fprintf('--- Data Reading Complete ---\n');

    % =========================================================================
    % PART 2: Resampling, Epoching, and Labeling
    % =========================================================================
    fprintf('--- Part 2: Resampling, Epoching, and Labeling ---\n');
    eeg_signal_input = subject_data.eeg_signal_raw_chunks;
    fs_original = subject_data.fs_original;
    block_onset_sec = subject_data.block_annotations.onset_sec;
    block_dur_sec = subject_data.block_annotations.dur_sec;
    block_labels_raw = subject_data.block_annotations.labels_raw;

    % --- Concatenate cell array ---
    if iscell(eeg_signal_input)
        fprintf('Input EEG signal is a cell array (%d cells). Concatenating...\n', numel(eeg_signal_input));
        is_numeric_vector = cellfun(@(x) isnumeric(x) && isvector(x), eeg_signal_input);
        if ~all(is_numeric_vector), error('Not all cells contain numeric vectors.'); end
        eeg_signal_original = vertcat(eeg_signal_input{:});
        fprintf('Concatenation complete. Signal length: %d samples.\n', length(eeg_signal_original));
    elseif isnumeric(eeg_signal_input) && isvector(eeg_signal_input)
        fprintf('Input EEG signal is a numeric vector.\n'); eeg_signal_original = eeg_signal_input;
    else, error('Input EEG signal has unexpected type: %s', class(eeg_signal_input)); end
    if ~isa(eeg_signal_original, 'double'), eeg_signal_original = double(eeg_signal_original); end

    % --- Resample ---
    fprintf('Resampling EEG signal from %.2f Hz to %.1f Hz...\n', fs_original, configParams.fs_target);
    if abs(fs_original - configParams.fs_target) > 1e-6
        [P, Q] = rat(configParams.fs_target / fs_original);
        if max(P,Q) > 1000, warning('Large resampling ratio (P=%d, Q=%d).', P, Q); end
        eeg_signal_resampled = resample(eeg_signal_original, P, Q);
        fs_new = configParams.fs_target;
        fprintf('Resampling complete. New signal length: %d samples.\n', length(eeg_signal_resampled));
    else, fprintf('Signal is already at target sampling rate.\n'); eeg_signal_resampled = eeg_signal_original; fs_new = fs_original; end

    % --- Calculate Epoch Boundaries ---
    epoch_length_samples = round(configParams.epoch_duration_sec * fs_new);
    num_samples_total = length(eeg_signal_resampled);
    num_epochs = floor(num_samples_total / epoch_length_samples);
    fprintf('Signal length: %d samples (%.2f Hz). Epoch length: %d samples (%d sec).\n', num_samples_total, fs_new, epoch_length_samples, configParams.epoch_duration_sec);
    fprintf('Total number of full epochs: %d\n', num_epochs);
    if num_epochs == 0, error('Signal too short for epochs.'); end

    % --- Assign Labels to Epochs & Map ---
    epoch_labels = strings(num_epochs, 1);
    epoch_start_times_sec = (0:(num_epochs-1)) * configParams.epoch_duration_sec;
    valid_epoch_indices = [];
    fprintf('Assigning labels to epochs...\n');
    for i = 1:num_epochs
        epoch_start_sec = epoch_start_times_sec(i);
        block_idx = find(block_onset_sec <= epoch_start_sec & epoch_start_sec < (block_onset_sec + block_dur_sec), 1, 'first');
        raw_label = "Unknown/NoAnnotation"; if ~isempty(block_idx), raw_label = block_labels_raw(block_idx); end
        mapped_label = "INVALID";
        switch lower(strtrim(raw_label))
            case "sleep stage w", mapped_label = "Wake"; case "sleep stage 1", mapped_label = "N1";
            case "sleep stage 2", mapped_label = "N2"; case "sleep stage 3", mapped_label = "N3";
            case "sleep stage 4", mapped_label = "N3"; case "sleep stage r", mapped_label = "REM";
            case {"sleep stage ?", "sleep stage movement", "movement time", "unknown/noannotation"}, mapped_label = "INVALID";
            otherwise, warning('Epoch %d: Unrecognized raw label "%s". Marking as INVALID.', i, raw_label); mapped_label = "INVALID";
        end
        epoch_labels(i) = mapped_label;
        if mapped_label ~= "INVALID", valid_epoch_indices(end+1) = i; end
    end
    fprintf('Label assignment complete. Found %d valid epochs.\n', length(valid_epoch_indices));

    % --- Extract Valid Epoch Signal Data ---
    num_valid_epochs = length(valid_epoch_indices);
    if num_valid_epochs == 0, error('No valid epochs found after labeling.'); end
    epoched_eeg = zeros(num_valid_epochs, epoch_length_samples, 'single');
    final_epoch_labels = strings(num_valid_epochs, 1);
    fprintf('Extracting signal data for valid epochs...\n');
    for i = 1:num_valid_epochs
        epoch_idx = valid_epoch_indices(i);
        start_sample = (epoch_idx - 1) * epoch_length_samples + 1;
        end_sample = epoch_idx * epoch_length_samples;
        epoched_eeg(i, :) = single(eeg_signal_resampled(start_sample:end_sample));
        final_epoch_labels(i) = epoch_labels(epoch_idx);
    end
    fprintf('Epoch signal extraction complete.\n');

    % --- Store Epoched Data ---
    subject_data.epoched_data.eeg = epoched_eeg;
    subject_data.epoched_data.labels = final_epoch_labels;
    subject_data.epoched_data.fs = fs_new;
    subject_data.epoched_data.num_valid_epochs = num_valid_epochs;
    clear eeg_signal_resampled eeg_signal_original eeg_signal_input epoch_labels valid_epoch_indices block_*
    fprintf('--- Epoching and Labeling Complete ---\n');

    % =========================================================================
    % PART 3: Spectrogram Generation and Cropping
    % =========================================================================
    fprintf('--- Part 3: Spectrogram Generation and Cropping ---\n');
    fs = subject_data.epoched_data.fs;
    epoched_eeg = subject_data.epoched_data.eeg;
    num_epochs = subject_data.epoched_data.num_valid_epochs; % Use num_valid_epochs

    window_samples = round(configParams.window_sec * fs);
    overlap_samples = round(window_samples * configParams.overlap_perc);

    % Create a subject-specific temporary directory
    [~, psg_name, ~] = fileparts(psgFilePath);
    temp_img_dir = fullfile(configParams.temp_dir_base, psg_name);
    if ~exist(temp_img_dir, 'dir'), mkdir(temp_img_dir); fprintf('Created temporary directory: %s\n', temp_img_dir);
    else, fprintf('Using existing temporary directory: %s\n', temp_img_dir); delete(fullfile(temp_img_dir, '*.png')); end

    spectrogram_images = zeros(num_epochs, configParams.final_img_size_paper(1), configParams.final_img_size_paper(2), 3, 'uint8');
    fprintf('Generating spectrogram images for %d epochs...\n', num_epochs);

    fig_h = figure('Visible', 'off', 'Units', 'inches', 'Position', [0, 0, configParams.image_width_in, configParams.image_height_in], ...
                   'PaperUnits', 'inches', 'PaperSize', [configParams.image_width_in, configParams.image_height_in], 'PaperPositionMode', 'manual', 'PaperPosition', [0 0 configParams.image_width_in configParams.image_height_in]);
    ax_h = axes(fig_h, 'Units', 'normalized', 'Position', [0 0 1 1]);

    for i = 1:num_epochs
        epoch_signal = epoched_eeg(i, :);
        [s, f, t] = spectrogram(epoch_signal, hamming(window_samples), overlap_samples, configParams.nfft, fs);
        freq_idx = find(f <= configParams.max_freq);
        s_db = 10 * log10(abs(s(freq_idx, :)).^2 + eps);

        imagesc(ax_h, t, f(freq_idx), s_db);
        set(ax_h, 'YDir', 'normal'); axis(ax_h, 'off'); colormap(ax_h, 'jet');

        temp_filename = fullfile(temp_img_dir, sprintf('epoch_%04d.png', i));
        print(fig_h, temp_filename, '-dpng', sprintf('-r%d', configParams.image_dpi));
        img_raw = imread(temp_filename);

        if size(img_raw, 1) < max(configParams.crop_rows) || size(img_raw, 2) < max(configParams.crop_cols)
             error('Epoch %d: Saved image size (%d H x %d W) is smaller than needed for cropping (max row %d, max col %d). Stopping.', ...
                     i, size(img_raw,1), size(img_raw,2), max(configParams.crop_rows), max(configParams.crop_cols));
        end
        img_cropped = img_raw(configParams.crop_rows, configParams.crop_cols, :);

        if size(img_cropped, 1) ~= configParams.final_img_size_paper(1) || size(img_cropped, 2) ~= configParams.final_img_size_paper(2)
            img_final = imresize(img_cropped, configParams.final_img_size_paper);
        else, img_final = img_cropped; end
        spectrogram_images(i, :, :, :) = img_final;
        % Only print progress occasionally within parfor to avoid command window clutter
        % if mod(i, 100) == 0 || i == num_epochs, fprintf('  Processed %d / %d spectrograms...\n', i, num_epochs); end
    end
    close(fig_h); % Close figure after loop for this subject
    fprintf('Spectrogram generation complete.\n');

    subject_data.spectrograms = spectrogram_images;

    fprintf('Cleaning up temporary image files...\n');
    delete(fullfile(temp_img_dir, '*.png'));
    rmdir(temp_img_dir);
    fprintf('Temporary directory removed.\n');
    fprintf('--- Spectrogram Generation Complete ---\n');

    % =========================================================================
    % PART 4: Normalization and Label Encoding
    % =========================================================================
    fprintf('--- Part 4: Normalization and Label Encoding ---\n');
    spectrogram_images_uint8 = subject_data.spectrograms;
    labels_string = subject_data.epoched_data.labels;
    num_epochs = subject_data.epoched_data.num_valid_epochs; % Use num_valid_epochs

    % --- Normalize Spectrogram Images ---
    fprintf('Normalizing spectrogram images (0-255) to floating point [0, 1]...\n');
    spectrogram_images_normalized = single(spectrogram_images_uint8) / 255.0;
    fprintf('Normalization complete. Data type: %s\n', class(spectrogram_images_normalized));

    % --- Encode Labels ---
    fprintf('Encoding string labels to numerical categories (0-4)...\n');
    labels_numerical = zeros(num_epochs, 1, 'int32');
    for i = 1:num_epochs
        label_str = labels_string(i);
        if isKey(configParams.label_map, label_str)
            labels_numerical(i) = configParams.label_map(label_str);
        else, warning('Label encoding: Unrecognized label "%s" at epoch %d. Assigning NaN.', label_str, i); labels_numerical(i) = NaN; end
    end
    if any(isnan(labels_numerical)), warning('Some labels could not be encoded.'); end
    fprintf('Label encoding complete.\n');

    % --- Prepare Outputs ---
    processedSpectrograms = spectrogram_images_normalized;
    processedLabels = labels_numerical;
    subjectInfo.fs_processed = fs_new;
    subjectInfo.num_valid_epochs = num_valid_epochs;
    subjectInfo.psg_file = psgFilePath;
    subjectInfo.hyp_file = hypFilePath;
    success = true; % Mark as successful

    fprintf('--- Data Preparation Complete for Subject ---\n');

catch ME
    % If any error occurred in the try block
    fprintf('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n');
    fprintf('ERROR processing subject:\n PSG: %s\n HYP: %s\n', psgFilePath, hypFilePath);
    fprintf('Error Message: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('Error occurred in file: %s, line: %d\n', ME.stack(1).file, ME.stack(1).line);
    end
    fprintf('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n');
    % Ensure outputs are empty and success is false (already initialized)
end

end % End of function preprocessSubjectEDFX