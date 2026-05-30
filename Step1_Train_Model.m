% =========================================================================
% File: Step1_Train_Model.m
% Purpose: Train a Random Forest model for three EMG actions.
%
% Data format supported:
%   1) CSV with columns: time_s, EMG_ch1_uV, EMG_ch2_uV
%   2) CSV with only EMG_ch1_uV, EMG_ch2_uV
%
% Label mapping:
%   1 = Curl (wanju)
%   2 = Shoulder press (tuijian)
%   3 = Lateral raise (cepingju)
% =========================================================================

clear; clc; close all;
rng(42);

scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end

cfg.dataPath = 'C:\Users\27578\Desktop\ProjectPreprocess_adjusted.zip\project\results\csv';
cfg.modelFile = fullfile(scriptDir, 'MyEMG_RF_Model.mat');
cfg.fallbackFs = 2000;
cfg.windowSeconds = 0.25;
cfg.windowOverlap = 0.50;
cfg.energyKeepPercentile = 10;
cfg.bandpassHz = [20 450];
cfg.notchHz = 50;
cfg.numTrees = 500;
cfg.cvTrees = 200;
cfg.minLeafSize = 3;
cfg.numPredictorsToSample = [];
cfg.preferMatWhenAvailable = false;
cfg.numChannels = [];

labelMap.labels = ["1", "2", "3"];
labelMap.names = ["Curl (wanju)", "Shoulder press (tuijian)", ...
    "Lateral raise (cepingju)"];

fprintf('Locating dataset...\n');
datasetDir = resolveDatasetPath(cfg.dataPath);
fprintf('Dataset directory: %s\n', datasetDir);

records = discoverRecords(datasetDir, cfg.preferMatWhenAvailable);
records = keepKnownLabels(records, labelMap);
if isempty(records)
    error('No training CSV/MAT files with labels 1/2/3 were found.');
end

fprintf('Found %d independent records.\n', numel(records));
printRecordCounts(records, labelMap);

fprintf('\nExtracting windowed EMG features...\n');
X = [];
Y = {};
recordIdx = [];
recordInfo = struct('name', {}, 'label', {}, 'action', {}, 'file', {}, ...
    'fs', {}, 'windows', {}, 'source', {});
featureNames = {};

for i = 1:numel(records)
    try
        [rawData, fs] = readEmgRecord(records(i).filePath, cfg.fallbackFs);
        data = preprocessEmg(rawData, fs, cfg);

        if isempty(cfg.numChannels)
            cfg.numChannels = size(data, 2);
            fprintf('Detected %d EMG channel(s).\n', cfg.numChannels);
        elseif size(data, 2) < cfg.numChannels
            warning('Skipping %s: expected %d channel(s), found %d.', ...
                records(i).name, cfg.numChannels, size(data, 2));
            continue;
        elseif size(data, 2) > cfg.numChannels
            data = data(:, 1:cfg.numChannels);
        end

        [Xi, featureNames] = extractWindowFeatures(data, fs, cfg);
    catch ME
        warning('Skipping %s: %s', records(i).name, ME.message);
        continue;
    end

    if isempty(Xi)
        warning('Skipping %s: no valid windows.', records(i).name);
        continue;
    end

    X = [X; Xi]; %#ok<AGROW>
    Y = [Y; repmat({char(records(i).label)}, size(Xi, 1), 1)]; %#ok<AGROW>
    recordIdx = [recordIdx; repmat(i, size(Xi, 1), 1)]; %#ok<AGROW>

    recordInfo(end + 1).name = records(i).name; %#ok<SAGROW>
    recordInfo(end).label = char(records(i).label);
    recordInfo(end).action = char(labelToName(records(i).label, labelMap));
    recordInfo(end).file = records(i).filePath;
    recordInfo(end).fs = fs;
    recordInfo(end).windows = size(Xi, 1);
    recordInfo(end).source = records(i).source;

    fprintf('  %-22s label %-2s Fs=%4d Hz windows=%3d\n', ...
        records(i).name, char(records(i).label), fs, size(Xi, 1));
end

if isempty(X)
    error('Feature matrix is empty; training cannot continue.');
end

cfg.numPredictorsToSample = max(1, round(sqrt(size(X, 2))));
fprintf('\nTotal windows: %d, features: %d, predictors per tree: %d\n', ...
    size(X, 1), size(X, 2), cfg.numPredictorsToSample);
printWindowCounts(Y, labelMap);

fprintf('\nRunning grouped cross-validation by record...\n');
trainingInfo = struct();
trainingInfo.groupedCv = groupedCvByRecord(X, Y, recordIdx, records, labelMap, cfg);

fprintf('\nTraining final Random Forest...\n');
[X_bal, Y_bal] = balanceClasses(X, Y);
[X_train, mu, sigma] = standardizeFit(X_bal);

rfModel = TreeBagger(cfg.numTrees, X_train, Y_bal, ...
    'Method', 'classification', ...
    'OOBPrediction', 'on', ...
    'OOBPredictorImportance', 'on', ...
    'MinLeafSize', cfg.minLeafSize, ...
    'NumPredictorsToSample', cfg.numPredictorsToSample, ...
    'Prior', 'uniform');

trainingInfo.createdAt = char(datetime('now'));
trainingInfo.datasetDir = datasetDir;
trainingInfo.totalWindows = size(X, 1);
trainingInfo.balancedWindows = size(X_train, 1);
trainingInfo.featureCount = size(X, 2);
trainingInfo.recordInfo = recordInfo;
trainingInfo.oobError = oobError(rfModel);

save(cfg.modelFile, 'rfModel', 'mu', 'sigma', 'featureNames', 'cfg', ...
    'labelMap', 'trainingInfo');

fprintf('\nTraining complete. Model saved to:\n%s\n', cfg.modelFile);
if ~isempty(trainingInfo.groupedCv.accuracy)
    fprintf('Grouped record-level CV accuracy: %.2f%%\n', ...
        trainingInfo.groupedCv.accuracy * 100);
end
fprintf('Final model OOB error: %.2f%%\n', trainingInfo.oobError(end) * 100);

plotTrainingFigures(rfModel, featureNames, trainingInfo.groupedCv, labelMap);

% =========================================================================
% Local functions
% =========================================================================

function datasetDir = resolveDatasetPath(dataPath)
    dataPath = char(dataPath);
    if isfolder(dataPath)
        datasetDir = dataPath;
        return;
    end

    lowerPath = lower(dataPath);
    zipPos = strfind(lowerPath, '.zip');
    if isempty(zipPos)
        error('Dataset path does not exist: %s', dataPath);
    end

    zipEnd = zipPos(1) + 3;
    zipFile = dataPath(1:zipEnd);
    if ~isfile(zipFile)
        error('Zip file was not found: %s', zipFile);
    end

    innerPath = '';
    if numel(dataPath) > zipEnd
        innerPath = dataPath(zipEnd + 1:end);
        innerPath = regexprep(innerPath, '^[\\/]+', '');
    end

    [~, zipName] = fileparts(zipFile);
    cacheRoot = fullfile(tempdir, ['emg_dataset_cache_' sanitizeName(zipName)]);
    datasetDir = fullfile(cacheRoot, innerPath);

    if ~isfolder(datasetDir)
        if isfolder(cacheRoot)
            try
                rmdir(cacheRoot, 's');
            catch
            end
        end
        mkdir(cacheRoot);
        unzip(zipFile, cacheRoot);
    end

    if ~isfolder(datasetDir)
        error('Zip was extracted, but inner folder was not found: %s', datasetDir);
    end
end

function safe = sanitizeName(name)
    safe = regexprep(name, '[^a-zA-Z0-9_]+', '_');
end

function records = discoverRecords(datasetDir, preferMat)
    matFiles = dir(fullfile(datasetDir, '*.mat'));
    csvFiles = dir(fullfile(datasetDir, '*.csv'));

    bases = strings(0, 1);
    for i = 1:numel(matFiles)
        [~, base] = fileparts(matFiles(i).name);
        bases(end + 1) = string(base); %#ok<AGROW>
    end
    for i = 1:numel(csvFiles)
        [~, base] = fileparts(csvFiles(i).name);
        bases(end + 1) = string(base); %#ok<AGROW>
    end
    bases = unique(bases);

    records = struct('name', {}, 'label', {}, 'filePath', {}, 'source', {});
    for i = 1:numel(bases)
        base = char(bases(i));
        token = regexp(base, '^(\d+)', 'tokens', 'once');
        if isempty(token)
            continue;
        end

        matPath = fullfile(datasetDir, [base '.mat']);
        csvPath = fullfile(datasetDir, [base '.csv']);
        if preferMat && isfile(matPath)
            filePath = matPath;
            source = 'mat';
        elseif isfile(csvPath)
            filePath = csvPath;
            source = 'csv';
        elseif isfile(matPath)
            filePath = matPath;
            source = 'mat';
        else
            continue;
        end

        records(end + 1).name = base; %#ok<AGROW>
        records(end).label = string(token{1});
        records(end).filePath = filePath;
        records(end).source = source;
    end

    if ~isempty(records)
        [~, order] = sort({records.name});
        records = records(order);
    end
end

function records = keepKnownLabels(records, labelMap)
    keep = false(numel(records), 1);
    for i = 1:numel(records)
        keep(i) = any(labelMap.labels == records(i).label);
    end
    records = records(keep);
end

function printRecordCounts(records, labelMap)
    labels = string({records.label});
    for i = 1:numel(labelMap.labels)
        fprintf('  %s %-28s %d records\n', char(labelMap.labels(i)), ...
            char(labelMap.names(i)), sum(labels == labelMap.labels(i)));
    end
end

function printWindowCounts(Y, labelMap)
    labels = string(Y);
    for i = 1:numel(labelMap.labels)
        fprintf('  %s %-28s %d windows\n', char(labelMap.labels(i)), ...
            char(labelMap.names(i)), sum(labels == labelMap.labels(i)));
    end
end

function [rawData, fs] = readEmgRecord(filePath, fallbackFs)
    [~, ~, ext] = fileparts(filePath);
    fs = fallbackFs;

    if strcmpi(ext, '.mat')
        S = load(filePath);
        if isfield(S, 'Fs') && isnumeric(S.Fs) && isscalar(S.Fs) && ...
                isfinite(S.Fs) && S.Fs > 0
            fs = double(S.Fs);
        end

        if isfield(S, 'data') && isnumeric(S.data)
            rawData = S.data;
        else
            names = fieldnames(S);
            rawData = [];
            for i = 1:numel(names)
                value = S.(names{i});
                if isnumeric(value) && ismatrix(value) && size(value, 2) >= 2
                    rawData = value;
                    break;
                end
            end
            if isempty(rawData)
                error('No numeric two-channel matrix was found in the MAT file.');
            end
        end
    else
        rawData = readmatrix(filePath);
        if size(rawData, 2) >= 2 && isLikelyTimeColumn(rawData(:, 1))
            fs = inferFsFromTime(rawData(:, 1), fallbackFs);
        end
    end
end

function fs = inferFsFromTime(t, fallbackFs)
    t = t(:);
    t = t(isfinite(t));
    dt = diff(t);
    dt = dt(isfinite(dt) & dt > 0);
    if isempty(dt)
        fs = fallbackFs;
        return;
    end

    fs = round(1 / median(dt));
    if ~isfinite(fs) || fs <= 0
        fs = fallbackFs;
    end
end

function data = preprocessEmg(rawData, fs, cfg)
    rawData = double(rawData);
    rawData = rawData(any(isfinite(rawData), 2), :);
    rawData = rawData(:, any(isfinite(rawData), 1));
    data = selectEmgColumns(rawData);

    if ~isempty(cfg.numChannels)
        if size(data, 2) < cfg.numChannels
            error('Expected %d channel(s), found %d.', cfg.numChannels, size(data, 2));
        end
        data = data(:, 1:cfg.numChannels);
    end

    for ch = 1:size(data, 2)
        x = data(:, ch);
        finiteMask = isfinite(x);
        if ~any(finiteMask)
            x(:) = 0;
        else
            x(~finiteMask) = median(x(finiteMask));
        end
        data(:, ch) = x;
    end

    data = detrend(data);

    if fs > 2 * cfg.bandpassHz(1)
        highCut = min(cfg.bandpassHz(2), 0.45 * fs);
        lowCut = cfg.bandpassHz(1);
        if highCut > lowCut
            try
                [b, a] = butter(4, [lowCut highCut] / (fs / 2), 'bandpass');
                data = filtfilt(b, a, data);
            catch ME
                warning('%s', ['Band-pass filtering skipped: ' ME.message]);
            end
        end
    end

    if cfg.notchHz > 0 && cfg.notchHz < fs / 2 && exist('iirnotch', 'file') == 2
        try
            wo = cfg.notchHz / (fs / 2);
            bw = wo / 35;
            [bn, an] = iirnotch(wo, bw);
            data = filtfilt(bn, an, data);
        catch ME
            warning('%s', ['Notch filtering skipped: ' ME.message]);
        end
    end
end

function data = selectEmgColumns(rawData)
    if isempty(rawData) || size(rawData, 2) < 1
        error('No numeric data was found.');
    end

    if size(rawData, 2) >= 3 && isLikelyTimeColumn(rawData(:, 1))
        data = rawData(:, 2:3);
    elseif size(rawData, 2) >= 2 && isLikelyTimeColumn(rawData(:, 1))
        data = rawData(:, 2:end);
    elseif size(rawData, 2) >= 2
        data = rawData(:, 1:2);
    else
        error('Could not find two EMG signal columns.');
    end

    if size(data, 2) < 2
        error('Only one EMG channel was found; two channels are required.');
    end
end

function tf = isLikelyTimeColumn(x)
    x = x(:);
    x = x(isfinite(x));
    if numel(x) < 10
        tf = false;
        return;
    end

    dx = diff(x);
    positiveRatio = mean(dx > 0);
    positiveSteps = dx(dx > 0);
    if isempty(positiveSteps)
        tf = false;
        return;
    end

    medianStep = median(positiveSteps);
    tf = positiveRatio > 0.95 && medianStep > 0 && medianStep < 1;
end

function [X, featureNames] = extractWindowFeatures(data, fs, cfg)
    win = max(32, round(cfg.windowSeconds * fs));
    if size(data, 1) < win
        win = size(data, 1);
    end
    step = max(1, round(win * (1 - cfg.windowOverlap)));
    starts = 1:step:(size(data, 1) - win + 1);

    if isempty(starts) || win < 16
        X = [];
        featureNames = {};
        return;
    end

    X = [];
    energy = zeros(numel(starts), 1);
    featureNames = {};
    for i = 1:numel(starts)
        seg = data(starts(i):starts(i) + win - 1, :);
        [row, featureNames] = extractFeatureVector(seg, fs);
        X = [X; row]; %#ok<AGROW>
        energy(i) = mean(sqrt(mean(seg .^ 2, 1)));
    end

    if size(X, 1) >= 5 && cfg.energyKeepPercentile > 0
        threshold = prctile(energy, cfg.energyKeepPercentile);
        keep = energy >= threshold;
        if sum(keep) >= 3
            X = X(keep, :);
        end
    end
end

function [row, names] = extractFeatureVector(seg, fs)
    row = [];
    names = {};
    for ch = 1:size(seg, 2)
        x = seg(:, ch);
        prefix = sprintf('Ch%d_', ch);
        [feat, featNames] = channelFeatures(x, fs, prefix);
        row = [row, feat]; %#ok<AGROW>
        names = [names, featNames]; %#ok<AGROW>
    end

    if size(seg, 2) >= 2
        ch1 = seg(:, 1);
        ch2 = seg(:, 2);
        rms1 = sqrt(mean(ch1 .^ 2));
        rms2 = sqrt(mean(ch2 .^ 2));
        mav1 = mean(abs(ch1));
        mav2 = mean(abs(ch2));
        c = corrcoef(ch1, ch2);
        if numel(c) >= 4 && isfinite(c(1, 2))
            corr12 = c(1, 2);
        else
            corr12 = 0;
        end

        row = [row, rms1 / (rms2 + eps), mav1 / (mav2 + eps), corr12];
        names = [names, {'RMS_Ratio_1_2', 'MAV_Ratio_1_2', 'Channel_Corr'}];
    end
    row(~isfinite(row)) = 0;
end

function [feat, names] = channelFeatures(x, fs, prefix)
    x = x(:);
    n = numel(x);
    dx = diff(x);
    threshold = 0.02 * std(x);

    mav = mean(abs(x));
    rmsv = sqrt(mean(x .^ 2));
    varv = var(x);
    wl = sum(abs(dx)) / max(1, n - 1);
    zc = sum(abs(diff(signWithZero(x))) > 0 & abs(dx) > threshold) / max(1, n - 1);
    ssc = sum(((dx(1:end - 1) .* dx(2:end)) < 0) & ...
        (abs(dx(1:end - 1) - dx(2:end)) > threshold)) / max(1, n - 2);
    wamp = sum(abs(dx) > threshold) / max(1, n - 1);
    dasdv = sqrt(mean(dx .^ 2));
    iemg = sum(abs(x)) / max(1, n);
    logdet = exp(mean(log(abs(x) + eps)));
    skewv = skewness(x);
    kurtv = kurtosis(x);

    [mnf, mdf, pkf, bandLow, bandMid, bandHigh] = frequencyFeatures(x, fs);

    feat = [mav, rmsv, varv, wl, zc, ssc, wamp, dasdv, iemg, logdet, ...
        skewv, kurtv, mnf, mdf, pkf, bandLow, bandMid, bandHigh];

    baseNames = {'MAV', 'RMS', 'VAR', 'WL', 'ZC', 'SSC', 'WAMP', 'DASDV', ...
        'IEMG', 'LOG', 'Skewness', 'Kurtosis', 'MNF', 'MDF', 'PKF', ...
        'Band_20_60', 'Band_60_150', 'Band_150_450'};
    names = cellfun(@(s) [prefix s], baseNames, 'UniformOutput', false);
end

function s = signWithZero(x)
    s = sign(x);
    s(s == 0) = 1;
end

function [mnf, mdf, pkf, bandLow, bandMid, bandHigh] = frequencyFeatures(x, fs)
    nfft = min(2048, max(128, 2 ^ nextpow2(numel(x))));
    winLen = min(numel(x), max(64, round(numel(x) / 2)));
    if winLen < 8
        mnf = 0; mdf = 0; pkf = 0; bandLow = 0; bandMid = 0; bandHigh = 0;
        return;
    end

    try
        [pxx, f] = pwelch(x, hamming(winLen), [], nfft, fs);
    catch
        try
            [pxx, f] = periodogram(x, [], nfft, fs);
        catch
            mnf = 0; mdf = 0; pkf = 0; bandLow = 0; bandMid = 0; bandHigh = 0;
            return;
        end
    end

    totalPower = sum(pxx) + eps;
    mnf = sum(f .* pxx) / totalPower;
    cpower = cumsum(pxx);
    mdfIdx = find(cpower >= totalPower / 2, 1, 'first');
    if isempty(mdfIdx)
        mdf = 0;
    else
        mdf = f(mdfIdx);
    end
    [~, pkIdx] = max(pxx);
    pkf = f(pkIdx);
    bandLow = bandPowerRatio(pxx, f, 20, 60, totalPower);
    bandMid = bandPowerRatio(pxx, f, 60, 150, totalPower);
    bandHigh = bandPowerRatio(pxx, f, 150, min(450, fs / 2), totalPower);
end

function ratio = bandPowerRatio(pxx, f, lowHz, highHz, totalPower)
    if highHz <= lowHz
        ratio = 0;
        return;
    end
    mask = f >= lowHz & f < highHz;
    ratio = sum(pxx(mask)) / totalPower;
end

function [Xb, Yb] = balanceClasses(X, Y)
    classes = unique(string(Y));
    counts = zeros(numel(classes), 1);
    for i = 1:numel(classes)
        counts(i) = sum(string(Y) == classes(i));
    end
    target = min(counts);

    idx = [];
    for i = 1:numel(classes)
        classIdx = find(string(Y) == classes(i));
        classIdx = classIdx(randperm(numel(classIdx), target));
        idx = [idx; classIdx(:)]; %#ok<AGROW>
    end
    idx = idx(randperm(numel(idx)));
    Xb = X(idx, :);
    Yb = Y(idx);
end

function [Xz, mu, sigma] = standardizeFit(X)
    mu = nanmeanLocal(X, 1);
    sigma = nanstdLocal(X, 0, 1);
    sigma(~isfinite(sigma) | sigma < eps) = 1;
    Xz = standardizeApply(X, mu, sigma);
end

function Xz = standardizeApply(X, mu, sigma)
    Xz = (X - mu) ./ sigma;
    Xz(~isfinite(Xz)) = 0;
end

function m = nanmeanLocal(X, dim)
    mask = isfinite(X);
    X(~mask) = 0;
    count = sum(mask, dim);
    count(count == 0) = 1;
    m = sum(X, dim) ./ count;
end

function s = nanstdLocal(X, flag, dim)
    m = nanmeanLocal(X, dim);
    centered = X - m;
    centered(~isfinite(centered)) = 0;
    count = sum(isfinite(X), dim);
    if flag == 0
        denom = max(count - 1, 1);
    else
        denom = max(count, 1);
    end
    s = sqrt(sum(centered .^ 2, dim) ./ denom);
end

function cvInfo = groupedCvByRecord(X, Y, recordIdx, records, labelMap, cfg)
    activeRecords = unique(recordIdx);
    activeLabels = strings(numel(activeRecords), 1);
    for i = 1:numel(activeRecords)
        activeLabels(i) = records(activeRecords(i)).label;
    end

    classCounts = zeros(numel(labelMap.labels), 1);
    for i = 1:numel(labelMap.labels)
        classCounts(i) = sum(activeLabels == labelMap.labels(i));
    end
    k = min(5, min(classCounts));

    cvInfo = struct('accuracy', [], 'foldAccuracy', [], 'truth', {{}}, ...
        'predicted', {{}}, 'scores', []);
    if k < 2
        warning('Too few records per class; grouped cross-validation was skipped.');
        return;
    end

    cvp = cvpartition(categorical(activeLabels), 'KFold', k);
    truthAll = strings(0, 1);
    predAll = strings(0, 1);
    scoreAll = [];
    foldAccuracy = zeros(k, 1);

    for fold = 1:k
        valRecordIds = activeRecords(test(cvp, fold));
        valMask = ismember(recordIdx, valRecordIds);
        trainMask = ~valMask;

        [Xtr, Ytr] = balanceClasses(X(trainMask, :), Y(trainMask));
        [Xtr, mu, sigma] = standardizeFit(Xtr);
        Xval = standardizeApply(X(valMask, :), mu, sigma);

        model = TreeBagger(cfg.cvTrees, Xtr, Ytr, ...
            'Method', 'classification', ...
            'MinLeafSize', cfg.minLeafSize, ...
            'NumPredictorsToSample', cfg.numPredictorsToSample, ...
            'Prior', 'uniform');

        [~, scores] = predict(model, Xval);
        valRecordVector = recordIdx(valMask);
        foldTruth = strings(numel(valRecordIds), 1);
        foldPred = strings(numel(valRecordIds), 1);

        for i = 1:numel(valRecordIds)
            rid = valRecordIds(i);
            rows = valRecordVector == rid;
            meanScore = mean(scores(rows, :), 1);
            [~, bestIdx] = max(meanScore);
            predLabel = string(model.ClassNames{bestIdx});
            trueLabel = records(rid).label;

            foldTruth(i) = trueLabel;
            foldPred(i) = predLabel;
            scoreAll = [scoreAll; meanScore]; %#ok<AGROW>
        end

        foldAccuracy(fold) = mean(foldTruth == foldPred);
        truthAll = [truthAll; foldTruth]; %#ok<AGROW>
        predAll = [predAll; foldPred]; %#ok<AGROW>
        fprintf('  Fold %d/%d: %.2f%%\n', fold, k, foldAccuracy(fold) * 100);
    end

    cvInfo.accuracy = mean(truthAll == predAll);
    cvInfo.foldAccuracy = foldAccuracy;
    cvInfo.truth = cellstr(truthAll);
    cvInfo.predicted = cellstr(predAll);
    cvInfo.scores = scoreAll;
end

function name = labelToName(label, labelMap)
    idx = find(labelMap.labels == string(label), 1, 'first');
    if isempty(idx)
        name = "Unknown action";
    else
        name = labelMap.names(idx);
    end
end

function plotTrainingFigures(rfModel, featureNames, cvInfo, labelMap)
    figure('Name', 'Random Forest OOB Error');
    plot(oobError(rfModel), 'LineWidth', 2);
    xlabel('Number of trees');
    ylabel('Out-of-bag error');
    title('Random Forest OOB error');
    grid on;

    imp = rfModel.OOBPermutedPredictorDeltaError;
    [sortedImp, order] = sort(imp, 'descend');
    topN = min(20, numel(sortedImp));
    figure('Name', 'Feature Importance');
    bar(sortedImp(1:topN), 'FaceColor', [0.2 0.55 0.8], 'EdgeColor', 'none');
    title('Top EMG feature importance');
    ylabel('OOB permuted importance');
    xticks(1:topN);
    xticklabels(featureNames(order(1:topN)));
    xtickangle(45);
    grid on;

    if ~isempty(cvInfo.accuracy) && exist('confusionchart', 'file') == 2
        truth = categorical(string(cvInfo.truth), labelMap.labels, labelMap.names);
        pred = categorical(string(cvInfo.predicted), labelMap.labels, labelMap.names);
        figure('Name', 'Grouped Cross-Validation Confusion Matrix');
        confusionchart(truth, pred, 'RowSummary', 'row-normalized', ...
            'ColumnSummary', 'column-normalized');
        title(sprintf('Grouped record-level CV %.2f%%', cvInfo.accuracy * 100));
    end
end
