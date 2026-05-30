% =========================================================================
% File: Step2_Blind_Test.m
% Purpose: Load the trained EMG Random Forest and classify a blind record.
%
% The blind CSV may contain:
%   1) time_s, EMG_ch1_uV, EMG_ch2_uV
%   2) EMG_ch1_uV, EMG_ch2_uV
% =========================================================================

clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end

modelFile = fullfile(scriptDir, 'MyEMG_RF_Model.mat');
if ~isfile(modelFile)
    error('Model file was not found: %s. Run Step1_Train_Model.m first.', modelFile);
end

load(modelFile, 'rfModel', 'mu', 'sigma', 'featureNames', 'cfg', ...
    'labelMap', 'trainingInfo');

if ~exist('cfg', 'var') || isempty(cfg)
    cfg = defaultCfg();
end
if ~isfield(cfg, 'fallbackFs'), cfg.fallbackFs = 2000; end
if ~isfield(cfg, 'windowSeconds'), cfg.windowSeconds = 0.25; end
if ~isfield(cfg, 'windowOverlap'), cfg.windowOverlap = 0.50; end
if ~isfield(cfg, 'energyKeepPercentile'), cfg.energyKeepPercentile = 0; end
if ~isfield(cfg, 'bandpassHz'), cfg.bandpassHz = [20 450]; end
if ~isfield(cfg, 'notchHz'), cfg.notchHz = 50; end
if ~isfield(cfg, 'numChannels'), cfg.numChannels = []; end

if ~exist('labelMap', 'var') || isempty(labelMap)
    labelMap.labels = ["1", "2", "3"];
    labelMap.names = ["Curl (wanju)", "Shoulder press (tuijian)", ...
        "Lateral raise (cepingju)"];
end

if ~exist('featureNames', 'var')
    featureNames = {};
end

[blindFileName, blindFilePath] = uigetfile( ...
    {'*.csv;*.mat', 'EMG records (*.csv, *.mat)'; '*.*', 'All files'}, ...
    'Select the blind EMG record');
if isequal(blindFileName, 0)
    error('No blind file was selected.');
end

blindFileFullPath = fullfile(blindFilePath, blindFileName);
fprintf('Blind file: %s\n', blindFileFullPath);

[rawData, fs] = readEmgRecord(blindFileFullPath, cfg.fallbackFs);
data = preprocessEmg(rawData, fs, cfg);

if ~isempty(cfg.numChannels)
    if size(data, 2) < cfg.numChannels
        error('Blind file has %d channel(s), but the model expects %d.', ...
            size(data, 2), cfg.numChannels);
    elseif size(data, 2) > cfg.numChannels
        data = data(:, 1:cfg.numChannels);
    end
end

[X_blind, blindFeatureNames] = extractWindowFeatures(data, fs, cfg);
if isempty(X_blind)
    error('No valid EMG windows were extracted from the blind file.');
end

if size(X_blind, 2) ~= numel(mu)
    error('Feature mismatch: blind file produced %d features, model expects %d.', ...
        size(X_blind, 2), numel(mu));
end

X_blind = standardizeApply(X_blind, mu, sigma);
[predWindowLabels, predScores] = predict(rfModel, X_blind);
classes = classNamesToString(rfModel.ClassNames);
meanScores = mean(predScores, 1);

[bestScore, bestIdx] = max(meanScores);
predLabel = classes(bestIdx);
predictedName = labelToName(predLabel, labelMap);
windowLabels = string(predWindowLabels);
windowAgreement = mean(windowLabels == predLabel);

fprintf('\n=======================================\n');
fprintf('Predicted action: %s [label %s]\n', char(predictedName), char(predLabel));
fprintf('Mean confidence: %.2f%%\n', bestScore * 100);
fprintf('Window agreement: %.2f%% (%d windows, Fs=%d Hz)\n', ...
    windowAgreement * 100, size(X_blind, 1), fs);
fprintf('---------------------------------------\n');
fprintf('Scores by action:\n');

for i = 1:numel(labelMap.labels)
    label = labelMap.labels(i);
    idx = find(classes == label, 1, 'first');
    if isempty(idx)
        score = 0;
    else
        score = meanScores(idx);
    end
    fprintf('  label %s  %-28s %6.2f%%\n', char(label), ...
        char(labelMap.names(i)), score * 100);
end
fprintf('=======================================\n');

if ~isempty(featureNames) && numel(featureNames) == numel(blindFeatureNames)
    fprintf('Feature extractor is aligned with the trained model.\n');
end
if exist('trainingInfo', 'var') && isfield(trainingInfo, 'groupedCv') && ...
        isfield(trainingInfo.groupedCv, 'accuracy') && ...
        ~isempty(trainingInfo.groupedCv.accuracy)
    fprintf('Training grouped CV accuracy was %.2f%%.\n', ...
        trainingInfo.groupedCv.accuracy * 100);
end

% =========================================================================
% Local functions
% =========================================================================

function cfg = defaultCfg()
    cfg.fallbackFs = 2000;
    cfg.windowSeconds = 0.25;
    cfg.windowOverlap = 0.50;
    cfg.energyKeepPercentile = 0;
    cfg.bandpassHz = [20 450];
    cfg.notchHz = 50;
    cfg.numChannels = [];
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

function Xz = standardizeApply(X, mu, sigma)
    Xz = (X - mu) ./ sigma;
    Xz(~isfinite(Xz)) = 0;
end

function labels = classNamesToString(classNames)
    labels = string(classNames);
    labels = labels(:);
end

function name = labelToName(label, labelMap)
    idx = find(labelMap.labels == string(label), 1, 'first');
    if isempty(idx)
        name = "Unknown action";
    else
        name = labelMap.names(idx);
    end
end
