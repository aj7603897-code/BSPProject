% =========================================================================
% File: Step0_Motion_Event_Detection_Unscheduled.m
% Purpose: Detect, count, and segment complete EMG motion cycles.
%
% Experiment protocol:
%   - Each record is about 30 s.
%   - This variant does not use a preset repetition count or repetition
%     interval for event detection.
%   - Complete repetitions after 30 s are kept when their boundaries return
%     to quiet activity before the record ends.
%
% Outputs:
%   EventDetectionResults_Unscheduled/summary.csv
%   EventDetectionResults_Unscheduled/event_boundaries.csv
%   EventDetectionResults_Unscheduled/segmented_records/<record>/<record>_repXX.csv
%   EventDetectionResults_Unscheduled/figures/<record>_events.png
% =========================================================================

clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end

cfg.dataDir = fullfile(scriptDir, 'csv');
cfg.outputDir = fullfile(scriptDir, 'EventDetectionResults_Unscheduled');
cfg.segmentRootDir = fullfile(cfg.outputDir, 'segmented_records');
cfg.figureDir = fullfile(cfg.outputDir, 'figures');
cfg.fallbackFs = 2000;

cfg.detectionMethod = 'threshold';
cfg.minPeakSeparationSeconds = 1.20;
cfg.mergeQuietGapSeconds = 0.70;
cfg.minPeakToThresholdRatio = 1.08;
cfg.gmmWindowSeconds = 0.10;
cfg.gmmStepSeconds = 0.05;
cfg.gmmActiveOnProbability = 0.65;
cfg.gmmActiveOffProbability = 0.35;
cfg.minActiveIslandSeconds = 0.45;
cfg.maxInactiveGapSeconds = 0.35;

cfg.envelopeRmsSeconds = 0.10;
cfg.tkeoSmoothSeconds = 0.06;
cfg.envelopeSmoothSeconds = 0.25;
cfg.minCycleSeconds = 0.40;
cfg.maxCycleSeconds = 5.50;
cfg.boundaryPadSeconds = 0.15;
cfg.thresholdActivePercentile = 90;
cfg.thresholdQuietPercentile = 20;
cfg.thresholdFraction = 0.28;
cfg.lowThresholdFraction = 0.45;
cfg.boundaryQuietWindowSeconds = 0.20;
cfg.boundaryQuietCheckPercentile = 25;
cfg.minQuietReturnFraction = 0.85;

cfg.saveFigures = true;
cfg.maxFiguresTotal = inf;

if ~isfolder(cfg.dataDir)
    error('CSV data directory was not found: %s', cfg.dataDir);
end

ensureFolder(cfg.outputDir);
ensureFolder(cfg.segmentRootDir);
ensureFolder(cfg.figureDir);

csvFiles = dir(fullfile(cfg.dataDir, '*.csv'));
if isempty(csvFiles)
    error('No CSV files were found in: %s', cfg.dataDir);
end

[~, order] = sort({csvFiles.name});
csvFiles = csvFiles(order);

summaryRows = struct('record', {}, 'label', {}, 'duration_s', {}, ...
    'total_count', {}, 'detected_count', {}, 'nonstandard_reject_count', {}, ...
    'threshold', {}, 'schedule_offset_s', {}, 'fs', {});
eventRows = struct('record', {}, 'label', {}, 'rep_index', {}, ...
    'start_s', {}, 'end_s', {}, 'duration_s', {}, 'peak_s', {}, ...
    'peak_env', {}, 'mean_env', {}, 'status', {}, 'reason', {}, ...
    'segment_file', {});

fprintf('Detecting EMG motion events in %d records...\n', numel(csvFiles));
figureCount = 0;

for i = 1:numel(csvFiles)
    filePath = fullfile(csvFiles(i).folder, csvFiles(i).name);
    [~, recordName] = fileparts(filePath);
    label = parseLabel(recordName);

    try
        [t, data, fs] = readEmgCsv(filePath, cfg.fallbackFs);
        [envelope, threshold, lowThreshold] = buildFusedEnvelope(data, fs, t, cfg);
        [events, rejected, scheduleOffset, threshold, lowThreshold] = detectMotionEvents(t, envelope, threshold, lowThreshold, cfg);
        [events, rejected] = rejectInvalidEvents(events, rejected, t, envelope, threshold, cfg);
        events = sortEvents(events);
        rejected = sortEvents(rejected);

        recordSegmentDir = fullfile(cfg.segmentRootDir, recordName);
        ensureFolder(recordSegmentDir);
        clearCsvFiles(recordSegmentDir);
        for k = 1:numel(events)
            segFile = sprintf('%s_rep%02d.csv', recordName, events(k).repIndex);
            segPath = fullfile(recordSegmentDir, segFile);
            writeSegmentCsv(segPath, t, data, events(k));
            events(k).segmentFile = fullfile('segmented_records', recordName, segFile);
        end

        summaryRows(end + 1) = makeSummaryRow(recordName, label, t, fs, ...
            events, rejected, threshold, scheduleOffset); %#ok<SAGROW>
        eventRows = appendEventRows(eventRows, recordName, label, events, 'valid');
        eventRows = appendEventRows(eventRows, recordName, label, rejected, 'rejected');

        if cfg.saveFigures && figureCount < cfg.maxFiguresTotal
            saveEventFigure(fullfile(cfg.figureDir, [recordName '_events.png']), ...
                t, data, envelope, threshold, lowThreshold, events, rejected, ...
                scheduleOffset, cfg);
            figureCount = figureCount + 1;
        end

        fprintf('  %-24s label=%s total=%d detected=%d nonstandard_rejected=%d threshold=%.3f\n', ...
            recordName, label, numel(events) + numel(rejected), numel(events), ...
            numel(rejected), threshold);
    catch ME
        warning('Skipping %s: %s', recordName, ME.message);
        for s = 1:numel(ME.stack)
            fprintf('    at %s line %d\n', ME.stack(s).name, ME.stack(s).line);
        end
    end
end

summaryTable = struct2table(summaryRows);
eventTable = struct2table(eventRows);
summaryPath = fullfile(cfg.outputDir, 'summary.csv');
eventPath = fullfile(cfg.outputDir, 'event_boundaries.csv');
writetable(summaryTable, summaryPath);
writetable(eventTable, eventPath);

fprintf('\nDone.\n');
fprintf('Summary: %s\n', summaryPath);
fprintf('Event boundaries: %s\n', eventPath);
fprintf('Segment CSV files: %s\n', cfg.segmentRootDir);
if cfg.saveFigures
    fprintf('Figures: %s\n', cfg.figureDir);
end

% =========================================================================
% Local functions
% =========================================================================

function ensureFolder(folderPath)
    if ~isfolder(folderPath)
        mkdir(folderPath);
    end
end

function clearCsvFiles(folderPath)
    oldFiles = dir(fullfile(folderPath, '*.csv'));
    for i = 1:numel(oldFiles)
        delete(fullfile(oldFiles(i).folder, oldFiles(i).name));
    end
end

function label = parseLabel(recordName)
    token = regexp(recordName, '^(\d+)', 'tokens', 'once');
    if isempty(token)
        label = '';
    else
        label = token{1};
    end
end

function [t, data, fs] = readEmgCsv(filePath, fallbackFs)
    raw = readmatrix(filePath);
    raw = raw(any(isfinite(raw), 2), :);
    raw = raw(:, any(isfinite(raw), 1));
    if size(raw, 2) < 2
        error('CSV must contain at least two numeric columns.');
    end

    if size(raw, 2) >= 3 && isLikelyTimeColumn(raw(:, 1))
        t = raw(:, 1);
        data = raw(:, 2:3);
    else
        data = raw(:, 1:2);
        t = (0:size(data, 1) - 1)' / fallbackFs;
    end

    fs = inferFs(t, fallbackFs);
    data = double(data);
    for ch = 1:size(data, 2)
        x = data(:, ch);
        finiteMask = isfinite(x);
        if any(finiteMask)
            x(~finiteMask) = median(x(finiteMask));
        else
            x(:) = 0;
        end
        data(:, ch) = x;
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
    positiveSteps = dx(dx > 0);
    tf = mean(dx > 0) > 0.95 && ~isempty(positiveSteps) && ...
        median(positiveSteps) > 0 && median(positiveSteps) < 1;
end

function fs = inferFs(t, fallbackFs)
    dt = diff(t(:));
    dt = dt(isfinite(dt) & dt > 0);
    if isempty(dt)
        fs = fallbackFs;
    else
        fs = round(1 / median(dt));
        if ~isfinite(fs) || fs <= 0
            fs = fallbackFs;
        end
    end
end

function [envelope, threshold, lowThreshold] = buildFusedEnvelope(data, fs, t, cfg)
    rmsWin = max(1, round(cfg.envelopeRmsSeconds * fs));
    tkeoWin = max(1, round(cfg.tkeoSmoothSeconds * fs));
    smoothWin = max(1, round(cfg.envelopeSmoothSeconds * fs));

    channelEnv = sqrt(movmean(data .^ 2, rmsWin, 1));
    tkeoEnv = zeros(size(data));
    for ch = 1:size(data, 2)
        x = data(:, ch);
        x = x - median(x);
        psi = zeros(size(x));
        if numel(x) >= 3
            psi(2:end - 1) = x(2:end - 1) .^ 2 - x(1:end - 2) .* x(3:end);
        end
        tkeoEnv(:, ch) = movmean(sqrt(max(psi, 0)), tkeoWin);
    end

    channelNorm = zeros(size(channelEnv));
    tkeoNorm = zeros(size(tkeoEnv));
    for ch = 1:size(channelEnv, 2)
        channelNorm(:, ch) = robustNormalize(channelEnv(:, ch), cfg);
        tkeoNorm(:, ch) = robustNormalize(tkeoEnv(:, ch), cfg);
    end

    rmsEnvelope = mean(channelNorm, 2);
    tkeoEnvelope = mean(tkeoNorm, 2);
    envelope = 0.65 * rmsEnvelope + 0.35 * tkeoEnvelope;
    envelope = movmean(envelope, smoothWin);

    [threshold, lowThreshold] = thresholdsFromEnvelope(envelope, cfg.thresholdFraction, cfg);
end

function y = robustNormalize(x, cfg)
    x = x(:);
    quiet = prctile(x, cfg.thresholdQuietPercentile);
    active = prctile(x, cfg.thresholdActivePercentile);
    scale = active - quiet;
    if ~isfinite(scale) || scale < eps
        scale = max(std(x), eps);
    end
    y = max(0, (x - quiet) / scale);
end

function [threshold, lowThreshold] = thresholdsFromEnvelope(envelope, thresholdFraction, cfg)
    quietMask = envelope <= prctile(envelope, cfg.thresholdQuietPercentile);
    quietValues = envelope(quietMask);
    if isempty(quietValues)
        quietValues = envelope(:);
    end
    baseline = median(quietValues);
    noise = median(abs(quietValues - baseline)) * 1.4826;
    activeLevel = prctile(envelope, cfg.thresholdActivePercentile);

    threshold = baseline + thresholdFraction * max(activeLevel - baseline, eps);
    threshold = max(threshold, baseline + 3 * noise);
    lowThreshold = baseline + cfg.lowThresholdFraction * (threshold - baseline);
end

function [events, rejected, scheduleOffset, threshold, lowThreshold] = detectMotionEvents(t, envelope, threshold, lowThreshold, cfg)
    scheduleOffset = NaN;
    rawEvents = detectUnscheduledBursts(t, envelope, threshold, lowThreshold, cfg);
    rawEvents = sortEvents(rawEvents);

    events = emptyEvents();
    rejected = emptyEvents();
    for i = 1:numel(rawEvents)
        rawEvents(i).repIndex = i;
        rawEvents(i).reason = 'unscheduled_burst_candidate';
        events(end + 1) = rawEvents(i); %#ok<AGROW>
    end
    [events, rejected] = suppressDuplicateSlotEvents(events, rejected, cfg);
end

function events = detectUnscheduledBursts(t, envelope, threshold, lowThreshold, cfg)
    if strcmpi(cfg.detectionMethod, 'gmm')
        active = detectActiveMaskGmm(t, envelope, threshold, cfg);
    else
        active = envelope >= threshold;
    end
    events = maskToEvents(active, t, envelope);
    events = refineEventBoundaries(events, envelope, lowThreshold, cfg, t);
    events = mergeBurstEvents(events, t, envelope, lowThreshold, cfg);
end

function active = detectActiveMaskGmm(t, envelope, threshold, cfg)
    fs = inferFs(t, 2000);
    win = max(3, round(cfg.gmmWindowSeconds * fs));
    step = max(1, round(cfg.gmmStepSeconds * fs));
    n = numel(envelope);
    centers = (1:step:n)';
    feature = zeros(numel(centers), 3);

    for i = 1:numel(centers)
        s = max(1, centers(i) - floor(win / 2));
        e = min(n, centers(i) + floor(win / 2));
        local = envelope(s:e);
        feature(i, 1) = log1p(mean(local));
        feature(i, 2) = log1p(max(local));
        feature(i, 3) = log1p(std(local));
    end

    activeProbability = estimateActiveProbability(feature);
    if isempty(activeProbability)
        active = envelope >= threshold;
        return;
    end

    activeWindow = hysteresisMask(activeProbability, ...
        cfg.gmmActiveOnProbability, cfg.gmmActiveOffProbability);
    activeSample = interp1(centers, double(activeWindow), (1:n)', 'nearest', 'extrap') > 0;
    activeSample = closeShortGaps(activeSample, round(cfg.maxInactiveGapSeconds * fs));
    activeSample = removeShortIslands(activeSample, round(cfg.minActiveIslandSeconds * fs));
    active = activeSample(:);
end

function pActive = estimateActiveProbability(feature)
    pActive = [];
    finiteRows = all(isfinite(feature), 2);
    feature = feature(finiteRows, :);
    if size(feature, 1) < 10 || std(feature(:, 1)) < eps
        return;
    end

    mu = mean(feature, 1);
    sigma = std(feature, [], 1);
    sigma(sigma < eps) = 1;
    z = (feature - mu) ./ sigma;

    try
        opts = statset('MaxIter', 300, 'Display', 'off');
        gm = fitgmdist(z, 2, 'RegularizationValue', 1e-5, ...
            'Replicates', 5, 'Options', opts);
        post = posterior(gm, z);
        [~, activeCluster] = max(gm.mu(:, 1));
        p = post(:, activeCluster);
    catch
        score = feature(:, 1);
        p = rankProbability(score);
    end

    pActive = zeros(size(finiteRows));
    pActive(finiteRows) = p;
end

function p = rankProbability(score)
    [sortedScore, order] = sort(score);
    if numel(sortedScore) < 2 || sortedScore(end) == sortedScore(1)
        p = zeros(size(score));
        return;
    end
    ranks = (0:numel(score) - 1)' / max(numel(score) - 1, 1);
    p = zeros(size(score));
    p(order) = ranks;
end

function mask = hysteresisMask(probability, onThreshold, offThreshold)
    mask = false(size(probability));
    active = false;
    for i = 1:numel(probability)
        if active
            if probability(i) <= offThreshold
                active = false;
            end
        elseif probability(i) >= onThreshold
            active = true;
        end
        mask(i) = active;
    end
end

function mask = closeShortGaps(mask, maxGapSamples)
    if maxGapSamples <= 0
        return;
    end
    gaps = maskToRanges(~mask);
    for i = 1:size(gaps, 1)
        touchesEdge = gaps(i, 1) == 1 || gaps(i, 2) == numel(mask);
        if ~touchesEdge && (gaps(i, 2) - gaps(i, 1) + 1) <= maxGapSamples
            mask(gaps(i, 1):gaps(i, 2)) = true;
        end
    end
end

function mask = removeShortIslands(mask, minIslandSamples)
    if minIslandSamples <= 1
        return;
    end
    islands = maskToRanges(mask);
    for i = 1:size(islands, 1)
        if (islands(i, 2) - islands(i, 1) + 1) < minIslandSamples
            mask(islands(i, 1):islands(i, 2)) = false;
        end
    end
end

function ranges = maskToRanges(mask)
    mask = mask(:);
    if ~any(mask)
        ranges = zeros(0, 2);
        return;
    end
    edges = diff([false; mask; false]);
    starts = find(edges == 1);
    ends = find(edges == -1) - 1;
    ranges = [starts ends];
end

function idx = findBoundaryLeft(envelope, peakIdx, leftLimit, lowThreshold)
    peakIdx = peakIdx(1);
    leftLimit = leftLimit(1);
    idx = peakIdx;
    while (idx > leftLimit) && (envelope(idx) > lowThreshold)
        idx = idx - 1;
    end
    if (idx == leftLimit) && (envelope(idx) > lowThreshold)
        [~, localMin] = min(envelope(leftLimit:peakIdx));
        idx = leftLimit + localMin - 1;
    end
end

function idx = findBoundaryRight(envelope, peakIdx, rightLimit, lowThreshold)
    peakIdx = peakIdx(1);
    rightLimit = rightLimit(1);
    idx = peakIdx;
    while (idx < rightLimit) && (envelope(idx) > lowThreshold)
        idx = idx + 1;
    end
    if (idx == rightLimit) && (envelope(idx) > lowThreshold)
        [~, localMin] = min(envelope(peakIdx:rightLimit));
        idx = peakIdx + localMin - 1;
    end
end

function events = maskToEvents(mask, t, envelope)
    events = emptyEvents();
    if ~any(mask)
        return;
    end

    edges = diff([false; mask(:); false]);
    starts = find(edges == 1);
    ends = find(edges == -1) - 1;
    for i = 1:numel(starts)
        events(end + 1) = makeEvent(starts(i), ends(i), t, envelope); %#ok<AGROW>
    end
end

function events = refineEventBoundaries(events, envelope, lowThreshold, cfg, t)
    pad = round(cfg.boundaryPadSeconds * inferFs(t, 2000));
    for i = 1:numel(events)
        s = findBoundaryLeft(envelope, events(i).peakIdx, 1, lowThreshold);
        e = findBoundaryRight(envelope, events(i).peakIdx, numel(envelope), lowThreshold);
        s = max(1, s - pad);
        e = min(numel(envelope), e + pad);
        events(i) = makeEvent(s, e, t, envelope);
    end
end

function events = mergeBurstEvents(events, t, envelope, lowThreshold, cfg)
    if numel(events) < 2
        return;
    end

    events = sortEvents(events);
    merged = events(1);
    for i = 2:numel(events)
        gap = events(i).startTime - merged(end).endTime;
        gap = min(gap(:));
        gapIdx = merged(end).endIdx:events(i).startIdx;
        valleyReturnedQuiet = any(envelope(gapIdx) <= lowThreshold);
        sameCycle = (gap <= cfg.mergeQuietGapSeconds) && ~valleyReturnedQuiet;
        if sameCycle
            startIdx = merged(end).startIdx;
            endIdx = events(i).endIdx;
            merged(end) = makeEvent(startIdx, endIdx, t, envelope);
        else
            merged(end + 1) = events(i); %#ok<AGROW>
        end
    end
    events = merged;
end

function [events, rejected] = suppressDuplicateSlotEvents(events, rejected, cfg)
    if numel(events) < 2
        return;
    end

    [~, order] = sort([events.peakValue], 'descend');
    keep = false(1, numel(events));
    duplicate = false(1, numel(events));
    for n = 1:numel(order)
        i = order(n);
        isDuplicate = false;
        keptIdx = find(keep);
        for kk = 1:numel(keptIdx)
            k = keptIdx(kk);
            peakDistance = abs(events(i).peakTime - events(k).peakTime);
            peakTooClose = any(peakDistance(:) < cfg.minPeakSeparationSeconds);
            overlap = max(0, min(events(i).endTime, events(k).endTime) - ...
                max(events(i).startTime, events(k).startTime));
            overlap = max(overlap(:));
            shorterDuration = min(events(i).endTime - events(i).startTime, ...
                events(k).endTime - events(k).startTime);
            shorterDuration = min(shorterDuration(:));
            overlapTooLarge = (shorterDuration > 0) && (overlap / shorterDuration > 0.50);
            if peakTooClose || overlapTooLarge
                isDuplicate = true;
                break;
            end
        end

        if isDuplicate
            duplicate(i) = true;
        else
            keep(i) = true;
        end
    end

    duplicateIdx = find(duplicate);
    for i = duplicateIdx
        events(i).reason = 'duplicate_slot_peak';
        rejected(end + 1) = events(i); %#ok<AGROW>
    end
    events = events(keep);
end

function [validEvents, rejected] = rejectInvalidEvents(events, rejected, t, envelope, threshold, cfg)
    validEvents = emptyEvents();
    for i = 1:numel(events)
        ev = events(i);
        duration = ev.endTime - ev.startTime;
        duration = max(duration(:));
        edgeWindow = cfg.boundaryQuietWindowSeconds;
        beforeOk = hasQuietBoundary(envelope, t, ev.startTime, ...
            min(ev.startTime + edgeWindow, ev.endTime), cfg);
        afterOk = hasQuietBoundary(envelope, t, ...
            max(ev.endTime - edgeWindow, ev.startTime), ev.endTime, cfg);
        beforeOk = all(beforeOk(:));
        afterOk = all(afterOk(:));

        peakValue = max(ev.peakValue(:));
        endTime = max(ev.endTime(:));

        if duration < cfg.minCycleSeconds
            ev.reason = 'too_short';
            rejected(end + 1) = ev; %#ok<AGROW>
        elseif duration > cfg.maxCycleSeconds
            ev.reason = 'too_long_or_multiple_cycles';
            rejected(end + 1) = ev; %#ok<AGROW>
        elseif peakValue < cfg.minPeakToThresholdRatio * threshold
            ev.reason = 'weak_peak_candidate';
            rejected(end + 1) = ev; %#ok<AGROW>
        elseif (endTime >= t(end) - 0.02) && ~afterOk
            ev.reason = 'record_ended_before_return_to_quiet';
            rejected(end + 1) = ev; %#ok<AGROW>
        elseif ~beforeOk || ~afterOk
            ev.reason = 'boundary_not_returned_to_quiet';
            rejected(end + 1) = ev; %#ok<AGROW>
        else
            ev.reason = 'valid_complete_cycle';
            validEvents(end + 1) = ev; %#ok<AGROW>
        end
    end
end

function ok = hasQuietBoundary(envelope, t, startTime, endTime, cfg)
    idx = t >= startTime & t <= endTime;
    if sum(idx) < max(5, round(0.05 * inferFs(t, 2000)))
        ok = false;
        return;
    end
    local = envelope(idx);
    quietLevel = prctile(envelope, cfg.thresholdQuietPercentile);
    activeLevel = prctile(envelope, cfg.thresholdActivePercentile);
    quietLimit = quietLevel + cfg.minQuietReturnFraction * ...
        (quietLevel + cfg.lowThresholdFraction * max(activeLevel - quietLevel, eps) - quietLevel);
    ok = prctile(local, cfg.boundaryQuietCheckPercentile) <= quietLimit;
end

function events = sortEvents(events)
    if isempty(events)
        return;
    end
    [~, order] = sort([events.startTime]);
    events = events(order);
end

function ev = makeEvent(startIdx, endIdx, t, envelope)
    localEnvelope = envelope(startIdx:endIdx);
    localEnvelope = localEnvelope(:);
    [peakValue, localPeak] = max(localEnvelope);
    peakIdx = startIdx + localPeak - 1;
    ev = struct();
    ev.startIdx = startIdx;
    ev.endIdx = endIdx;
    ev.peakIdx = peakIdx;
    ev.startTime = t(startIdx);
    ev.endTime = t(endIdx);
    ev.peakTime = t(peakIdx);
    ev.peakValue = peakValue;
    ev.meanValue = mean(localEnvelope);
    ev.repIndex = 0;
    ev.reason = '';
    ev.segmentFile = '';
end

function events = emptyEvents()
    events = struct('startIdx', {}, 'endIdx', {}, 'peakIdx', {}, ...
        'startTime', {}, 'endTime', {}, 'peakTime', {}, 'peakValue', {}, ...
        'meanValue', {}, 'repIndex', {}, 'reason', {}, 'segmentFile', {});
end

function writeSegmentCsv(segPath, t, data, ev)
    idx = ev.startIdx:ev.endIdx;
    T = table(t(idx), data(idx, 1), data(idx, 2), ...
        'VariableNames', {'time_s', 'EMG_ch1_uV', 'EMG_ch2_uV'});
    writetable(T, segPath);
end

function row = makeSummaryRow(recordName, label, t, fs, events, rejected, threshold, scheduleOffset)
    row = struct();
    row.record = recordName;
    row.label = label;
    row.duration_s = t(end) - t(1);
    row.total_count = numel(events) + numel(rejected);
    row.detected_count = numel(events);
    row.nonstandard_reject_count = numel(rejected);
    row.threshold = threshold;
    row.schedule_offset_s = scheduleOffset;
    row.fs = fs;
end

function eventRows = appendEventRows(eventRows, recordName, label, events, status)
    for i = 1:numel(events)
        row = struct();
        row.record = recordName;
        row.label = label;
        row.rep_index = events(i).repIndex;
        row.start_s = events(i).startTime;
        row.end_s = events(i).endTime;
        row.duration_s = events(i).endTime - events(i).startTime;
        row.peak_s = events(i).peakTime;
        row.peak_env = events(i).peakValue;
        row.mean_env = events(i).meanValue;
        row.status = status;
        row.reason = events(i).reason;
        row.segment_file = events(i).segmentFile;
        eventRows(end + 1) = row; %#ok<AGROW>
    end
end

function saveEventFigure(figPath, t, data, envelope, threshold, lowThreshold, events, rejected, scheduleOffset, cfg)
    fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 720]);

    subplot(2, 1, 1);
    plot(t, data(:, 1), 'Color', [0.1 0.35 0.8]); hold on;
    plot(t, data(:, 2), 'Color', [0.8 0.25 0.1]);
    for i = 1:numel(events)
        patch([events(i).startTime events(i).endTime events(i).endTime events(i).startTime], ...
            ylimForPatch(data), [0.3 0.8 0.4], 'FaceAlpha', 0.12, 'EdgeColor', 'none');
    end
    title('Synchronized two-channel EMG with valid motion cycles');
    xlabel('Time (s)');
    ylabel('EMG (\muV)');
    legend({'Ch1', 'Ch2'}, 'Location', 'best');
    grid on;

    subplot(2, 1, 2);
    plot(t, envelope, 'k', 'LineWidth', 1.3); hold on;
    yline(threshold, 'r--', 'Threshold');
    yline(lowThreshold, 'Color', [0.8 0.45 0.1], 'LineStyle', '--');
    for i = 1:numel(events)
        xline(events(i).startTime, 'g-', sprintf('rep%d start', events(i).repIndex));
        xline(events(i).endTime, 'g-');
    end
    for i = 1:numel(rejected)
        xline(rejected(i).startTime, 'm--', 'rejected');
        xline(rejected(i).endTime, 'm--');
    end
    title('Fused EMG envelope, adaptive threshold, and unscheduled event boundaries');
    xlabel('Time (s)');
    ylabel('Normalized envelope');
    grid on;

    exportgraphics(fig, figPath, 'Resolution', 160);
    close(fig);
end

function y = ylimForPatch(data)
    low = min(data(:));
    high = max(data(:));
    y = [low low high high];
end
