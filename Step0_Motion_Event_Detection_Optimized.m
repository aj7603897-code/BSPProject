% =========================================================================
% File: Step0_Motion_Event_Detection_Optimized.m
% Purpose: Detect, count, and segment complete EMG motion cycles.
%
% Experiment protocol:
%   - Each record is about 30 s.
%   - Five complete repetitions are expected.
%   - Repetitions start around 5, 10, 15, 20, and 25 s.
%   - The whole action schedule can shift if recording started early/late.
%   - Complete repetitions after 30 s are kept when their boundaries return
%     to quiet activity before the record ends.
%
% Outputs:
%   EventDetectionResults/summary.csv
%   EventDetectionResults/event_boundaries.csv
%   EventDetectionResults/segmented_records/<record>/<record>_repXX.csv
%   EventDetectionResults/figures/<record>_events.png
% =========================================================================

clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end

cfg.dataDir = fullfile(scriptDir, 'csv');
cfg.outputDir = fullfile(scriptDir, 'EventDetectionResults_Optimized');
cfg.segmentRootDir = fullfile(cfg.outputDir, 'segmented_records');
cfg.figureDir = fullfile(cfg.outputDir, 'figures');
cfg.fallbackFs = 2000;

cfg.expectedReps = 5;
cfg.maxExtraReps = 3;
cfg.firstRepTime = 5;
cfg.repInterval = 5;
cfg.offsetSearchMin = -2.5;
cfg.offsetSearchMax = 6.0;
cfg.offsetSearchStep = 0.10;
cfg.slotHalfWidth = 2.45;
cfg.minPeakSeparationSeconds = 2.20;
cfg.valleySearchFraction = 0.45;

cfg.envelopeRmsSeconds = 0.10;
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
cfg.extraRepMinPeakThresholdRatio = 1.80;
cfg.rejectElevatedBoundaries = false;

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
    'expected_count', {}, 'valid_count', {}, 'rejected_count', {}, ...
    'count_error', {}, 'threshold', {}, 'schedule_offset_s', {}, 'fs', {});
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
        [events, rejected, scheduleOffset] = detectMotionEvents(t, envelope, threshold, lowThreshold, cfg);
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
            cfg.expectedReps, events, rejected, threshold, scheduleOffset); %#ok<SAGROW>
        eventRows = appendEventRows(eventRows, recordName, label, events, 'valid');
        eventRows = appendEventRows(eventRows, recordName, label, rejected, 'rejected');

        if cfg.saveFigures && figureCount < cfg.maxFiguresTotal
            saveEventFigure(fullfile(cfg.figureDir, [recordName '_events.png']), ...
                t, data, envelope, threshold, lowThreshold, events, rejected, ...
                scheduleOffset, cfg);
            figureCount = figureCount + 1;
        end

        fprintf('  %-24s label=%s valid=%d/%d rejected=%d offset=%+.2fs threshold=%.3f\n', ...
            recordName, label, numel(events), cfg.expectedReps, numel(rejected), ...
            scheduleOffset, threshold);
    catch ME
        warning('Skipping %s: %s', recordName, ME.message);
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
    smoothWin = max(1, round(cfg.envelopeSmoothSeconds * fs));

    channelEnv = sqrt(movmean(data .^ 2, rmsWin, 1));
    channelNorm = zeros(size(channelEnv));
    for ch = 1:size(channelEnv, 2)
        x = channelEnv(:, ch);
        quiet = prctile(x, cfg.thresholdQuietPercentile);
        active = prctile(x, cfg.thresholdActivePercentile);
        scale = active - quiet;
        if ~isfinite(scale) || scale < eps
            scale = max(std(x), eps);
        end
        channelNorm(:, ch) = max(0, (x - quiet) / scale);
    end

    envelope = mean(channelNorm, 2);
    envelope = movmean(envelope, smoothWin);

    quietMask = t <= max(0.5, cfg.firstRepTime - 1);
    if sum(quietMask) < fs
        quietMask = envelope <= prctile(envelope, cfg.thresholdQuietPercentile);
    end
    quietValues = envelope(quietMask);
    baseline = median(quietValues);
    noise = median(abs(quietValues - baseline)) * 1.4826;
    activeLevel = prctile(envelope, cfg.thresholdActivePercentile);

    threshold = baseline + cfg.thresholdFraction * max(activeLevel - baseline, eps);
    threshold = max(threshold, baseline + 3 * noise);
    lowThreshold = baseline + cfg.lowThresholdFraction * (threshold - baseline);
end

function [events, rejected, scheduleOffset] = detectMotionEvents(t, envelope, threshold, lowThreshold, cfg)
    scheduleOffset = estimateScheduleOffset(t, envelope, threshold, cfg);
    totalSlots = cfg.expectedReps + cfg.maxExtraReps;
    slotCenters = cfg.firstRepTime + scheduleOffset + (0:totalSlots - 1) * cfg.repInterval;

    rejected = emptyEvents();
    candidates = emptySlotCandidates();
    for rep = 1:numel(slotCenters)
        slotStart = slotCenters(rep) - cfg.slotHalfWidth;
        slotEnd = slotCenters(rep) + cfg.slotHalfWidth;
        if slotEnd < t(1) || slotStart > t(end)
            continue;
        end

        [candidate, hasCandidate] = detectSlotPeak(t, envelope, threshold, ...
            slotStart, slotEnd, rep);
        if hasCandidate
            candidates(end + 1) = candidate; %#ok<AGROW>
        end
    end
    events = buildEventsFromSlotValleys(t, envelope, threshold, lowThreshold, candidates, cfg);
    [events, rejected] = suppressDuplicateSlotEvents(events, rejected, cfg);
end

function scheduleOffset = estimateScheduleOffset(t, envelope, threshold, cfg)
    offsets = cfg.offsetSearchMin:cfg.offsetSearchStep:cfg.offsetSearchMax;
    if isempty(offsets)
        scheduleOffset = 0;
        return;
    end

    bestScore = -inf;
    scheduleOffset = 0;
    for offset = offsets
        score = 0;
        weightedHitCount = 0;
        for rep = 1:cfg.expectedReps
            center = cfg.firstRepTime + offset + (rep - 1) * cfg.repInterval;
            idx = t >= center - cfg.slotHalfWidth & t <= center + cfg.slotHalfWidth;
            if ~any(idx)
                continue;
            end
            slotEnvelope = envelope(idx);
            slotTime = t(idx);
            [peak, localPeakIdx] = max(slotEnvelope);
            peakTime = slotTime(localPeakIdx);
            centerCloseness = max(0, 1 - (abs(peakTime - center) / cfg.slotHalfWidth) ^ 2);
            if peak >= threshold
                weightedHitCount = weightedHitCount + centerCloseness;
                score = score + centerCloseness * min(peak / max(threshold, eps), 4);
            else
                score = score + 0.15 * centerCloseness * peak / max(threshold, eps);
            end
        end

        score = score + 8 * weightedHitCount - 0.20 * abs(offset);
        if score > bestScore
            bestScore = score;
            scheduleOffset = offset;
        end
    end
end

function [candidate, hasCandidate] = detectSlotPeak(t, envelope, threshold, slotStart, slotEnd, rep)
    slotIdx = find(t >= slotStart & t <= slotEnd);
    hasCandidate = false;
    candidate = emptySlotCandidates();
    if isempty(slotIdx)
        return;
    end

    [peakValue, localPeak] = max(envelope(slotIdx));
    if peakValue < threshold
        return;
    end

    peakIdx = slotIdx(localPeak);
    candidate = struct('repIndex', rep, 'slotStartIdx', slotIdx(1), ...
        'slotEndIdx', slotIdx(end), 'peakIdx', peakIdx, ...
        'peakValue', peakValue);
    hasCandidate = true;
end

function events = buildEventsFromSlotValleys(t, envelope, threshold, lowThreshold, candidates, cfg)
    events = emptyEvents();
    if isempty(candidates)
        return;
    end

    [~, order] = sort([candidates.peakIdx]);
    candidates = candidates(order);
    pad = round(cfg.boundaryPadSeconds * inferFs(t, 2000));

    for i = 1:numel(candidates)
        leftLimit = candidates(i).slotStartIdx;
        rightLimit = candidates(i).slotEndIdx;

        if i > 1
            leftSeparator = findValleyBetweenPeaks(envelope, candidates(i - 1).peakIdx, ...
                candidates(i).peakIdx, cfg);
            leftLimit = min(candidates(i).peakIdx, leftSeparator + 1);
        end
        if i < numel(candidates)
            rightSeparator = findValleyBetweenPeaks(envelope, candidates(i).peakIdx, ...
                candidates(i + 1).peakIdx, cfg);
            rightLimit = max(candidates(i).peakIdx, rightSeparator);
        end

        [startIdx, endIdx] = findCycleBoundsWithinSeparators(envelope, ...
            threshold, lowThreshold, leftLimit, rightLimit);
        startIdx = max(leftLimit, startIdx - pad);
        endIdx = min(rightLimit, endIdx + pad);

        if endIdx > startIdx
            ev = makeEvent(startIdx, endIdx, t, envelope);
            ev.repIndex = candidates(i).repIndex;
            ev.reason = 'slot_valley_candidate';
            events(end + 1) = ev; %#ok<AGROW>
        end
    end
end

function [startIdx, endIdx] = findCycleBoundsWithinSeparators(envelope, threshold, lowThreshold, leftLimit, rightLimit)
    if rightLimit <= leftLimit
        startIdx = leftLimit;
        endIdx = rightLimit;
        return;
    end

    localIdx = leftLimit:rightLimit;
    activeLocal = find(envelope(localIdx) >= threshold);
    if isempty(activeLocal)
        [~, localPeak] = max(envelope(localIdx));
        peakIdx = localIdx(localPeak);
        startIdx = findBoundaryLeft(envelope, peakIdx, leftLimit, lowThreshold);
        endIdx = findBoundaryRight(envelope, peakIdx, rightLimit, lowThreshold);
        return;
    end

    firstActiveIdx = localIdx(activeLocal(1));
    lastActiveIdx = localIdx(activeLocal(end));
    startIdx = findBoundaryLeft(envelope, firstActiveIdx, leftLimit, lowThreshold);
    endIdx = findBoundaryRight(envelope, lastActiveIdx, rightLimit, lowThreshold);
end

function valleyIdx = findValleyBetweenPeaks(envelope, leftPeakIdx, rightPeakIdx, cfg)
    if rightPeakIdx <= leftPeakIdx + 1
        valleyIdx = leftPeakIdx;
        return;
    end

    spanStart = leftPeakIdx + 1;
    spanEnd = rightPeakIdx - 1;
    spanLength = spanEnd - spanStart + 1;
    trim = floor((1 - cfg.valleySearchFraction) * spanLength / 2);
    searchStart = min(spanEnd, spanStart + trim);
    searchEnd = max(searchStart, spanEnd - trim);
    [~, localMin] = min(envelope(searchStart:searchEnd));
    valleyIdx = searchStart + localMin - 1;
end

function idx = findBoundaryLeft(envelope, peakIdx, leftLimit, lowThreshold)
    idx = peakIdx;
    while idx > leftLimit && envelope(idx) > lowThreshold
        idx = idx - 1;
    end
    if idx == leftLimit && envelope(idx) > lowThreshold
        [~, localMin] = min(envelope(leftLimit:peakIdx));
        idx = leftLimit + localMin - 1;
    end
end

function idx = findBoundaryRight(envelope, peakIdx, rightLimit, lowThreshold)
    idx = peakIdx;
    while idx < rightLimit && envelope(idx) > lowThreshold
        idx = idx + 1;
    end
    if idx == rightLimit && envelope(idx) > lowThreshold
        [~, localMin] = min(envelope(peakIdx:rightLimit));
        idx = peakIdx + localMin - 1;
    end
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
        for k = keptIdx
            peakTooClose = abs(events(i).peakTime - events(k).peakTime) < ...
                cfg.minPeakSeparationSeconds;
            overlap = max(0, min(events(i).endTime, events(k).endTime) - ...
                max(events(i).startTime, events(k).startTime));
            shorterDuration = min(events(i).endTime - events(i).startTime, ...
                events(k).endTime - events(k).startTime);
            overlapTooLarge = shorterDuration > 0 && overlap / shorterDuration > 0.50;
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
        edgeWindow = cfg.boundaryQuietWindowSeconds;
        beforeOk = hasQuietBoundary(envelope, t, ev.startTime, ...
            min(ev.startTime + edgeWindow, ev.endTime), cfg);
        afterOk = hasQuietBoundary(envelope, t, ...
            max(ev.endTime - edgeWindow, ev.startTime), ev.endTime, cfg);

        if ev.repIndex < 1 || ev.repIndex > cfg.expectedReps + cfg.maxExtraReps
            ev.reason = 'outside_expected_rep_windows';
            rejected(end + 1) = ev; %#ok<AGROW>
        elseif duration < cfg.minCycleSeconds
            ev.reason = 'too_short';
            rejected(end + 1) = ev; %#ok<AGROW>
        elseif duration > cfg.maxCycleSeconds
            ev.reason = 'too_long_or_multiple_cycles';
            rejected(end + 1) = ev; %#ok<AGROW>
        elseif ev.repIndex > cfg.expectedReps && ...
                ev.peakValue < cfg.extraRepMinPeakThresholdRatio * threshold
            ev.reason = 'weak_extra_rep_candidate';
            rejected(end + 1) = ev; %#ok<AGROW>
        elseif ev.endTime >= t(end) - 0.02 && ~afterOk
            ev.reason = 'record_ended_before_return_to_quiet';
            rejected(end + 1) = ev; %#ok<AGROW>
        elseif cfg.rejectElevatedBoundaries && (~beforeOk || ~afterOk)
            ev.reason = 'boundary_not_returned_to_quiet';
            rejected(end + 1) = ev; %#ok<AGROW>
        else
            if ~beforeOk || ~afterOk
                ev.reason = 'valid_with_elevated_boundary';
            else
                ev.reason = 'valid_complete_cycle';
            end
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
    [peakValue, localPeak] = max(envelope(startIdx:endIdx));
    peakIdx = startIdx + localPeak - 1;
    ev = struct();
    ev.startIdx = startIdx;
    ev.endIdx = endIdx;
    ev.peakIdx = peakIdx;
    ev.startTime = t(startIdx);
    ev.endTime = t(endIdx);
    ev.peakTime = t(peakIdx);
    ev.peakValue = peakValue;
    ev.meanValue = mean(envelope(startIdx:endIdx));
    ev.repIndex = 0;
    ev.reason = '';
    ev.segmentFile = '';
end

function events = emptyEvents()
    events = struct('startIdx', {}, 'endIdx', {}, 'peakIdx', {}, ...
        'startTime', {}, 'endTime', {}, 'peakTime', {}, 'peakValue', {}, ...
        'meanValue', {}, 'repIndex', {}, 'reason', {}, 'segmentFile', {});
end

function candidates = emptySlotCandidates()
    candidates = struct('repIndex', {}, 'slotStartIdx', {}, 'slotEndIdx', {}, ...
        'peakIdx', {}, 'peakValue', {});
end

function writeSegmentCsv(segPath, t, data, ev)
    idx = ev.startIdx:ev.endIdx;
    T = table(t(idx), data(idx, 1), data(idx, 2), ...
        'VariableNames', {'time_s', 'EMG_ch1_uV', 'EMG_ch2_uV'});
    writetable(T, segPath);
end

function row = makeSummaryRow(recordName, label, t, fs, expectedCount, events, rejected, threshold, scheduleOffset)
    row = struct();
    row.record = recordName;
    row.label = label;
    row.duration_s = t(end) - t(1);
    row.expected_count = expectedCount;
    row.valid_count = numel(events);
    row.rejected_count = numel(rejected);
    row.count_error = numel(events) - expectedCount;
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
    expectedTimes = cfg.firstRepTime + scheduleOffset + ...
        (0:cfg.expectedReps + cfg.maxExtraReps - 1) * cfg.repInterval;
    for i = 1:numel(expectedTimes)
        if expectedTimes(i) >= t(1) && expectedTimes(i) <= t(end)
            xline(expectedTimes(i), ':', 'Color', [0.4 0.4 0.4]);
        end
    end
    title('Fused EMG envelope, adaptive threshold, and event boundaries');
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
