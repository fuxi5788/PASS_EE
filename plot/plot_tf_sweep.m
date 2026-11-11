%% plot_tf_sweep.m
%  Energy efficiency versus frame length: Proposed / ADMM / DSD / BGAT.
%
%  The FPA column was removed on 2026-09-13 (the report and both CSVs now carry
%  MOVING-ANTENNA schemes only: ADMM / DSD / SAC / BGAT), so nothing here
%  refers to it any more.  The report parser below picks its series up BY NAME
%  from the RESULTS table header, so adding or dropping a column never shifts
%  the others.
%
%  Data: the sweep text report written by run_tf_sweep.py (SOURCE = 'txt',
%        default) -- its "RESULTS (mean EE, Mbit/J)" table is parsed, column
%        names are taken from the report header, and PMOTOR is read for the
%        output file name;
%        or tf_sweep_data.csv (SOURCE = 'csv').
%
%  Output: <mulsim>/plot/fig_tf_sweep[_PMOTOR<x>]_4algo.eps (+ .png), y-axis
%        zoomed so the small gaps between the schemes stay readable.
%
%  Style follows plot_eposide_reward.m: white figure, r-o / b-s / m-^ / c-d
%  series, Times New Roman, minor ticks, Painter EPS.
%
%  Verified with MATLAB R2018b.

clc; close all;

%% ------------------------------ settings ---------------------------------
SOURCE  = 'csv';                              % self-contained: reads the local
                                              % tf_sweep_data.csv (0.3-1.4 s)
if ~exist('TXT', 'var') || isempty(TXT)
    TXT = 'tf_sweep_report_PMOTOR1.0.txt';    % caller may point elsewhere
end
CSV     = 'tf_sweep_data.csv';                % used when SOURCE = 'csv'
OUTNAME = 'fig_tf_sweep';                     % <OUTNAME><_PMOTOR..><SUFFIX>
SUFFIX  = '_4algo';                           % Proposed / ADMM / DSD / BGAT

XLAB = 'Frame duration (s)';
YLAB = 'Energy efficiency (bps/Hz/J)';

% Plot order, legend text and line spec (colour + marker).
ALGOS  = {'SAC',  'BGAT',  'DSD',  'ADMM',};
LABELS = {'GNN-SAC','BGAT', 'DSD', 'ADMM'};
SPEC   = {'r-o', 'c-d', 'm-^', 'b-s' };

% ADMM 9.08 -> SAC 10.56: zoom onto the four moving schemes (no FPA ramp).
YLIM = [9.0 10.6];
XLIM = [0.6 1.2];
%% ------------------------------- read ------------------------------------
Y = cell(1, numel(ALGOS));
pmotor = NaN;
tag = '';

if strcmpi(SOURCE, 'txt')
    if ~exist(TXT, 'file')
        error('plot_tf_sweep:missingReport', ...
              'report not found: %s  (run: python run_tf_sweep.py)', TXT);
    end
    fid = fopen(TXT, 'r');
    lines = {};
    while true
        ln = fgetl(fid);
        if ~ischar(ln)
            break
        end
        lines{end + 1} = ln;                                   %#ok<SAGROW>
    end
    fclose(fid);

    % PMOTOR, for the output file name
    for i = 1:numel(lines)
        tok = regexp(lines{i}, 'PMOTOR\s*=\s*([0-9.]+)', 'tokens', 'once');
        if ~isempty(tok)
            pmotor = str2double(tok{1});
            break
        end
    end
    if ~isnan(pmotor)
        tag = sprintf('_PMOTOR%.1f', pmotor);
    end

    % the RESULTS table
    i0 = 0;
    for i = 1:numel(lines)
        if ~isempty(strfind(lines{i}, 'RESULTS (mean EE'))    %#ok<STREMP>
            i0 = i;
            break
        end
    end
    if i0 == 0
        error('plot_tf_sweep:noResults', ...
              'no "RESULTS (mean EE" section found in %s', TXT);
    end
    cols = strsplit(strtrim(lines{i0 + 1}));
    nCol = numel(cols);
    M = [];
    for i = i0 + 2:numel(lines)
        ln = strtrim(lines{i});
        if isempty(ln)
            break
        end
        v = sscanf(ln, '%f');
        if numel(v) < nCol
            break
        end
        M = [M; v(1:nCol).'];                                  %#ok<AGROW>
    end
    if isempty(M)
        error('plot_tf_sweep:emptyTable', 'no data rows after %s', TXT);
    end
    x = M(:, 1);
    for a = 1:numel(ALGOS)
        k = find(strcmpi(cols, ALGOS{a}), 1);
        if ~isempty(k)
            Y{a} = M(:, k);
        end
    end
    fprintf('source : %s\n', TXT);

else
    try
        T = readtable(CSV, 'VariableNamingRule', 'preserve');   % R2019b+
    catch
        T = readtable(CSV);                                     % older releases
    end
    vn = T.Properties.VariableNames;
    algcol = T.(vn{1});
    if iscategorical(algcol)
        algcol = cellstr(algcol);
    end
    x = [];
    for a = 1:numel(ALGOS)
        keep = strcmp(algcol, ALGOS{a});
        if ~any(keep)
            continue
        end
        xv = double(T.(vn{2})(keep));
        yv = double(T.(vn{3})(keep));
        g = ~isnan(xv) & ~isnan(yv);
        [xv, k] = sort(xv(g));
        Y{a} = yv(g);
        Y{a} = Y{a}(k);
        x = union(x, xv);
    end
    fprintf('source : %s\n', CSV);
end

x = x(:);
if ~isnan(pmotor)
    fprintf('PMOTOR : %.2f W\n', pmotor);
end

%% ------------------------------ plot -------------------------------------
% Resolve the fig directory relative to THIS script (not the CWD), so the
% figures land in <mulsim>/plot regardless of where they are run from.
thisdir = fileparts(mfilename('fullpath'));
figdir  = fullfile(thisdir);
if ~exist(figdir, 'dir')
    mkdir(figdir);
end

figure('Color', 'w', 'Position', [100 100 560 420]);
hold on; grid on; box on;

h = gobjects(1, numel(ALGOS));
lab = cell(1, numel(ALGOS));
n = 0;
for a = 1:numel(ALGOS)
    if isempty(Y{a})                       % column missing from the report
        continue
    end
    n = n + 1;
    h(n) = plot(x, Y{a}, SPEC{a}, 'LineWidth', 1.5, 'MarkerSize', 5);
    lab{n} = LABELS{a};
end
if n == 0
    error('plot_tf_sweep:noData', 'no algorithm column found in %s', TXT);
end

set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, ...
    'XMinorTick', 'on', 'YMinorTick', 'on');
xlabel(XLAB);
ylabel(YLAB);
legend(h(1:n), lab(1:n), 'Location', 'southeast');
xticks(x);
xlim(XLIM);
ylim(YLIM);
xticklabels(compose('%.1f', xticks));
yticklabels(compose('%.1f', yticks));

%%%%---eps---%%%%
set(gcf, 'PaperPositionMode', 'auto');
outpath = fullfile(figdir, [OUTNAME '.eps']);
print(gcf, outpath, '-depsc2', '-painters');
% print(gcf, fullfile(figdir, [OUTNAME tag SUFFIX '.png']), '-dpng', '-r300');
fprintf('Saved: %s\n', outpath);
