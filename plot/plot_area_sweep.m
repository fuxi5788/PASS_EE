%% plot_area_sweep.m
%  Energy efficiency versus the scale of the service area.
%
%  Data: area_sweep_data.csv (long layout: algo,scale,ee_mean,ee_std,n,src)
%        written by run_area_sweep.py.  The service area is the SQUARE of side
%        D = 40*sqrt(scale) m, so scale = 1.0 is the 40 m x 40 m reference; the
%        scale = 1.0 row is NOT recomputed by the runner -- it is read from
%        n_sweep_data.csv (N = 4, Tframe = 1.0), i.e. the same point as in the
%        EE-vs-Tframe and EE-vs-N figures, and is marked src = 'reused'.
%        Only the mean EE (column 3) is plotted; std and n are not used here.
%
%  Output: <mulsim>/plot/fig_area_sweep.eps (+ .png).
%
%  Style follows plot_n_sweep.m: white figure, Times New Roman, minor ticks,
%  Painter EPS, VALUES ONLY (no error bars).
%
%  Expected shape: every scheme loses EE as the service area grows (users get
%  farther from the waveguides), with the proposed GNN-SAC on top at every
%  scale.

clc; close all;

%% ------------------------------ settings ---------------------------------
CSV     = 'area_sweep_data.csv';
OUTNAME = 'fig_area_sweep';

XLAB = 'Scale size of the service area';
YLAB = 'Energy efficiency (bits/Hz/J)';

ALGOS  = {'SAC',  'BGAT', 'DSD',  'ADMM', 'FPA'};
LABELS = {'GNN-SAC', 'BGAT', 'DSD', 'ADMM', 'FPA'};
SPEC   = {'r-o', 'c-d', 'm-^', 'b-s', 'k--*'};

%% ------------------------------- read ------------------------------------
if ~exist(CSV, 'file')
    error('plot_area_sweep:missingCsv', 'not found: %s (run: python run_area_sweep.py --write-csv)', CSV);
end
T = readtable(CSV);
vn = T.Properties.VariableNames;
algcol = T.(vn{1});
if iscategorical(algcol)
    algcol = cellstr(algcol);
end
sc = double(T.(vn{2}));
mu = double(T.(vn{3}));
x  = unique(sc(:));

Y = cell(1, numel(ALGOS));
for a = 1:numel(ALGOS)
    keep = strcmp(algcol, ALGOS{a});
    if ~any(keep)
        continue
    end
    [~, k] = sort(sc(keep));
    Y{a} = mu(keep);
    Y{a} = Y{a}(k);
end
fprintf('source : %s\n', CSV);

%% ------------------------------ plot -------------------------------------
thisdir = fileparts(mfilename('fullpath'));
figdir  = thisdir;    % self-contained: write the figure next to this script
if ~exist(figdir, 'dir')
    mkdir(figdir);
end

figure('Color', 'w', 'Position', [100 100 560 420]);
hold on; grid on; box on;

h = gobjects(1, numel(ALGOS));
lab = cell(1, numel(ALGOS));
n = 0;
for a = 1:numel(ALGOS)
    if isempty(Y{a})
        continue
    end
    n = n + 1;
    h(n) = plot(x, Y{a}, SPEC{a}, 'LineWidth', 1.5, 'MarkerSize', 5);
    lab{n} = LABELS{a};
end
if n == 0
    error('plot_area_sweep:noData', 'no algorithm column found in %s', CSV);
end

set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, ...
    'XMinorTick', 'on', 'YMinorTick', 'on');
xlabel(XLAB);
ylabel(YLAB);
legend(h(1:n), lab(1:n), 'Location', 'southeast');
xticks(x);
xlim([min(x), max(x)]);
xticklabels(compose('%.1f', xticks));

%%%%---eps---%%%%
set(gcf, 'PaperPositionMode', 'auto');
outpath = fullfile(figdir, [OUTNAME '.eps']);
print(gcf, outpath, '-depsc2', '-painters');
print(gcf, fullfile(figdir, [OUTNAME '.png']), '-dpng', '-r300');
fprintf('Saved: %s\n', outpath);
