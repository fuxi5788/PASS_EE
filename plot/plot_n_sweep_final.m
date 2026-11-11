%% plot_n_sweep.m
%  Energy efficiency versus the number of pinch points (PAs) per waveguide.
%
%  Data: n_sweep_data.csv (long layout: algo,N,ee_mean,ee_std,n) written by
%        run_n_sweep.py.  Tframe = 1.0 s, M = 3 waveguides, K = 3 users,
%        D = 40 m, Pmax = 25 dBm, seed 32; N = 4 ... 10.
%        Only the mean EE (column 3) is plotted; std and n are not used here.
%
%  Output: <mulsim>/plot/fig_n_sweep.eps (+ .png).
%
%  Style follows plot_tf_sweep.m: white figure, Times New Roman, minor ticks,
%  Painter EPS.  The FPA series is drawn dashed grey: its pinch points are
%  fixed (no movement, no radiation-ratio optimisation) and the M*N antennas are
%  used as MIMO ports, so its EE rises with N (6.96 -> 8.71) but stays the
%  lowest of the five curves.

clc; close all;

%% ------------------------------ settings ---------------------------------
CSV     = 'n_sweep_all_final.csv';
OUTNAME = 'fig_n_sweep';

XLAB = 'Number of PAs on each waveguide';
YLAB = 'Energy efficiency (bps/Hz/J)';

ALGOS  = {'SAC',  'BGAT', 'DSD',  'ADMM', 'FPA'};
LABELS = {'GNN-SAC', 'BGAT', 'DSD', 'ADMM', 'FPA'};
SPEC   = {'r-o', 'c-d', 'm-^', 'b-s', 'k--*'};

%% ------------------------------- read ------------------------------------
if ~exist(CSV, 'file')
    error('plot_n_sweep:missingCsv', 'not found: %s (run: python run_n_sweep.py --write-csv)', CSV);
end
T = readtable(CSV);
vn = T.Properties.VariableNames;
algcol = T.(vn{1});
if iscategorical(algcol)
    algcol = cellstr(algcol);
end
N  = double(T.(vn{2}));
mu = double(T.(vn{3}));
x  = unique(N(:));

Y = cell(1, numel(ALGOS));
Xa = cell(1, numel(ALGOS));  % store N values per algorithm
for a = 1:numel(ALGOS)
    keep = strcmp(algcol, ALGOS{a});
    if ~any(keep)
        continue
    end
    [Ns, k] = sort(N(keep));
    Y{a} = mu(keep);
    Y{a} = Y{a}(k);
    Xa{a} = Ns;
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
    h(n) = plot(Xa{a}, Y{a}, SPEC{a}, 'LineWidth', 1.5, 'MarkerSize', 5);
    lab{n} = LABELS{a};
end
if n == 0
    error('plot_n_sweep:noData', 'no algorithm column found in %s', CSV);
end

set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, ...
    'XMinorTick', 'on', 'YMinorTick', 'on');
xlabel(XLAB);
ylabel(YLAB);
legend(h(1:n), lab(1:n), 'Location', 'best');
xticks(x);
xlim([min(x), 10]);

%%%%---eps---%%%%
set(gcf, 'PaperPositionMode', 'auto');
outpath = fullfile(figdir, [OUTNAME '.eps']);
print(gcf, outpath, '-depsc2', '-painters');
print(gcf, fullfile(figdir, [OUTNAME '.png']), '-dpng', '-r300');
fprintf('Saved: %s\n', outpath);
