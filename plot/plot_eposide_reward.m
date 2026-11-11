% function plot_hierarchy_v2()
% plot_hierarchy_v2.m
% Read hierarchy_v2_data.mat and plot recent10 ee curves with SEM.
% Run in MATLAB: >> plot_hierarchy_v2
clear;close all;
    npz_path = 'results_convergence_20260910_113046.npz';
    mat_path = 'hierarchy_v2_data.mat';
    p_path = 'plot_eposide_reward1110.mat';
%     if exist(npz_path, 'file')
%         fprintf('Loading NPZ: %s\n', npz_path);
%         try
%             data = load_npz_into_struct(npz_path);
%         catch err
%             warning('NPZ load failed (%s). Falling back to %s.', err.message, mat_path);
%             data = load(mat_path);
%         end
%     elseif exist(mat_path, 'file')
%         fprintf('Loading MAT: %s\n', mat_path);
%         data = load(mat_path);
%     else
%         error('Neither %s nor %s found.', npz_path, mat_path);
%     end
    data = load(p_path);
    algos   = { 'sac_ees', 'ddpg_ees', 'ppo_ees', 'a2c_ees', 'gnn_sac_ees'};
    labels  = { 'SAC', 'DDPG', 'PPO', 'A2C', 'GNN-SAC'};

    colors = [ 
              0.49 0.18 0.56;     % purple- SAC
              1.00 0.50 0.05;     % orange- DDPG
              0.17 0.63 0.17;     % green - PPO
              0.12 0.47 0.71;     % red   - A2C
              0.84 0.15 0.16 ];   % blue  - GNN-SAC
                               

    W = 10;  % rolling window for SEM

    figure('Color', 'w', 'Position',[100 100 560 420] ); % [100 100 600 400]
    hold on; box on; grid on;

    hLines = gobjects(numel(algos), 1);  % for explicit legend
    
    for i = 1:numel(algos)
        d = data.(algos{i});
        d = d(:);
        x = (1:length(d))';
        y = movmean(d,  W);          % smoothed mean curve
        % Mean +/- std band
        sd = rolling_std(d,  W);
        upper = y + sd;
        lower = y - sd;

        % Shaded area - not in legend
        % Use an opaque light color instead of transparency so that the
        % band also survives EPS export (EPS does not support alpha).
        lightColor = 1 - (1 - colors(i,:)) * 0.55;   % 25% color + 75% white
        fill([x; flipud(x)], [upper; flipud(lower)], lightColor, ...
             'EdgeColor', 'none', 'HandleVisibility', 'off');

        % Mean curve - no markers
        hLines(i) = plot(x, y, 'Color', colors(i,:), 'LineWidth', 1.5);
    end

    xlabel('Episode', 'FontName', 'Times New Roman', 'FontSize', 11);
    ylabel('Average Reward', 'FontName', 'Times New Roman', 'FontSize', 11);

    legend(hLines, labels, 'Location', 'southeast', ...
           'FontName', 'Times New Roman', 'FontSize', 10);

    set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, ...
             'XMinorTick', 'on', 'YMinorTick', 'on');

    xlim([0 max(x)]);
    ylim([0 12]);
    hold off;
    % EPS export
    set(gcf, 'PaperPositionMode', 'auto');
    thisdir = fileparts(mfilename('fullpath'));
    figdir  = thisdir;    % self-contained: write the figure next to this script
    if ~exist(figdir, 'dir'), mkdir(figdir); end
    outpath = fullfile(figdir, 'fig_episode_vs_reward.eps');
    print(gcf, outpath, '-depsc2', '-painters');
    fprintf('Saved: %s\n', outpath);

    % PNG export (transparency-safe, use for quick check)
%     pngpath = fullfile(figdir, 'hierarchy_v2_shaded.png');
%     print(gcf, pngpath, '-dpng', '-opengl', '-r300');
%     fprintf('Saved: %s\n', pngpath);
% end

function data = load_npz_into_struct(npz_path)
    npz = py.numpy.load(npz_path);
    keys   = {'gnn_sac', 'sac', 'ddpg', 'ppo', 'a2c'};
    fields = {'GNN_SAC', 'SAC', 'DDPG', 'PPO', 'A2C'};
    for i = 1:numel(keys)
        ees   = double(npz{[keys{i} '_ees']});
        avg_r = double(npz{[keys{i} '_avg_rewards']});
        n  = numel(ees);
        ep = (1:n)';
        data.(fields{i}) = struct( ...
            'episode',     ep, ...
            'raw_ee',      ees, ...
            'recent10_ee', movmean(ees, 10), ...
            'avg_r',       avg_r, ...
            'recent10_r',  movmean(avg_r, 10));
    end
end


function s = rolling_std(v, W)
    n = numel(v);
    s = zeros(size(v));
    for i = 1:n
        lo = max(1, i - W + 1);
        win = v(lo:i);
        s(i) = std(win);
    end
end


