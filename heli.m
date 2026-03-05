% ====================== 主程序：多变量PSO优化时间窗口长度 ======================
clc;tic;
clear all;
close all;

% 启动并行池
if isempty(gcp('nocreate'))
    parpool('local', 4);
    fprintf('已开启并行计算池，使用4个核心\n');
end

% 参数设置
pop = 80;               % 种群大小
dim = 20;               % 窗口数量p（问题维度）
maxIter = 100;          % 最大迭代次数
max_order = 5;
cu_k = 3; % 聚类数

% ====================== 1. 加载多变量时间序列数据 ======================
% 假设CSV文件有多列，每列是一个变量
data0 = readtable('NN5_Complete_101.csv');
% 提取多个变量（例如第2-7列，排除第一列的时间戳）
variable_indices = 2:6;  % 根据实际数据调整
M = length(variable_indices);  % 变量数量

% 构建多变量时间序列矩阵：M×N
time_series_matrix = zeros(M, height(data0)-1);
for m = 1:M
    time_series_matrix(m, :) = table2array(data0(2:end, variable_indices(m)));
    % time_series_matrix(m, :) = table2array(data0(1:end, variable_indices(m)));
end

% time_series_matrix = time_series_matrix(:,1:1000);
N = size(time_series_matrix, 2);  % 数据点总数

fprintf('多变量时间序列信息：\n');
fprintf('  变量数量 M = %d\n', M);
fprintf('  数据点数 N = %d\n', N);
fprintf('  总数据量 = %d\n', M*N);

% 显示原始时间序列
figure(1);
for m = 1:min(4, M)  % 最多显示前4个变量
    subplot(2, 2, m);
    plot(time_series_matrix(m, :), 'b-', 'LineWidth', 1);
    xlabel('时间步');
    ylabel('幅值');
    title(sprintf('变量 %d', m));
    grid on;
end

% ====================== 2. PSO参数设置 ======================
ub = ones(1, dim);
lb = zeros(1, dim);
vmax = 0.1 * ones(1, dim);
vmin = -0.1 * ones(1, dim);

% 适应度函数（多变量版本）- 使用mdl_w0_safe
fobj = @(X)compute_fitness_multivariate_safe(X, time_series_matrix, dim, N, M, max_order);

% ====================== 3. 运行PSO算法 ======================
fprintf('\n开始多变量PSO优化公共窗口划分...\n');
fprintf('窗口数量 p = %d, 变量数量 M = %d\n', dim, M);

[Best_Pos, Best_fitness, IterCurve] = pso(pop, dim, ub, lb, fobj, vmax, vmin, maxIter);

% ====================== 4. 解码最优解 ======================
proportions = Best_Pos;
proportions(proportions < 0.01) = 0.01;
proportions = proportions / sum(proportions);
window_lengths = round(proportions * N);
window_lengths(end) = N - sum(window_lengths(1:end-1));

% 确保所有窗口长度至少为 max_order+2，避免拟合问题
min_window_length = max_order + 5;  % 加一些余量
for i = 1:dim
    if window_lengths(i) < min_window_length
        window_lengths(i) = min_window_length;
    end
end
% 重新调整最后一个窗口
window_lengths(end) = N - sum(window_lengths(1:end-1));
if window_lengths(end) < min_window_length
    % 如果最后一个窗口太小，从前面的窗口借一些点
    for i = dim-1:-1:1
        if window_lengths(i) > min_window_length + 5
            transfer = min_window_length - window_lengths(end);
            window_lengths(i) = window_lengths(i) - transfer;
            window_lengths(end) = window_lengths(end) + transfer;
            break;
        end
    end
end

% 最后再次检查所有窗口
for i = 1:dim
    if window_lengths(i) < max_order + 2
        window_lengths(i) = max_order + 2;
    end
end
window_lengths(end) = N - sum(window_lengths(1:end-1));

fprintf('\n优化结果：\n');
fprintf('最优适应度（平均粒体积）: %.4f\n', Best_fitness);
fprintf('公共窗口长度序列: \n');
disp(window_lengths);
fprintf('平均窗口长度: %.1f\n', mean(window_lengths));
fprintf('最小窗口长度: %d\n', min(window_lengths));
fprintf('最大窗口长度: %d\n', max(window_lengths));

% ====================== 5. 计算各变量在每个窗口的适应度 ======================
fprintf('\n各变量适应度详情：\n');
fprintf('变量\t总适应度\t平均适应度/窗口\n');
cum_lengths = [0, cumsum(window_lengths)];

for m = 1:M
    var_fitness = 0;
    valid_windows = 0;
    for i = 1:dim
        start_idx = cum_lengths(i) + 1;
        end_idx = cum_lengths(i+1);
        % 确保索引有效
        end_idx = min(end_idx, N);
        if start_idx > end_idx || window_lengths(i) < max_order+2
            continue;
        end
        
        window_data = time_series_matrix(m, start_idx:end_idx);
        T_i = window_lengths(i);
        x = (1:T_i)';
        
        try
            [~,~,~,~,~, optimal_f] = mdl_w0_safe(x, window_data', max_order);
            window_fitness = optimal_f / T_i;
            var_fitness = var_fitness + window_fitness;
            valid_windows = valid_windows + 1;
        catch
            % 如果出错，跳过这个窗口
            continue;
        end
    end
    if valid_windows > 0
        var_avg_fitness = var_fitness / valid_windows;
    else
        var_avg_fitness = 0;
    end
    fprintf('%2d\t%.4f\t\t%.4f\n', m, var_fitness, var_avg_fitness);
end

% ====================== 6. 可视化结果 ======================
% 收敛曲线
figure(2);
plot(IterCurve, 'r-', 'linewidth', 2);
xlabel('迭代次数');
ylabel('平均适应度');
title('多变量PSO收敛曲线');
grid on;

% 展示所有变量的划分效果
figure(3);
set(gcf, 'Position', [100, 100, 1200, 800]); % 设置大窗口

% 计算子图行数
rows = ceil(M/2);  % 每行2个，向上取整
cols = 2;

for m = 1:M
    subplot(rows, cols, m);
    plot(time_series_matrix(m, :), 'b-', 'LineWidth', 1);
    hold on;
    
    % 绘制分割线
    cum_length = cumsum(window_lengths);
    for i = 1:length(cum_length)-1
        plot([cum_length(i), cum_length(i)], ylim, 'r--', 'LineWidth', 0.8);
    end
    
    xlabel('时间步');
    ylabel('幅值');
    title(sprintf('变量 %d', m));
    grid on;
    
    % 只在第一个子图显示图例
    if m == 1
        legend('原始序列', '窗口边界', 'Location', 'best');
    end
end

% 添加总标题
sgtitle('所有变量的公共窗口划分', 'FontSize', 14, 'FontWeight', 'bold');

% 与均匀划分对比
figure(4);
subplot(2,1,1);
% 显示第一个变量
plot(time_series_matrix(1, :), 'b-', 'LineWidth', 1.5);
hold on;
cum_length = cumsum(window_lengths);
for i = 1:length(cum_length)-1
    plot([cum_length(i), cum_length(i)], [min(time_series_matrix(1, :)), ...
         max(time_series_matrix(1, :))], 'r--', 'LineWidth', 1.5);
end
title('PSO优化的公共窗口划分（变量1）');
grid on;

subplot(2,1,2);
plot(time_series_matrix(1, :), 'b-', 'LineWidth', 1.5);
hold on;
uniform_length = round(N/dim);
for i = 1:dim-1
    plot([i*uniform_length, i*uniform_length], [min(time_series_matrix(1, :)), ...
         max(time_series_matrix(1, :))], 'g--', 'LineWidth', 1.5);
end
title('均匀窗口划分（变量1）');
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  对每个变量每个窗口进行粒化  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = window_lengths; % 每个窗口的长度

opt_order = cell(1, M); % 每个序列每段子序列的最优阶数
opt_mdl_value = cell(1, M); % 每个序列每段子序列最优阶数对应的mdl
opt_y_fit = cell(1, M);
opt_xishu = cell(1, M);
y = cell(1, M); % 用来存储原始数据y值-用于拟合
x0 = cell(1, M); % 用来存储原始数据x值-用于画每一段图像
lihua = cell(1, M); % 存储每个序列每段子序列的粒化参数[拟合系数，方差，T]

% 初始化细胞数组
for j = 1:M
    opt_order{j} = zeros(1, dim);
    opt_mdl_value{j} = zeros(1, dim);
    opt_y_fit{j} = cell(1, dim);
    opt_xishu{j} = cell(1, dim);
    y{j} = cell(1, dim);
    x0{j} = cell(1, dim);
    lihua{j} = cell(1, dim);
end

opt_sigma = cell(1, M); % 每个序列每段子序列最优阶数对应的sigma
opt_f = cell(1, M); % 每个序列每段子序列隶属度之和

for j = 1:M
    for i = 1:dim
        start_idx = sum(T(1:i-1)) + 1;
        end_idx = sum(T(1:i));
        end_idx = min(end_idx, N); % 确保索引不越界
        
        if start_idx > end_idx || T(i) < max_order+2
            % 如果窗口太小，使用默认值
            opt_order{j}(i) = 1;
            opt_mdl_value{j}(i) = 0;
            opt_y_fit{j}{i} = zeros(T(i), 1);
            opt_xishu{j}{i} = zeros(max_order+1, 1);
            opt_sigma{j}{i} = 0.1;
            opt_f{j}{i} = T(i);
            y{j}{i} = time_series_matrix(j, start_idx:end_idx)';
            x0{j}{i} = start_idx:end_idx;
            lihua{j}{i} = [zeros(1, max_order+2), T(i)];
        else
            x_ni = (1:T(i))';  % 用于拟合
            y{j}{i} = time_series_matrix(j, start_idx:end_idx)';
            x0{j}{i} = start_idx:end_idx;
            
            try
                [opt_order{j}(i), opt_mdl_value{j}(i), opt_y_fit{j}{i}, opt_xishu{j}{i}, opt_sigma{j}{i}, opt_f{j}{i}] = ...
                    mdl_w0_safe(x_ni, y{j}{i}, max_order);
                
                % 确保opt_xishu有正确的长度
                coeff_len = length(opt_xishu{j}{i});
                if coeff_len < max_order+1
                    temp = zeros(max_order+1, 1);
                    temp(1:coeff_len) = opt_xishu{j}{i};
                    opt_xishu{j}{i} = temp;
                end
                
                lihua{j}{i} = [opt_xishu{j}{i}(1:(min(opt_order{j}(i)+1, max_order+1)))', opt_sigma{j}{i}, T(i)];
            catch
                % 如果出错，使用默认值
                opt_order{j}(i) = 1;
                opt_mdl_value{j}(i) = 0;
                opt_y_fit{j}{i} = y{j}{i};
                opt_xishu{j}{i} = zeros(max_order+1, 1);
                opt_xishu{j}{i}(1:2) = polyfit(x_ni, y{j}{i}, 1);
                opt_sigma{j}{i} = var(y{j}{i});
                opt_f{j}{i} = T(i);
                lihua{j}{i} = [opt_xishu{j}{i}(1:2)', opt_sigma{j}{i}, T(i)];
            end
        end
    end
end

% 计算粒度序列之间的DTW距离矩阵
ddtw = zeros(M);
for i = 1:M
    for j = 1:M
        ddtw(i,j) = dtw_li(lihua{i},lihua{j});
    end
end

% 对比直接用原始序列计算dtw
ydtw = zeros(M);
for i = 1:M
    for j = 1:M
        ydtw(i,j) = dtw(time_series_matrix(i,:),time_series_matrix(j,:));
    end
end

% 层次聚类
dis_xl1 = nonzeros(triu(ddtw, 1))'; % 把距离矩阵向量化，因为层次聚类的输入只能是向量
if ~isempty(dis_xl1)
    cc = linkage(dis_xl1,'average');
    figure(6);
    dendrogram(cc);
    cu = cluster(cc,'maxclust',cu_k); % 得到每个规则的簇信息
end

elapsedTime = toc;
fprintf('程序运行时间：%.4f 秒\n', elapsedTime);

% ====================== 辅助函数（多变量版本） ======================
function fitness = compute_fitness_multivariate_safe(proportions, time_series_matrix, p, N, M, max_order)
    % 多变量适应度函数：计算所有变量的平均适应度，使用mdl_w0_safe
    % 输入: proportions - 比例向量 [1×p]
    %       time_series_matrix - M×N 矩阵，M个变量，N个时间点
    %       p - 窗口数量
    %       N - 总数据点数
    %       M - 变量数量
    %       max_order - 最大多项式阶数
    % 输出: fitness - 所有变量的平均适应度

    % 1. 比例向量预处理
    proportions(proportions < 0.001) = 0.001;
    proportions = proportions / sum(proportions);
    
    % 2. 计算实际窗口长度
    window_lengths = round(proportions * N);
    window_lengths(end) = N - sum(window_lengths(1:end-1));
    
    % 确保最小窗口长度足够进行多项式拟合
    min_window_length = max_order + 3;
    for i = 1:p
        if window_lengths(i) < min_window_length
            window_lengths(i) = min_window_length;
        end
    end
    % 重新调整最后一个窗口
    window_lengths(end) = N - sum(window_lengths(1:end-1));
    
    if any(window_lengths <= 0)
        fitness = -inf;  % 最大化问题，用 -inf 作为惩罚
        return;
    end
    
    % 3. 计算每个变量的适应度
    total_fitness_all_vars = 0;
    
    % 并行计算各个变量（外循环并行）
    parfor m = 1:M
        var_fitness = 0;
        valid_windows = 0;
        
        % 计算该变量在所有窗口的适应度之和
        start_idx = 1;
        for i = 1:p
            end_idx = start_idx + window_lengths(i) - 1;
            if end_idx > N
                end_idx = N;
            end
            
            if start_idx > end_idx || window_lengths(i) < max_order+2
                start_idx = end_idx + 1;
                continue;
            end
            
            % 提取该变量在这个窗口的数据
            window_data = time_series_matrix(m, start_idx:end_idx)';
            T_i = window_lengths(i);
            x = (1:T_i)';
            
            % 计算该窗口的适应度
            try
                [~,~,~,~,~, optimal_f] = mdl_w0_safe(x, window_data, max_order);
                if optimal_f > 0
                    window_fitness = optimal_f / T_i;
                    var_fitness = var_fitness + window_fitness;
                    valid_windows = valid_windows + 1;
                end
            catch
                % 如果出错，跳过这个窗口
            end
            
            start_idx = end_idx + 1;
        end
        
        % 该变量的总适应度（对所有窗口求和）
        if valid_windows > 0
            total_fitness_all_vars = total_fitness_all_vars + (var_fitness / valid_windows);
        end
    end
    
    % 4. 返回平均适应度
    if M > 0
        fitness = total_fitness_all_vars / M;  % 平均到每个变量
    else
        fitness = 0;
    end
end

% ====================== 其他辅助函数 ======================
function [X] = initialization(pop, lb, ub, dim)
    X = zeros(pop, dim);
    for i = 1:pop
        for j = 1:dim
            X(i, j) = (ub(j) - lb(j)) * rand() + lb(j);
        end
    end
end

function [X] = BoundaryCheck(X, ub, lb, dim)
    [rows, ~] = size(X);
    if rows > 1
        for i = 1:rows
            for j = 1:dim
                if X(i, j) > ub(j)
                    X(i, j) = ub(j);
                end
                if X(i, j) < lb(j)
                    X(i, j) = lb(j);
                end
            end
        end
    else
        for j = 1:dim
            if X(j) > ub(j)
                X(j) = ub(j);
            end
            if X(j) < lb(j)
                X(j) = lb(j);
            end
        end
    end
end

function [Best_Pos, Best_fitness, IterCurve] = pso(pop, dim, ub, lb, fobj, vmax, vmin, maxIter)
    c1 = 2.0;
    c2 = 2.0;
    w_max = 0.9;
    w_min = 0.4;
    
    % 初始化
    X = initialization(pop, lb, ub, dim);
    V = initialization(pop, vmin, vmax, dim);
    
    fitness = zeros(1, pop);
    for i = 1:pop
        fitness(i) = fobj(X(i, :));
    end
    
    pBest = X;
    pBestFitness = fitness;
    
    [gBestFitness, index] = max(fitness);
    gBest = X(index, :);
    
    Xnew = X;
    fitnessNew = fitness;
    
    % 迭代
    for t = 1:maxIter
        w = w_max - (w_max - w_min) * t / maxIter;
        
        for i = 1:pop
            r1 = rand(1, dim);
            r2 = rand(1, dim);
            
            V(i, :) = w * V(i, :) + c1 .* r1 .* (pBest(i, :) - X(i, :)) ...
                    + c2 .* r2 .* (gBest - X(i, :));
            
            V(i, :) = BoundaryCheck(V(i, :), vmax, vmin, dim);
            
            Xnew(i, :) = X(i, :) + V(i, :);
            Xnew(i, :) = BoundaryCheck(Xnew(i, :), ub, lb, dim);
            
            fitnessNew(i) = fobj(Xnew(i, :));
            
            if fitnessNew(i) > pBestFitness(i)
                pBest(i, :) = Xnew(i, :);
                pBestFitness(i) = fitnessNew(i);
            end
            
            if fitnessNew(i) > gBestFitness
                gBestFitness = fitnessNew(i);
                gBest = Xnew(i, :);
            end
        end
        
        X = Xnew;
        fitness = fitnessNew;
        
        IterCurve(t) = gBestFitness;
        
        if mod(t, 20) == 0
            fprintf('  迭代 %d/%d, 最优适应度: %.4f\n', t, maxIter, gBestFitness);
        end
    end
    
    Best_Pos = gBest;
    Best_fitness = gBestFitness;
end

% 先定义安全的mdl_w0函数
function [optimal_order, optimal_mdl_value, optimal_y_fit, optimal_xishu, optimal_sigma, optimal_f] = mdl_w0_safe(x, y, max_order)
    % 安全的mdl_w0版本，避免窗口太小导致的错误
    
    n = length(y);
    
    % 初始化输出变量，确保始终有值
    optimal_order = 1;
    optimal_mdl_value = Inf;
    optimal_y_fit = y;  % 默认使用原始数据
    optimal_xishu = zeros(max_order+1, 1);
    optimal_sigma = var(y);
    optimal_f = n;  % 默认隶属度之和为n
    
    % 如果数据点太少，无法进行多项式拟合，直接返回
    if n < 2
        % 只有一个数据点，返回默认值
        optimal_order = 0;
        optimal_mdl_value = 0;
        optimal_y_fit = y;
        optimal_xishu(1) = y;
        optimal_sigma = 0;
        optimal_f = 1;
        return;
    end
    
    % 限制最大阶数不超过数据点数-1
    max_order = min(max_order, n-1);
    
    if max_order < 0
        % 无法拟合任何多项式
        return;
    end
    
    % 正常执行mdl_w0逻辑
    try
        min_mdl = Inf;
        mdl_values = Inf(1, max_order+1);
        error_terms = zeros(1, max_order+1);
        penalty_terms = zeros(1, max_order+1);
        p = zeros(max_order+1, max_order+1);
        y_fit = zeros(n, max_order+1);
        residuals = zeros(n, max_order+1);
        f = zeros(n, max_order+1);
        
        % 计算各阶sigma值
        sigma_squared = Inf(1, max_order+1);
        rss = zeros(1, max_order+1);
        
        for k = 0:max_order
            try
                if k <= n-1  % 确保多项式阶数不超过数据点数量
                    p_coeff = polyfit(x, y, k);
                    p(1:k+1, k+1) = p_coeff;
                    y_fit(:, k+1) = polyval(p_coeff, x);
                    residuals(:, k+1) = y - y_fit(:, k+1);
                    rss(k+1) = sum(residuals(:, k+1).^2);
                    sigma_squared(k+1) = rss(k+1) / n;
                else
                    % 阶数太高，跳过
                    sigma_squared(k+1) = Inf;
                end
            catch
                sigma_squared(k+1) = Inf;
            end
        end
        
        % 移除无效值
        valid_idx = isfinite(sigma_squared) & sigma_squared > 0;
        if any(valid_idx)
            sigma_valid = sigma_squared(valid_idx);
            sigma_med = median(sigma_valid);
            sigma_global = sigma_med;
            sigma_max = max(sigma_valid);
            sigma_min = min(sigma_valid);
            
            w = zeros(1, max_order+1);
            sigma_w = zeros(1, max_order+1);
            
            % 计算各阶MDL值
            for i = 0:max_order
                if valid_idx(i+1) && i <= n-1
                    if sigma_max ~= sigma_min
                        w(i+1) = 1 - ((sigma_squared(i+1) - sigma_min) / (sigma_max - sigma_min));
                    else
                        w(i+1) = 0.5;
                    end
                    sigma_w(i+1) = w(i+1) * sigma_squared(i+1) + (1-w(i+1)) * sigma_global;
                    
                    % 计算隶属度
                    if sigma_w(i+1) > 0
                        f(:, i+1) = exp(-((residuals(:, i+1)).^2) / (2 * sigma_w(i+1)));
                        f_sum = sum(f(:, i+1));
                        if f_sum > 0
                            error_term = (n / 2) * log(n / f_sum);
                            penalty_term = ((i + 1) / 2) * log(n);
                            mdl = error_term + penalty_term;
                            
                            error_terms(i+1) = error_term;
                            penalty_terms(i+1) = penalty_term;
                            mdl_values(i+1) = mdl;
                            
                            % 记录最优阶数
                            if mdl < min_mdl
                                min_mdl = mdl;
                                optimal_order = i;
                                optimal_mdl_value = mdl_values(i+1);
                                optimal_y_fit = y_fit(:, i+1);
                                optimal_xishu = zeros(max_order+1, 1);
                                optimal_xishu(1:i+1) = p(1:i+1, i+1);
                                optimal_sigma = sigma_squared(i+1);
                                optimal_f = f_sum;
                            end
                        end
                    end
                end
            end
        end
        
        % 如果没有找到有效的阶数，使用线性拟合
        if min_mdl == Inf
            optimal_order = min(1, n-1);
            p_coeff = polyfit(x, y, optimal_order);
            optimal_xishu(1:optimal_order+1) = p_coeff;
            optimal_y_fit = polyval(p_coeff, x);
            residuals = y - optimal_y_fit;
            optimal_sigma = sum(residuals.^2) / n;
            if optimal_sigma > 0
                optimal_f = sum(exp(-residuals.^2/(2*optimal_sigma)));
                optimal_mdl_value = (n/2)*log(n/optimal_f);
            else
                optimal_f = n;
                optimal_mdl_value = 0;
            end
        end
        
    catch ME
        % 如果出错，使用默认线性拟合
        warning('mdl_w0_safe出错，使用默认线性拟合。错误信息:', '%s', ME.message);
        optimal_order = min(1, n-1);
        if n >= 2
            p_coeff = polyfit(x, y, optimal_order);
            optimal_xishu(1:optimal_order+1) = p_coeff;
            optimal_y_fit = polyval(p_coeff, x);
            residuals = y - optimal_y_fit;
            optimal_sigma = sum(residuals.^2) / n;
            optimal_f = sum(exp(-residuals.^2/(2*optimal_sigma)));
            optimal_mdl_value = (n/2)*log(n/optimal_f);
        end
    end
end