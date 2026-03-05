function distance = unequal_d(a, b)
% distance 两个不等长的多项式模糊信息粒之间的距离

% 第一步：先判断是否相等
if a(end) == b(end)
    distance = equal_d(a, b);
    return;
end

% 第二步：确保a为短粒，b为长粒
if a(end) > b(end)
    [a, b] = deal(b, a);
end

% 创建b1和b2
b1 = b; b2 = b;
b1(end) = a(end);
b2(end) = b(end) - a(end);

% 解析参数
coeff_a = a(1:end-2);
sigma_a = a(end-1);
T_a = a(end);

coeff_b = b(1:end-2);
sigma_b = b(end-1);
T_b = b(end);

% 调整b2的常数项
b_at_Ta = polyval(coeff_b, T_a);
order_b = length(coeff_b) - 1;
if order_b > 0
    b2_coeff = coeff_b;
    b2_coeff(end) = b_at_Ta;
    b2(1:end-2) = b2_coeff;
else
    b2(1:end-2) = b_at_Ta;
end

% ============ 第一步：特殊处理短粒为0阶的情况 ============
order_a = length(coeff_a) - 1;

if order_a == 0
    % 短粒是0阶多项式：直接连接末端两点构造1阶新粒
    % 短粒函数值
    a_val = polyval(coeff_a, T_a);
    % 长粒在T_b处的函数值
    b_val = polyval(coeff_b, T_b);
    
    % 计算斜率：k = (b_val - a_val) / T_d
    T_d = T_b - T_a;
    if abs(T_d) < eps
        k = 0;
    else
        k = (b_val - a_val) / T_d;
    end
    
    % 构造1阶多项式：d(x) = k*x + b，要求d(0) = a_val
    b_intercept = a_val;  % 因为d(0) = a_val
    
    % 验证：d(T_d) = k*T_d + a_val = b_val
    d_at_Td = k * T_d + a_val;
    if abs(d_at_Td - b_val) > 1e-6
        warning('构造的1阶多项式不满足终点条件: d(T_d)=%.6f, b(T_b)=%.6f', d_at_Td, b_val);
    end
    
    % 构造d粒：[k, b_intercept, 0, T_d]
    d = [k, b_intercept, 0, T_d];
    
    % 计算距离
    distance_part1 = equal_d(a, b1);
    distance_part2 = equal_d(d, b2);
    distance = distance_part1 + distance_part2;
    
    return;  % 直接返回，不再执行后面的代码
end

% ============ 第二步：处理短粒非0阶的情况 ============
% 计算导数
compute_derivatives = @(coeff, x, max_order) get_derivatives(coeff, x, max_order);
max_needed_order = min(20, 2*order_a);
deriv_a = compute_derivatives(coeff_a, T_a, max_needed_order);
deriv_b = compute_derivatives(coeff_b, T_b, max_needed_order);

% 构造新粒d（d的阶数与a相同）
order_d = order_a;
num_coeffs = order_d + 1;
T_d = T_b - T_a;

% 构建条件矩阵
A = zeros(num_coeffs, num_coeffs);
b_vec = zeros(num_coeffs, 1);
row = 0;

% 条件1：起点值 d(0) = a(T_a)
row = row + 1;
A(row, end) = 1;  % c_0的系数
b_vec(row) = deriv_a(1);

% 条件2：终点值 d(T_d) = b(T_b)
if num_coeffs >= 2
    row = row + 1;
    for j = 1:num_coeffs
        power = order_d - (j-1);
        A(row, j) = T_d^power;
    end
    b_vec(row) = deriv_b(1);
end

% 添加导数条件（交替：短粒导数、长粒导数）
if num_coeffs > 2
    deriv_order = 1;
    while row < num_coeffs
        % 短粒导数条件：d^(k)(0) = a^(k)(T_a)
        if row >= num_coeffs, break; end
        row = row + 1;
        
        k = deriv_order;
        if k <= order_d
            coeff_idx = order_d - k + 1;
            A(row, coeff_idx) = factorial(k);
            b_vec(row) = deriv_a(k+1);
        end
        
        % 长粒导数条件：d^(k)(T_d) = b^(k)(T_b)
        if row >= num_coeffs, break; end
        row = row + 1;
        
        if k <= order_d
            for j = 1:num_coeffs
                power = order_d - (j-1);
                if power >= k
                    coeff = factorial(power) / factorial(power - k) * T_d^(power - k);
                    A(row, j) = coeff;
                end
            end
            b_vec(row) = deriv_b(k+1);
        end
        
        deriv_order = deriv_order + 1;
    end
end

% 检查维度
if size(A, 1) ~= num_coeffs || size(A, 2) ~= num_coeffs
    error('矩阵A维度错误: 期望 %dx%d, 实际 %dx%d', ...
        num_coeffs, num_coeffs, size(A,1), size(A,2));
end

if length(b_vec) ~= num_coeffs
    error('向量b长度错误: 期望 %d, 实际 %d', num_coeffs, length(b_vec));
end

% 求解系数
try
    if rank(A) < num_coeffs
        coeff_d = pinv(A) * b_vec;
    else
        coeff_d = A \ b_vec;
    end
catch ME
    fprintf('多项式阶数: %d, 系数数量: %d\n', order_d, num_coeffs);
    fprintf('A矩阵维度: %dx%d\n', size(A,1), size(A,2));
    error('求解线性方程组失败: %s', ME.message);
end

% 构造d粒
d = [coeff_d(:)', 0, T_d];

% 计算距离
distance_part1 = equal_d(a, b1);
distance_part2 = equal_d(d, b2);
distance = distance_part1 + distance_part2;

end

function deriv_values = get_derivatives(coeff, x, max_order)
deriv_values = zeros(max_order+1, 1);
current_coeff = coeff;
deriv_values(1) = polyval(current_coeff, x);
for k = 1:max_order
    if length(current_coeff) > 1
        current_coeff = polyder(current_coeff);
        if ~isempty(current_coeff)
            deriv_values(k+1) = polyval(current_coeff, x);
        else
            deriv_values(k+1) = 0;
        end
    else
        deriv_values(k+1) = 0;
    end
end
end