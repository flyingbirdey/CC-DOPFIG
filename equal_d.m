% 计算两个等长的多项式模糊信息粒之间的距离
function distance = equal_d(a,b)
% distnace 两个等长的多项式模糊信息粒之间的距离
% a 第一个粒的参数向量，如[b3,b2,b1,b0,deta,T]
% b 第二个粒的参数向量，如[b2,b1,b0,deta,T]
% a = [0.001,-0.02,0.15,-0.5,1,0,7,10];
% a = [7,0,10];
% b = [5,0,10];
% 积分区间
low = 0;
up = a(end);
% 两个粒的方差
s1 = a(end-1);
s2 = b(end-1);

p1 = length(a)-3;% 两个粒的阶数
p2 = length(b)-3;
xi1 = a(1:end-2); % 两个粒的系数向量
xi2 = b(1:end-2); 
% shi1 = @(x) polyval(xi1, x);% 两个粒对应的表达式
% shi2 = @(x) polyval(xi2, x);

% 通用多项式函数（修复维度匹配问题）
% 对每个t值计算多项式值，确保维度兼容
poly_fun = @(coeff, t) arrayfun(@(xx) sum(coeff .* (xx .^ (0 : 1 : length(coeff)-1))), t);
% 被积函数：多项式1 - 多项式2
h = @(t) abs(poly_fun(xi1, t) - poly_fun(xi2, t)); % 绝对值符号别忘了，保证1-2与2-1距离计算结果一致
% 计算距离
distance = integral(h, low, up) + sqrt(2*pi)*abs(s1-s2)*up/2;


end