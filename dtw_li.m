% dtw计算两个粒度时间序列之间的距离

function [Dis] = dtw_li(P1, P2)
% 快速版本，不包含可视化和详细输出
    
    % 转换为元胞数组
    if ~iscell(P1)
        P1 = num2cell(P1, 2);
    end
    if ~iscell(P2)
        P2 = num2cell(P2, 2);
    end
    
    x_len = length(P1);
    y_len = length(P2);
    
    % 计算距离矩阵
    distance = zeros(x_len, y_len);
    for i = 1:x_len
        for j = 1:y_len
            grain_i = P1{i}(:)';
            grain_j = P2{j}(:)';
            distance(i,j) = unequal_d(grain_i, grain_j);
        end
    end
    
    % 动态规划
    DP = zeros(x_len, y_len);
    DP(1,1) = distance(1,1);
    
    for i = 2:x_len
        DP(i,1) = distance(i,1) + DP(i-1,1);
    end
    for j = 2:y_len
        DP(1,j) = distance(1,j) + DP(1,j-1);
    end
    
    for i = 2:x_len
        for j = 2:y_len
            DP(i,j) = distance(i,j) + min([DP(i-1,j), DP(i,j-1), DP(i-1,j-1)]);
        end
    end
    
    % 回溯
    i = x_len;
    j = y_len;
    pp = 0;
    alignment = [];
    
    while ~((i == 1) && (j == 1))
        alignment = [alignment; i, j];
        
        if i == 1
            j = j - 1;
        elseif j == 1
            i = i - 1;
        else
            [~, idx] = min([DP(i-1,j-1), DP(i-1,j), DP(i,j-1)]);
            switch idx
                case 1
                    i = i - 1; j = j - 1;
                case 2
                    i = i - 1;
                case 3
                    j = j - 1;
            end
        end
        pp = pp + 1;
    end
    
    alignment = [alignment; 1, 1];
    alignment = flipud(alignment);
    
    % 归一化距离
    Dis = DP(x_len, y_len) / pp;
end






