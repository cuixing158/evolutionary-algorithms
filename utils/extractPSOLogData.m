function [iterations, bestFvals] = extractPSOLogData(filename)
    % 从PSO日志文件中提取迭代次数和最佳适应值
    % 输入：
    %   filename - 日志文件名（如'pso_iter.txt'）
    % 输出：
    %   iterations - 迭代次数数组
    %   bestFvals - 最佳适应值数组
    
    % 读取文件内容
    fid = fopen(filename, 'r');
    if fid == -1
        error('文件打开失败，请检查文件名或路径');
    end
    fileContent = textscan(fid, '%s', 'Delimiter', '\n');
    fclose(fid);
    lines = fileContent{1};
    
    % 初始化变量
    iterations = [];
    bestFvals = [];
    
    % 正则表达式匹配数据行（匹配类似"   12            650       2.973e+05..."的行）
    pattern = '^\s*(\d+)\s+(\d+)\s+([\d\.e+-]+)\s+';
    
    % 遍历每一行
    for i = 1:length(lines)
        line = lines{i};
        tokens = regexp(line, pattern, 'tokens');
        
        if ~isempty(tokens)
            % 提取三列数据：Iteration, f-count, Best f(x)
            iter = str2double(tokens{1}{1});
            bestFval = str2double(tokens{1}{3});
            
            % 排除表头行（通过数值范围判断）
            if iter > 0 && isfinite(bestFval)
                iterations = [iterations; iter];
                bestFvals = [bestFvals; bestFval];
            end
        end
    end
end