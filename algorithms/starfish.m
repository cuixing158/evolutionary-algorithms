function [x,fval,Convergence_curve] = starfish(fobj,nvars,lb,ub,options)
% Brief: STARFISH OPTIMIZATION ALGORITHM (SFOA,Optimized Version)
% Details:
%   STARFISH attempts to solve problems of the form:
%       min F(X)  subject to  LB <= X <= UB
%        X
% This function implements STARFISH OPTIMIZATION ALGORITHM. It is inspired 
% by the behavior of starfish in nature, particularly their movement and 
% regeneration abilities. Starfish explore their environment to find food 
% (exploration phase) and exploit nearby resources efficiently (exploitation phase). 
% Additionally, starfish can regenerate lost arms,which is mimicked in the 
% algorithm to maintain diversity and avoid getting stuck in suboptimal solutions.
%
% Syntax:
%       [x, fval, Convergence_curve] = starfish(fobj, nvars, lb, ub)
%       [x, fval, Convergence_curve] = starfish(fobj, nvars, lb, ub, options)
%
% Inputs:
%       fobj   - Function handle(https://ww2.mathworks.cn/help/matlab/function-handles.html)
%                of the objective function to optimize.
%                The function should accept a row vector of size [1, nvars]
%                and return a scalar objective value.
%       nvars  - Positive integer specifying the number of variables(dimensions).
%       lb     - Row vector specifying the lower bounds of the variables.
%       ub     - Row vector specifying the upper bounds of the variables.
%       options (Optional) - A structure with the following fields:
%           SearchAgentsNumber - Number of search agents (default: min(100, 10*nvars)).
%           MaxIterations      - Maximum number of iterations (default: 500).
%           PlotFcns           - Logical flag to enable/disable plot(default:false).
%           UseParallel        - Logical flag to enable/disable parallel computing (default: false).
%
% Outputs:
%       x                  - Best solution found by the algorithm (row vector).
%       fval               - Objective value of the best solution.
%       Convergence_curve  - Array storing the best objective value at
%                            each iteration.
%
% Example1:
%    Minimize the Rosenbrock function:
%       f = @(x) (1 - x(1))^2 + 100 * (x(2) - x(1)^2)^2;
%       nvars = 2;
%       lb = -10 * ones(1, nvars);
%       ub = 10 * ones(1, nvars);
%       x = starfish(f, nvars, lb, ub);
%
% Example2:
%   Optimize a complex and time-consuming performance objective function(Rastrigin), 
%   which is a classic test function with many local minima and a complex search 
%   space. Below is a comparison example of optimization using PSO and GJO.
%
%       f = @complexObjective;
%       nvars = 12;
%       lb = -10*ones(1,nvars);
%       ub = 10*ones(1,nvars);
%       
%       t1 = tic;
%       options = optimoptions('particleswarm', UseParallel=true, PlotFcn= @pswplotbestf, SwarmSize=100,MaxIterations=60);
%       [x1,fval1]=particleswarm(f,nvars,lb,ub,options);
%       toc(t1)
%       
%       t2 = tic;
%       [x2,fval2]=starfish(f,nvars,lb,ub,UseParallel=true,PlotFcns=true,SearchAgentsNumber=100,MaxIterations=60);
%       toc(t2)
%       
%       function y = complexObjective(x)  
%           % x 是一个 n 维向量  
%           % Rastrigin函数（多峰函数，优化难度较大）  
%           A = 10;  
%           n = length(x);  
%           rastrigin = A * n + sum(x.^2 - A * cos(2 * pi * x));  
%       
%           % 计算密集型模拟部分（人为加入耗时）  
%           pauseTime = 0.01; % 模拟每次评估耗时0.01秒  
%           tic; while toc < pauseTime; end  
%       
%           % 非线性耦合项  
%           coupling = 0;  
%           for i = 1:n-1  
%               coupling = coupling + 100*(x(i+1)-x(i)^2)^2 + (1 - x(i))^2;  
%           end  
%       
%           % 组合目标函数  
%           y = rastrigin + coupling;  
%       end  
%
% References:
%    Starfish optimization algorithm (SFOA)
%    Original Code Created by Dr. Changting Zhong (Email: zhongct@hainanu.edu.cn)
%    Paper: Changting Zhong, Gang Li, Zeng Meng, Haijiang Li, Ali Riza Yildiz, Seyedali Mirjalili. 
%    Starfish Optimization Algorithm (SFOA): A bio-inspired metaheuristic algorithm for global optimization compared with 100 optimizers
%    Neural Computing and Applications, 2025, 37: 3641-3683.
%
% See also:
%       particleswarm, ga, patternsearch

% Author:                          cuixingxing
% Email:                           cuixingxing150@gmail.com
% Created:                         01-May-2025 20:45:55
% Version history revision notes:
%                                  None
% Implementation In Matlab R2025a
% Copyright © 2025 TheMatrix.All Rights Reserved.
%

arguments
    fobj (1,1) function_handle % what's the function handle? see here, https://ww2.mathworks.cn/help/matlab/function-handles.html
    nvars (1,1) {mustBePositive, mustBeInteger}
    lb (1,:) double
    ub (1,:) double
    options.SearchAgentsNumber (1,1) double {mustBeGreaterThanOrEqual(options.SearchAgentsNumber,2)} = min(100,10*nvars)
    options.MaxIterations(1,1) double {mustBeGreaterThanOrEqual(options.MaxIterations,5)}= 500
    options.PlotFcns (1,1) logical = false
    options.UseParallel (1,1) logical = false
end

% Extract options
SearchAgentsNumber = options.SearchAgentsNumber;
MaxIterations = options.MaxIterations;
PlotFcns = options.PlotFcns;
UseParallel = options.UseParallel;

% Adjust bounds if scalar
if size(lb, 2) == 1
    lb = lb * ones(1, nvars);
end
if size(ub, 2) == 1
    ub = ub * ones(1, nvars);
end

% Initialize population using vectorized operations
Xpos = lb + (ub - lb) .* rand(SearchAgentsNumber, nvars);
Fitness = zeros(SearchAgentsNumber, 1);

% Initial fitness evaluation with optional parallel computing
if UseParallel
    parfor i = 1:SearchAgentsNumber
        Fitness(i) = fobj(Xpos(i,:));
    end
else
    for i = 1:SearchAgentsNumber
        Fitness(i) = fobj(Xpos(i,:));
    end
end

if PlotFcns
    fig = figure('Name', 'starfish optimization', 'NumberTitle', 'off');
    ax = axes(fig);
    lineObj = animatedline(ax, 'LineWidth', 2, 'Color', 'b');
    xlabel(ax, 'Iteration');
    ylabel(ax, 'Best Objective Value');
    title(ax, "Convergence Curve");
    grid(ax, 'on');

    % Add stop button
    stopButton = uicontrol(fig, 'Style', 'pushbutton', 'String', 'Stop', ...
        'Position', [20 20 50 20], 'Callback', @(src, event) stopCallback(src, event));
    stopButton.UserData = false; % Initialize stop flag
end

% Find initial best solution
[fval, order] = min(Fitness);
x = Xpos(order, :);
Convergence_curve = zeros(1, MaxIterations);

% Main evolution loop
T = 0;
while T<MaxIterations
    T = T+1;
    theta = pi/2 * T / MaxIterations;
    tEO = (MaxIterations - T) / MaxIterations * cos(theta);
    
    % Exploration or exploitation phase
    if rand < 0.5  % Exploration
        newX = Xpos;
        if nvars > 5
            jp1 = randperm(nvars, 5);
            for j = 1:5
                pm = (2 * rand - 1) * pi;
                if rand < 0.5
                    newX(:, jp1(j)) = Xpos(:, jp1(j)) + pm * (x(jp1(j)) - Xpos(:, jp1(j))) * cos(theta);
                else
                    newX(:, jp1(j)) = Xpos(:, jp1(j)) - pm * (x(jp1(j)) - Xpos(:, jp1(j))) * sin(theta);
                end
            end
        else
            jp2 = ceil(nvars * rand);
            im = randperm(SearchAgentsNumber);
            rand1 = 2 * rand - 1;
            rand2 = 2 * rand - 1;
            newX(:, jp2) = tEO * Xpos(:, jp2) + rand1 * (Xpos(im(1), jp2) - Xpos(:, jp2)) + ...
                           rand2 * (Xpos(im(2), jp2) - Xpos(:, jp2));
        end
    else  % Exploitation
        df = randperm(SearchAgentsNumber, 5);
        dm = x - Xpos(df, :);  % Vectorized difference calculation
        r1 = rand;
        r2 = rand;
        kp = randperm(5, 2);
        newX = Xpos + r1 * dm(kp(1), :) + r2 * dm(kp(2), :);
        newX(end, :) = exp(-T * SearchAgentsNumber / MaxIterations) .* Xpos(end, :);
    end
    newX = clip(newX,lb,ub);% Since R2024a, USE newX = max(min(newX, ub), lb) for previous version

    
    % Fitness evaluation with optional parallel computing
    if UseParallel
        newFit = zeros(SearchAgentsNumber, 1);
        parfor i = 1:SearchAgentsNumber
            newFit(i) = fobj(newX(i,:));
        end
        
        newIdxs = newFit<Fitness;
        Fitness(newIdxs) = newFit(newIdxs);
        Xpos(newIdxs,:) = newX(newIdxs,:);

        [fval,bestIdx] = min(Fitness);
        x = newX(bestIdx,:);
    else
        for i = 1:SearchAgentsNumber
            newFit = fobj(newX(i,:));
            if newFit < Fitness(i)
                Fitness(i) = newFit;
                Xpos(i,:) = newX(i,:);
                if newFit < fval
                    fval = newFit;
                    x = newX(i,:);
                end
            end
        end
    end
    
    Convergence_curve(T) = fval;
    
     % Plot convergence curve if enabled
    if PlotFcns
        addpoints(lineObj, T, fval);
        ax.Title.String = "Best Function Value:"+string(fval);
        drawnow;

        % Check stop button
        if stopButton.UserData
            break;
        end
    end
end % end of while loop

% Trim convergence curve if stopped early
if T < MaxIterations
    Convergence_curve = Convergence_curve(1:T);
end

if T==MaxIterations && PlotFcns
    stopButton.Visible = "off";
end

% Callback function for stop button
function stopCallback(src, ~)
    src.UserData = true;
    set(src, 'Visible', 'off'); 
end
end