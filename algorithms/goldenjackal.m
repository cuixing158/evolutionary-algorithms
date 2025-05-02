function [x, fval, Convergence_curve] = goldenjackal(fobj, nvars, lb, ub, options)
% Brief: GOLDEN JACKAL OPTIMIZATION ALGORITHM (GJO,Optimized Version)
% Details:
%   GOLDENJACKAL attempts to solve problems of the form:
%       min F(X)  subject to  LB <= X <= UB
%        X
%
%   This function implements the Golden Jackal Optimization (GJO) algorithm,
%   a nature-inspired optimization algorithm based on the hunting behavior
%   of golden jackals. This optimized version is highly suitable for
%   computationally expensive objective functions, utilizing parallel
%   computing, vectorization, and pre-computation techniques.
%
% Syntax:
%       [x, fval, Convergence_curve] = goldenjackal(fobj, nvars, lb, ub)
%       [x, fval, Convergence_curve] = goldenjackal(fobj, nvars, lb, ub, options)
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
%       x = goldenjackal(f, nvars, lb, ub);
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
%       [x2,fval2]=goldenjackal(f,nvars,lb,ub,UseParallel=true,PlotFcns=true,SearchAgentsNumber=100,MaxIterations=60);
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
%       Chopra, Nitish, and Muhammad Mohsin Ansari. "Golden Jackal Optimization:
%       A Novel Nature-Inspired Optimizer for Engineering Applications."
%       Expert Systems with Applications (2022): 116924.
%  DOI: https://doi.org/10.1016/j.eswa.2022.116924
%
% See also:
%       particleswarm, ga, patternsearch

% Author:                          cuixingxing
% Email:                           cuixingxing150@gmail.com
% Created:                         30-Apr-2025 10:45:45
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

SearchAgentsNumber = options.SearchAgentsNumber;
MaxIterations = options.MaxIterations;
PlotFcns = options.PlotFcns;
UseParallel = options.UseParallel;

%% Initialize Golden Jackal pair
x = zeros(1, nvars);
xExpand = zeros(SearchAgentsNumber,nvars);
fval = inf;
Female_Jackal_pos = zeros(1, nvars);
Female_Jackal_score = inf;

%% Initialize the positions of search agents
Positions = initialization(SearchAgentsNumber, nvars, lb,ub);
Convergence_curve = zeros(1, MaxIterations);

%% Pre-computation for optimization
% Pre-allocate memory
fitness = zeros(SearchAgentsNumber, 1);
Male_Positions = zeros(SearchAgentsNumber, nvars);
Female_Positions = zeros(SearchAgentsNumber, nvars);

% Pre-compute Levy flight and random numbers
RL = 0.05 * levy(SearchAgentsNumber, nvars, 1.5);% Levy flight step size
l = 0; % Loop counter

if PlotFcns
    fig = figure('Name', 'goldenjackal optimization', 'NumberTitle', 'off');
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

%% Main loop
while l < MaxIterations
    l = l + 1;

    % Evaluate objective function for all search agents
    if UseParallel
        parfor i = 1:SearchAgentsNumber
            fitness(i) = fobj(Positions(i, :));
        end
    else
        for i = 1:SearchAgentsNumber
            fitness(i) = fobj(Positions(i, :));
        end
    end

    % Update Male and Female Jackal using mink
    [min_values, min_indices] = mink(fitness, 2);

    % Update Male Jackal
    if min_values(1) < fval
        fval = min_values(1);
        x = Positions(min_indices(1), :);
        xExpand = repmat(x,SearchAgentsNumber,1); % SearchAgentsNumber x nvars size
    end

    % Update Female Jackal
    if length(min_values) > 1 && min_values(2) < Female_Jackal_score
        Female_Jackal_score = min_values(2);
        Female_Jackal_pos = Positions(min_indices(2), :);
        Female_Jackal_pos = repmat(Female_Jackal_pos,SearchAgentsNumber,1); % SearchAgentsNumber x nvars size
    end

    % Vectorized position update
    r1 = rand(SearchAgentsNumber, nvars);
    E1 = 1.5 * (1 - (l / MaxIterations));
    E0 = 2 * r1 - 1;
    E = E1 * E0; % Evading energy

    % Exploration/Exploitation mask
    mask = abs(E) < 1;

    % Vectorized distance and position updates
    D_male_jackal = abs(RL .* xExpand - Positions);
    D_female_jackal = abs(RL .* Female_Jackal_pos - Positions);
    Male_Positions(mask) = xExpand(mask) - E(mask) .* D_male_jackal(mask);
    Female_Positions(mask) = Female_Jackal_pos(mask) - E(mask) .* D_female_jackal(mask);

    mask = ~mask;
    D_male_jackal = abs(xExpand - RL .* Positions);
    D_female_jackal = abs(Female_Jackal_pos - RL .* Positions);
    Male_Positions(mask) = xExpand(mask) - E(mask) .* D_male_jackal(mask);
    Female_Positions(mask) = Female_Jackal_pos(mask) - E(mask) .* D_female_jackal(mask);

    % Update positions
    Positions = (Male_Positions + Female_Positions) / 2;

    % Vectorized boundary checking
    Positions = clip(Positions,lb,ub); % Since R2024a, USE Positions = max(min(Positions, ub), lb) for previous version

    Convergence_curve(l) = fval;

    % Plot convergence curve if enabled
    if PlotFcns
        addpoints(lineObj, l, fval);
        ax.Title.String = "Best Function Value:"+string(fval);
        drawnow;

        % Check stop button
        if stopButton.UserData
            break;
        end
    end
end

% Trim convergence curve if stopped early
if l < MaxIterations
    Convergence_curve = Convergence_curve(1:l);
end

if l==MaxIterations && PlotFcns
    stopButton.Visible = "off";
end

% Callback function for stop button
function stopCallback(src, ~)
    src.UserData = true;
    set(src, 'Visible', 'off'); 
end
end

%% Support functions
function X = initialization(SearchAgents_no, nvars, lb,ub)
Boundary_no = size(ub, 2);
if Boundary_no == 1
    X = rand(SearchAgents_no, nvars) .* (ub - lb) + lb;
else
    X = zeros(SearchAgents_no, nvars);
    for i = 1:nvars
        X(:, i) = rand(SearchAgents_no, 1) .* (ub(i) - lb(i)) + lb(i);
    end
end
end

function z = levy(n, m, beta)
num = gamma(1 + beta) * sin(pi * beta / 2);
den = gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2);
sigma_u = (num / den)^(1 / beta);
u = random('Normal', 0, sigma_u, n, m);
v = random('Normal', 0, 1, n, m);
z = u ./ (abs(v).^(1 / beta));
end
