function [gbest,fval, cg_curve] = miragesearch(fobj, nvars, lb, ub, options)
% Brief: Implements the Mirage Search Optimization (MSO) algorithm for global optimization.
% Details:
%    The MSO algorithm is inspired by the mirage phenomenon and uses superior and inferior
%    mirage search strategies to explore and exploit the search space. It is designed to
%    solve complex, multi-modal, and non-convex optimization problems.
%
% Syntax:
%     [gbest, fval, cg_curve] = miragesearch(fobj, dim, lb, ub)
%     [gbest, fval, cg_curve] = miragesearch(fobj, dim, lb, ub, options)
%
% Inputs:
%    fobj - Objective function handle to be minimized.
%           Example: @(x) sum(x.^2) for a simple sphere function.
%    dim - Dimensionality of the optimization problem (number of variables).
%    lb - Lower bounds for the variables (1xdim or scalar).
%    ub - Upper bounds for the variables (1xdim or scalar).
%    options (Optional) - A structure with the following fields:
%        SearchAgentsNumber - Number of search agents (default: min(100, 10*nvars)).
%        MaxIterations      - Maximum number of iterations (default: 500).
%        PlotFcns           - Logical flag to enable/disable plot(default:false).
%        UseParallel        - Logical flag to enable/disable parallel computing (default: false).
%
% Outputs:
%    gbest - Best solution found (1xdim vector).
%    fval - Objective function value at the best solution.
%    cg_curve - Convergence curve showing the best objective value at each iteration.
%
% Example1:
%    Minimize the Rosenbrock function:
%       f = @(x) (1 - x(1))^2 + 100 * (x(2) - x(1)^2)^2;
%       nvars = 2;
%       lb = -10 * ones(1, nvars);
%       ub = 10 * ones(1, nvars);
%       x = miragesearch(f, nvars, lb, ub);
%
% Example2:
%   Optimize a complex and time-consuming performance objective function(Rastrigin),
%   which is a classic test function with many local minima and a complex search
%   space. Below is a comparison example of optimization using PSO and MSO.
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
%       [x2,fval2]=miragesearch(f,nvars,lb,ub,UseParallel=true,PlotFcns=true,SearchAgentsNumber=100,MaxIterations=60);
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
% Reference:
%    Jiahao He, Shijie Zhao, Jiayi Ding, Yiming Wang
%    "Mirage search optimization: Application to path planning and engineering design problems."
%    Advances in Engineering Software, 2025.
%
% See also:
%       particleswarm, ga, patternsearch

% Author:                          cuixingxing
% Email:                           cuixingxing150@gmail.com
% Created:                         04-May-2025 18:37:24
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

%%
pos = initialization(SearchAgentsNumber, nvars, ub, lb);  % Initializing populations
pops = zeros(SearchAgentsNumber, 1);
nfes = 0;
cg_curve = zeros(1, MaxIterations);

% Initial fitness evaluation
if UseParallel
    parfor i = 1:SearchAgentsNumber
        pops(i) = fobj(pos(i, :));
    end
else
    for i = 1:SearchAgentsNumber
        pops(i) = fobj(pos(i, :));
    end
end

nfes = nfes + SearchAgentsNumber;
[fval, idx] = min(pops);
gbest = pos(idx, :);

% Setup plotting if enabled
if PlotFcns
    fig = figure('Name', 'Mirage Search Optimization', 'NumberTitle', 'off');
    ax = axes(fig);
    lineObj = animatedline(ax, 'LineWidth', 2, 'Color', 'b');
    xlabel(ax, 'Iteration');
    ylabel(ax, 'Best Objective Value');
    title(ax, 'Convergence Curve');
    grid(ax, 'on');

    % Add stop button
    stopButton = uicontrol(fig, 'Style', 'pushbutton', 'String', 'Stop', ...
        'Position', [20 20 50 20], 'Callback', @(src, event) stopCallback(src, event));
    stopButton.UserData = false; % Initialize stop flag
end

iter = 0;
while iter < MaxIterations
    iter = iter + 1;
    ac = randperm(SearchAgentsNumber-1) + 1;
    cv = ceil((SearchAgentsNumber*(2/3)) * ((MaxIterations - nfes + 1) / MaxIterations)); % Selection of individuals for Superior mirage search

    %% Superior mirage search
    % newPos = pos;
    % newPops = pops;
    if UseParallel
        % Temporary arrays for parallel updates
        tempCos = zeros(cv, 1);
        tempCosx = zeros(cv, nvars);
        parfor j = 1:cv
            idx = ac(j);
            cosx = zeros(1, nvars);
            for k = 1:nvars
                h = (gbest(k) - pos(idx, k)) * rand();
                cmax = 1;
                if h > 5 * atanh(-(nfes / MaxIterations) + 1) + cmax
                    h = 5 * atanh(-(nfes / MaxIterations) + 1) + cmax;
                end
                if h < cmax
                    h = cmax;
                end
                zf = randi(2) * 2 - 3;
                a = rand() * 20;
                b = rand() * (45 - a / 2);
                z = randi(2);
                if z == 1  % Case 1
                    C = b + 90;
                    D = 180 - C - a;
                    B = 180 - 2 * D;
                    A = 180 - B + a - 90;
                    dx = (sind(B) * h * sind(C)) / (sind(D) * sind(A));
                    dx = dx * zf;
                elseif z == 2 && a < b  % Case 2
                    C = 90 - b;
                    D = 90 + a - b;
                    B = 180 - 2 * D;
                    A = 180 - B - a - 90;
                    dx = (sind(B) * h * sind(C)) / (sind(D) * sind(A));
                    dx = dx * zf;
                elseif z == 2 && a > b  % Case 3
                    C = 90 - b;
                    D = 180 - C - a;
                    B = 180 - 2 * D;
                    A = 180 - B - 90 + a;
                    dx = (sind(B) * h * sind(C)) / (sind(D) * sind(A));
                    dx = dx * zf;
                else
                    dx = 0;
                end
                cosx(k) = pos(idx, k) + dx;
            end
            % Bound the variable
            Flag4ub = cosx > ub;
            Flag4lb = cosx < lb;
            cosx = (cosx .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
            tempCos(j) = fobj(cosx);
            tempCosx(j, :) = cosx;
        end
        nfes = nfes + cv;
        % Update population
        for j = 1:cv
            cos = tempCos(j);
            cosx = tempCosx(j, :);
            if fval > cos
                fval = cos;
                gbest = cosx;
            end
            pos = [pos; cosx];
            pops = [pops; cos];
        end
    else
        for j = 1:cv
            idx = ac(j);
            cosx = zeros(1, nvars);
            for k = 1:nvars
                h = (gbest(k) - pos(idx, k)) * rand();
                cmax = 1;
                if h > 5 * atanh(-(nfes / MaxIterations) + 1) + cmax
                    h = 5 * atanh(-(nfes / MaxIterations) + 1) + cmax;
                end
                if h < cmax
                    h = cmax;
                end
                zf = randi(2) * 2 - 3;
                a = rand() * 20;
                b = rand() * (45 - a / 2);
                z = randi(2);
                if z == 1  % Case 1
                    C = b + 90;
                    D = 180 - C - a;
                    B = 180 - 2 * D;
                    A = 180 - B + a - 90;
                    dx = (sind(B) * h * sind(C)) / (sind(D) * sind(A));
                    dx = dx * zf;
                elseif z == 2 && a < b  % Case 2
                    C = 90 - b;
                    D = 90 + a - b;
                    B = 180 - 2 * D;
                    A = 180 - B - a - 90;
                    dx = (sind(B) * h * sind(C)) / (sind(D) * sind(A));
                    dx = dx * zf;
                elseif z == 2 && a > b  % Case 3
                    C = 90 - b;
                    D = 180 - C - a;
                    B = 180 - 2 * D;
                    A = 180 - B - 90 + a;
                    dx = (sind(B) * h * sind(C)) / (sind(D) * sind(A));
                    dx = dx * zf;
                else
                    dx = 0;
                end
                cosx(k) = pos(idx, k) + dx;
            end
            % Bound the variable
            Flag4ub = cosx > ub;
            Flag4lb = cosx < lb;
            cosx = (cosx .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
            nfes = nfes + 1;
            cos = fobj(cosx);
            if fval > cos
                fval = cos;
                gbest = cosx;
            end
            pos = [pos; cosx];
            pops = [pops; cos];
        end
    end

    %% Selection of optimal individuals to renew the population
    tt = sortrows([pops, pos], 1);
    tt = tt(1:SearchAgentsNumber, :);
    pos = tt(:, 2:end);
    pops = tt(:, 1);

    %% Inferior mirage search
    if UseParallel
        % Temporary arrays for parallel updates
        tempCos = zeros(SearchAgentsNumber, 1);
        tempCosx = zeros(SearchAgentsNumber, nvars);
        parfor j = 1:SearchAgentsNumber
            if gbest ~= pos(j, :)
                hh = (gbest - pos(j, :));
            else
                hh = ones(1, nvars) * 0.05 * (randi(2) * 2 - 3);
            end
            zf = sign(hh);
            hh = abs(hh .* rand(1, nvars));
            gama = rand(1, nvars) .* 90 .* ((MaxIterations - nfes*0.99) / MaxIterations);
            amax = atand(1 ./ (2 * tand(gama)));
            amin = atand((sind(gama) .* cosd(gama)) ./ (1 + (sind(gama)).^2));
            fai = (amax - amin) .* rand() + amin;
            omg = asind(rand() .* sind(fai + gama));
            x = (hh ./ tand(gama)) - ((((hh ./ sind(gama)) - (hh .* sind(fai)) ./ (cosd(fai + gama))) .* cosd(omg)) ./ cosd(omg - gama));
            cosx = pos(j, :) + x .* zf;
            % Bound the variable
            Flag4ub = cosx > ub;
            Flag4lb = cosx < lb;
            cosx = (cosx .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
            tempCos(j) = fobj(cosx);
            tempCosx(j, :) = cosx;
        end
        nfes = nfes + SearchAgentsNumber;
        % Update population
        for j = 1:SearchAgentsNumber
            cos = tempCos(j);
            cosx = tempCosx(j, :);
            if fval > cos
                fval = cos;
                gbest = cosx;
            end
            pos = [pos; cosx];
            pops = [pops; cos];
        end
    else
        for j = 1:SearchAgentsNumber
            if gbest ~= pos(j, :)
                hh = (gbest - pos(j, :));
            else
                hh = ones(1, nvars) * 0.05 * (randi(2) * 2 - 3);
            end
            zf = sign(hh);
            hh = abs(hh .* rand(1, nvars));
            gama = rand(1, nvars) .* 90 .* ((MaxIterations - nfes*0.99) / MaxIterations);
            amax = atand(1 ./ (2 * tand(gama)));
            amin = atand((sind(gama) .* cosd(gama)) ./ (1 + (sind(gama)).^2));
            fai = (amax - amin) .* rand() + amin;
            omg = asind(rand() .* sind(fai + gama));
            x = (hh ./ tand(gama)) - ((((hh ./ sind(gama)) - (hh .* sind(fai)) ./ (cosd(fai + gama))) .* cosd(omg)) ./ cosd(omg - gama));
            cosx = pos(j, :) + x .* zf;
            % Bound the variable
            Flag4ub = cosx > ub;
            Flag4lb = cosx < lb;
            cosx = (cosx .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
            cos = fobj(cosx);
            nfes = nfes + 1;
            if fval > cos
                fval = cos;
                gbest = cosx;
            end
            pos = [pos; cosx];
            pops = [pops; cos];
        end
    end

    %% Selection of optimal individuals to renew the population
    tt = sortrows([pops, pos], 1);
    tt = tt(1:SearchAgentsNumber, :);
    pos = tt(:, 2:end);
    pops = tt(:, 1);
    cg_curve(iter) = fval;

    % Plot convergence curve if enabled
    if PlotFcns
        addpoints(lineObj, iter, fval);
        ax.Title.String = "Best Function Value: " + string(fval);
        drawnow;
        % Check stop button
        if stopButton.UserData
            break;
        end
    end
end

% Trim convergence curve if stopped early
if iter < MaxIterations
    cg_curve = cg_curve(1:iter);
end

if iter == MaxIterations && PlotFcns
    stopButton.Visible = "off";
end

% Callback function for stop button
    function stopCallback(src, ~)
        src.UserData = true;
        set(src, 'Visible', 'off');
    end
end

function Positions = initialization(SearchAgents_no, nvars, ub, lb)
Boundary_no = size(ub, 2);
if Boundary_no == 1
    Positions = rand(SearchAgents_no, nvars) .* (ub - lb) + lb;
else
    Positions = zeros(SearchAgents_no, nvars);
    for idx = 1:nvars
        Positions(:, idx) = rand(SearchAgents_no, 1) .* (ub(idx) - lb(idx)) + lb(idx);
    end
end
end