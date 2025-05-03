%% 一些元启发式进化算法比较
% 主要是对近年来goldenjackal(2022)，starfish(2025)一些群体智能优化算法做一些性能测试比较，体验其差异。对原始算法做了进一步的高度优化，比如并行处理，并支持实时性能绘图，规范代码书写，以符合MATLAB官方内置优化形式，目的在于提供易于直接使用的新型算法，可快速集成到自己的工程项目中！
% 
% starfish收敛速度相对最快，goldenjackal优化结果相对较好，但与starfish优化结果差距不大，miragesearch和particleswarm收敛和最优解相对较差。

% Requirements
%% 
% * MATLAB R2025a or later
%% 
% * Global Optimization Toolbox™
% Algorithm Setup
% 选择合适的评估算法和一些超参设置。

addpath("algorithms","utils")
f = @rosenbrock;% 性能测评函数，可自由下拉条选择评估函数
nvars = 12; % 优化维度数量，可自由定制
lb = -10 * ones(1, nvars); % 下边界，可自由定制
ub = 10 * ones(1, nvars);% 上边界，可自由定制
Npop = 50; % 群体数量，可自由定制
Max_it = 100;% 迭代终止最大次数，可自由定制
% Optimization Algorithms

t1 = tic;
[xposbest1,fvalbest1,Curve1] = starfish(f,nvars,lb,ub,SearchAgentsNumber=Npop,MaxIterations=Max_it,PlotFcns=true,UseParallel=false);
toc(t1)

t2 = tic;
[xposbest2,fvalbest2,Curve2] = goldenjackal(f,nvars,lb,ub,SearchAgentsNumber=Npop,MaxIterations=Max_it,PlotFcns=true,UseParallel=false);
toc(t2)
%% 
% 由于内置的<https://au.mathworks.com/help/gads/particleswarm.html particleswarm>函数不易拿到内部每次迭代的函数值，故通过记录log方式打印每次迭代结果，然后解析Iteration-f(x)对应的值，最后综合相关算法做性能对比绘图。
% 
% *最好以传统的".m"文件形式运行本小节的代码！*

diary pso_iter.txt % 这个diary函数对mlx内嵌输出不记录log！所以最好以传统的“.m”文件形式运行本小节的代码！
options = optimoptions('particleswarm', UseParallel=false, PlotFcn= @pswplotbestf, SwarmSize=Npop,MaxIterations=Max_it,Display="iter");
t3 = tic;
[xposbest3,fvalbest3,exitflag,output] = particleswarm(f, nvars, lb, ub, options);
toc(t3)
diary off

[iterations, bestFvals] = extractPSOLogData("pso_iter.txt");
delete("pso_iter.txt");
Curve3 = bestFvals(:)';

t4 = tic;
[xposbest4,fvalbest4,Curve4] = MSO(f,nvars,lb,ub,Npop,Max_it);
toc(t4)

%% Performance Curve
figure;
grid on;
hold on;

plot([Curve1;Curve2;Curve3;Curve4]',LineWidth=2)
xlabel("Iteration");
ylabel("Function Value");
title("Convergence Performance Curves of Various Algorithms")
legend(["starfish:"+string(vpa(fvalbest1,10)),...
    "goldenjackal:"+string(vpa(fvalbest2,10)),...
    "particleswarm:"+string(vpa(fvalbest3,10)),...
     "miragesearch:"+string(vpa(fvalbest4,10))])

%% Some Benchmark Performance Functions

% Optimization Test Functions for Performance Evaluation
% These functions are designed to test optimization algorithms on complex,
% multi-modal, non-convex problems with multiple local optima.
% Each function is implemented for n-dimensional input with boundary constraints.

% 1.complexObjective
function y = complexObjective(x)
% x 是一个 n 维向量
% Rastrigin函数（多峰函数，优化难度较大）
A = 10;
n = length(x);
rastrigin = A * n + sum(x.^2 - A * cos(2 * pi * x));

% 计算密集型模拟部分（人为加入耗时）
pauseTime = 0.01; % 模拟每次评估耗时0.01秒
tic; while toc < pauseTime; end

% 非线性耦合项
coupling = 0;
for i = 1:n-1
    coupling = coupling + 100*(x(i+1)-x(i)^2)^2 + (1 - x(i))^2;
end

% 组合目标函数
y = rastrigin + coupling;
end

% 2.multiModalTestFunc
function y = multiModalTestFunc(x)  
% multiModalTestFunc - 多极点复杂测试函数  
% 输入:  
%   x - n维向量，优化变量  
%  
% 输出:  
%   y - 函数值，目标函数输出，越小越优  

% 参数设置  
n = length(x);  
m = 5;  % 多极点数目  

% 初始化极点位置，可以随机也可以预设  
% 这里预设m个极点，每个极点在n维空间里随机分布[-5,5]  
persistent poles  
if isempty(poles)  
    rng(1); % 保证重复性  
    poles = 10 * rand(m, n) - 5;  
end  

% 每个极点对应的高斯峰，权重和宽度可以不一样  
weights = linspace(1, 2, m);          % 权重，也可以自定义  
sigmas = linspace(0.5, 1.5, m);      % 峰的宽度  

% 计算每个极点上的高斯型函数值并组合  
y = 0;  
for i=1:m  
    diff = x - poles(i, :);  
    distSq = diff * diff'; % 欧氏距离平方  
    % 多极点负高斯峰，目标函数值为负峰叠加 (需要最小化)  
    y = y - weights(i) * exp(-distSq / (2 * sigmas(i)^2));  
end  

% 加入一个全局趋势，比如一个简单的凹函数，增加搜索难度  
y = abs(y) + 0.01 * sum(x.^2);  
end  


% 3. Rastrigin Function
% Characteristics: Highly multi-modal, many local minima, non-convex, symmetric
% Global minimum: f(x) = 0 at x = [0, 0, ..., 0]
% Bounds: x_i in [-5.12, 5.12]
function f = rastrigin(x)
    n = length(x);
    A = 10;
    f = A * n + sum(x.^2 - A * cos(2 * pi * x));
end

% 4. Ackley Function
% Characteristics: Multi-modal, non-convex, nearly flat outer region, deep global minimum
% Global minimum: f(x) = 0 at x = [0, 0, ..., 0]
% Bounds: x_i in [-32.768, 32.768]
function f = ackley(x)
    n = length(x);
    a = 20;
    b = 0.2;
    c = 2 * pi;
    sum1 = sum(x.^2);
    sum2 = sum(cos(c * x));
    f = -a * exp(-b * sqrt(sum1 / n)) - exp(sum2 / n) + a + exp(1);
end

% 5. Griewank Function
% Characteristics: Multi-modal, non-convex, many widespread local minima
% Global minimum: f(x) = 0 at x = [0, 0, ..., 0]
% Bounds: x_i in [-600, 600]
function f = griewank(x)
    n = length(x);
    sum_term = sum(x.^2 / 4000);
    prod_term = prod(cos(x ./ sqrt(1:n)));
    f = 1 + sum_term - prod_term;
end

% 6. Schwefel Function
% Characteristics: Multi-modal, non-convex, deceptive global minimum far from local minima
% Global minimum: f(x) = 0 at x = [420.9687, 420.9687, ..., 420.9687]
% Bounds: x_i in [-500, 500]
function f = schwefel(x)
    n = length(x);
    f = 418.9829 * n - sum(x .* sin(sqrt(abs(x))));
end

% 7. Rosenbrock Function
% Characteristics: Non-convex, narrow parabolic valley, difficult to converge to global minimum
% Global minimum: f(x) = 0 at x = [1, 1, ..., 1]
% Bounds: x_i in [-5, 10]
function f = rosenbrock(x)
    n = length(x);
    f = sum(100 * (x(2:n) - x(1:n-1).^2).^2 + (1 - x(1:n-1)).^2);
end

% References
%% 
% # Chopra, Nitish, and Muhammad Mohsin Ansari. "Golden Jackal Optimization:A 
% Novel Nature-Inspired Optimizer for Engineering Applications." Expert Systems 
% with Applications (2022): 116924.
% # Changting Zhong, Gang Li, Zeng Meng, Haijiang Li, Ali Riza Yildiz, Seyedali 
% Mirjalili."Starfish Optimization Algorithm (SFOA): A bio-inspired metaheuristic 
% algorithm for global optimization compared with 100 optimizers. " Neural Computing 
% and Applications(2025), 37: 3641-3683.
% # Jiahao He, Shijie Zhao, Jiayi Ding, Yiming Wang. "Mirage search optimization: 
% Application to path planning and engineering design problems." Advances in Engineering 
% Software(2025), DOI: https://doi.org/10.1016/j.advengsoft.2025.103883