%_______________________________________________________________________________________%
%  Mirage Search Optimization (MSO)
%  Developed in MATLAB R2022a
%
%  programmer: Shijie Zhao and Jiahao He
%  E-mail: zhaoshijie@lntu.edu.cn
%          lntuhjh@126.com
%  The code is based on the following papers.
%  Jiahao He, Shijie Zhao, Jiayi Ding, Yiming Wang
%  Mirage search optimization: Application to path planning and engineering design problems
%  Advances in Engineering Software
%_______________________________________________________________________________________%
function [gbest,fval, cg_curve] = MSO(fobj, dim, lb, ub, PopSize, MaxIter)

fval = inf;
pos    = initialization(PopSize, dim, ub, lb);  %Initializing populations
nfes = 0;

for i = 1:PopSize
    pops(i) = fobj(pos(i, :));
    nfes = nfes + 1;
    if fval > pops(i)
        fval = pops(i);
        gbest  = pos(i, :);
    end
end
pops = pops';
iter = 0;
while iter<MaxIter
    iter = iter+1;
    ac = randperm(PopSize-1) + 1;
    cv = ceil((PopSize*(2/3)) * ((MaxIter - nfes + 1) / MaxIter));%Selection of individuals for Superior mirage search
    %% Superior mirage search
    for j = ac(1 : cv)  
        for k = 1:dim
            h    = (gbest(k) - pos(j, k)) * rand();
            cmax = 1;
            if h > 5 * atanh( -(nfes / MaxIter) + 1) + cmax
                h = 5 * atanh( -(nfes / MaxIter) + 1) + cmax;
            end
            if h < cmax
                h = cmax;
            end
            zf = randi(2) * 2 - 3;
            a  = rand() * 20;
            b  = rand() * (45 - a / 2);
            z  = randi(2);
            if z == 1  %Case 1
                C  = b + 90;
                D  = 180 - C - a;
                B  = 180 - 2 * D;
                A  = 180 - B + a - 90;
                dx = ( sind(B) * h * sind(C) ) / ( sind(D) * sind(A) );
                dx = dx * zf;
            elseif z == 2 && a < b  %Case 2
                C  = 90 - b;
                D  = 90 + a - b;
                B  = 180 - 2 * D;
                A  = 180 - B - a - 90;
                dx = ( sind(B) * h * sind(C) ) / ( sind(D) * sind(A) );
                dx = dx * zf;
            elseif z == 2 && a > b  %Case 3
                C  = 90 - b;
                D  = 180 - C - a;
                B  = 180 - 2 * D;
                A  = 180 - B - 90 + a;
                dx = ( sind(B) * h * sind(C) ) / ( sind(D) * sind(A) );
                dx = dx * zf;
            else
                dx = 0;
            end
            cosx(:,k) = pos(j, k) + dx;
        end
        %Bound the variable
        Flag4ub   = cosx > ub;
        Flag4lb   = cosx < lb;
        cosx = (cosx .* ( ~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
        nfes = nfes + 1;
        cos = fobj( cosx);
        if fval > cos
            fval = cos;
            gbest  = cosx;
        end
        pos = [pos;cosx];
        pops = [pops;cos];
    end
    %% Selection of optimal individuals to renew the population
    tt = sortrows([pops, pos], 1);
    tt = tt(1:PopSize,:);
    pos= tt(:, 2 : end);
    pops = tt(:,1);
    %% Inferior mirage search
    for j = 1:PopSize  
        if gbest ~= pos(j, :)
            hh = (gbest - pos(j, :)) ;
        else
            hh = ones(1, dim) * 0.05 *( randi(2) * 2 - 3);
        end
        zf   = sign(hh);
        hh   = abs(hh .* rand(1, dim));
        gama = rand(1, dim) .* 90.* ((MaxIter - nfes*0.99) / MaxIter);
        amax = atand(1 ./ (2 * tand(gama)));
        amin = atand((sind(gama) .* cosd(gama)) ./ (1 + (sind(gama)) .^ 2));
        fai  = (amax - amin) .* rand() + amin ;
        omg  = asind(rand() .* sind(fai + gama)) ;
        x    = (hh ./ tand(gama)) - ((((hh ./ sind(gama)) - (hh .* sind(fai)) ./ (cosd( fai + gama))) .* cosd(omg)) ./ cosd(omg - gama));
        cosx = pos(j, :) + x .* zf;
        %Bound the variable
        Flag4ub   = cosx > ub;  
        Flag4lb   = cosx < lb;
        cosx = (cosx .* ( ~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
        cos = fobj(cosx);
        nfes = nfes + 1;
        if fval  > cos
            fval = cos;
            gbest  = cosx;
        end
        pos = [pos;cosx];
        pops = [pops;cos];
    end
    %% Selection of optimal individuals to renew the population
    tt  = sortrows([pops, pos], 1);
    tt = tt(1:PopSize,:);
    pos = tt(:, 2 : end);
    pops = tt(:, 1);
    cg_curve(iter) = fval;
end


function Positions = initialization(SearchAgents_no, dim, ub, lb)
Boundary_no= length(ub); % numnber of boundaries
% If the boundaries of all variables are equal and user enter a signle
% number for both ub and lb
if Boundary_no == 1
    Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
end
% If each variable has a different lb and ub
if Boundary_no > 1
    for i = 1:dim
        ub_i = ub(i);
        lb_i = lb(i);
        Positions(:, i) = rand(SearchAgents_no, 1) .* (ub_i - lb_i) + lb_i;
    end
end