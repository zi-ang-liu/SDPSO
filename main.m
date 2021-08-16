%************************************************************************************************** 
%  SDPSO: strategy dynamics particle swarm optimizar 
%  Version: SDPSO 1.0
%  Author: Ziang Liu, Tatsushi Nishi
%  Email: liu.ziang@okayama-u.ac.jp
%  Date: 2021/08/16
%**************************************************************************************************

%% Clear workspace and command window
clc;
clear;

% random seed
rand('state', sum(100*clock));

% mex cec14_func.cpp -DWINDOWS
% f = cec14_func(x,func_num); here x is a D*pop_size matrix.

%D=10, 30, 50, 100; Runs / problem: 51;
%MaxFES: 10000*D (Max_FES for 10D = 100000; for 30D = 300000; for 50D = 500000; for 100D = 1000000)

 
%% Initialize parameters

% run 1 time
run=1; %30;

% dimension number
dimension=30;

% search range
range=[-100 100];

% optimal solution
optima = cumsum(100*ones(1,30));

% Preallocating a matrix for solutions
solution=zeros(30,run);

% parameter settings
pop=40;
pop1=floor(0.3*pop);
alpha=0.1;
r=4;
lp=200;

% number of function evaluations
max_FES=10000*dimension;
% number of iterations
max_iteration=ceil(max_FES/pop);
% Preallocating a matrix for convergence graphs
con_graph=ones(run,max_FES,30);

%% Iteration

for func_num=1:30
    
    for i=1:run
        
        % SDPSO
        [position, value,convergence] = SDPSO(pop,range,dimension,max_iteration,max_FES,func_num,r,lp,alpha,pop1);
        
        % data for convergence graph
        con_graph(i,:,func_num)=convergence(1:max_FES);
        
        % record error
        solution(func_num,i) = value-optima(func_num);
        
    end
    
    % mean error
    m = mean(solution(func_num,:),2);
    
    % standard deviation
    s = std(solution(func_num,:),0,2);
    
    % output the results
    fprintf('Func_%d\n Mean:\t%d\n Std:\t%d\n', func_num, m, s);
    
end

%% Save

% algo_name='SDPSO_';
% file_name= [algo_name,num2str(pop),'pop_',num2str(dimension),'D_',num2str(run),'run','.mat'];
% save(file_name,'solution')

% convergence_name= [algo_name,'convergence_',num2str(popsize),'pop_',num2str(dimension),'D_',num2str(run),'run','.mat'];
% save(convergence_name,'con_graph', '-v7.3')

%% convergence graphs

% for i=1:30
%     figure
%     semilogy(median(con_graph(:,:,i)));
% end


