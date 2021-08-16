function [position, value,convergence] = SDPSO(popsize,range,dimension,max_iteration,max_FES,func_num,fe,co,alpha,num_g1)

%************************************************************************************************** 
%  SDPSO: strategy dynamics particle swarm optimizar 
%  Version: SDPSO 1.0
%  Author: Ziang Liu, Tatsushi Nishi
%  Email: liu.ziang@okayama-u.ac.jp
%  Date: 2021/08/16
%**************************************************************************************************

%% Initialize parameters

% Search range
range_min=range(1)*ones(popsize,dimension);
range_max=range(2)*ones(popsize,dimension);

interval = range_max-range_min;
v_max=interval*0.5;
v_min=-v_max;

% Initialize positions and velocities
pos = range_min+ interval.*rand(popsize,dimension);
vel =v_min+(v_max-v_min).*rand(popsize,dimension);

%strategy number and initial probaility
strategy_num = 4;
p=ones(1,strategy_num)*(1/strategy_num);

% Preallocation
stop_num=zeros(1,popsize);
fri_best_pos=zeros(popsize,dimension);

success_mem=zeros(1,strategy_num);
failure_mem=zeros(1,strategy_num);
rk=cumsum(ones(1,strategy_num)./strategy_num);
strategy_improve=zeros(1,strategy_num);

%% Initialize parameters for each strategies

% CLPSO
c1=3-(1:max_iteration)*1.5/max_iteration;
w1=0.9-(1:max_iteration)*(0.7/max_iteration); 
% refreshing gap
m=5;
obj_func_slope=zeros(popsize,1);
fri_best=(1:popsize)'*ones(1,dimension);
j=0:(1/(popsize-1)):1;
j=j*10;
Pc=ones(dimension,1)*(0.0+((0.5).*(exp(j)-exp(j(1)))./(exp(j(popsize))-exp(j(1)))));

% LDWPSO
w2=0.9-(1:max_iteration)*(0.7/max_iteration); 
c2_1=2.5-(1:max_iteration)*2/max_iteration;
c2_2=0.5+(1:max_iteration)*2/max_iteration;

% LIPS
nsize=3;

% UPSO
c3_1=2.5-(1:max_iteration)*2/max_iteration;
c3_2=0.5+(1:max_iteration)*2/max_iteration;
w3=0.9-(1:max_iteration)*(0.7/max_iteration);
u_3=0.1;
mu_3=0;
sigma_3=0.01;
nor=normrnd(mu_3,sigma_3,max_FES,dimension);
neighbor_3(1,:)=[popsize,2];
for i=2:(popsize-1)
    neighbor_3(i,:)=[i-1,i+1];
end
neighbor_3(popsize,:)=[popsize-1,1];

%% Initial fitness evaluation

% the number of function evaluations
fitcount=0;

% function evaluation (CEC 2014)
result = (cec14_func(pos',func_num))';
fitcount=fitcount+popsize;

% update gbest 
[gbest_val,g_index]=min(result);
gbest_pos=pos(g_index,:); 

% update pbest
pbest_pos=pos; 
pbest_val=result';

% convergence graph
convergence(1:fitcount)=gbest_val;

%% Initialize CLPSO
for i=1:num_g1
    fri_best(i,:)=i*ones(1,dimension);
    friend1=ceil(popsize*rand(1,dimension));
    friend2=ceil(popsize*rand(1,dimension));
    friend=(pbest_val(friend1)<pbest_val(friend2)).*friend1+(pbest_val(friend1)>=pbest_val(friend2)).*friend2;
    toss=ceil(rand(1,dimension)-Pc(:,i)');
    if toss==ones(1,dimension)
        temp_index=randperm(dimension);
        toss(1,temp_index(1))=0;
        clear temp_index;
    end
    fri_best(i,:)=(1-toss).*friend+toss.*fri_best(i,:);
    for d=1:dimension
        fri_best_pos(i,d)=pbest_pos(fri_best(i,d),d);
    end
end

for i=num_g1+1:popsize
    fri_best(i,:)=i*ones(1,dimension);
    friend1=ceil(popsize*rand(1,dimension));
    friend2=ceil(popsize*rand(1,dimension));
    friend=(pbest_val(friend1)<pbest_val(friend2)).*friend1+(pbest_val(friend1)>=pbest_val(friend2)).*friend2;
    toss=ceil(rand(1,dimension)-Pc(:,i)');
    if toss==ones(1,dimension)
        temp_index=randperm(dimension);
        toss(1,temp_index(1))=0;
        clear temp_index;
    end
    fri_best(i,:)=(1-toss).*friend+toss.*fri_best(i,:);
    for d=1:dimension
        fri_best_pos(i,d)=pbest_pos(fri_best(i,d),d);
    end
end

%% Iteration

k=0;

while k<=max_iteration && fitcount<=max_FES
    
    k=k+1;
    gbest_pos_temp=repmat(gbest_pos,popsize,1);

    % Pop1: CLPSO
    for i=1:num_g1
        if obj_func_slope(i)>m
            fri_best(i,:)=i*ones(1,dimension);
            friend1=(ceil(num_g1*rand(1,dimension)));
            friend2=(ceil(num_g1*rand(1,dimension)));
            friend=(pbest_val(friend1)<pbest_val(friend2)).*friend1+(pbest_val(friend1)>=pbest_val(friend2)).*friend2;
            toss=ceil(rand(1,dimension)-Pc(:,i)');
            if toss==ones(1,dimension)
                temp_index=randperm(dimension);
                toss(1,temp_index(1))=0;
                clear temp_index;
            end
            fri_best(i,:)=(1-toss).*friend+toss.*fri_best(i,:);
            for d=1:dimension
                fri_best_pos(i,d)=pbest_pos(fri_best(i,d),d);
            end
            obj_func_slope(i)=0;
        end
        
        % update positions
        delta(i,:)=(c1(k).*rand(1,dimension).*(fri_best_pos(i,:)-pos(i,:)));
        vel(i,:)=w1(k)*vel(i,:)+delta(i,:);
        vel(i,:)=((vel(i,:)<v_min(i,:)).*v_min(i,:))+((vel(i,:)>v_max(i,:)).*v_max(i,:))+(((vel(i,:)<v_max(i,:))&(vel(i,:)>v_min(i,:))).*vel(i,:));
        pos(i,:)=pos(i,:)+vel(i,:);
        
        if (sum(pos(i,:)>range_max(i,:))+sum(pos(i,:)<range_min(i,:))==0)
            % Fitness evaluation
            result(i) = (cec14_func(pos(i,:)',func_num))';
            fitcount=fitcount+1;
            
            convergence(fitcount)=gbest_val;
            
            % Restart
            if mod(fitcount,max_FES/fe)==0
                p=ones(1,strategy_num)*(1/strategy_num);
            end
            
            % Update the adoption probability for each strategy
            if mod(fitcount,co)==0
                total = (success_mem+failure_mem);
                total(find(total==0))=1;
                strategy_improve=strategy_improve./total;
                if isequal(strategy_improve,zeros(1,strategy_num))
                    strategy_improve=ones(1,strategy_num);
                end
                strategy_improve(find(strategy_improve==0))=0.1*min(strategy_improve(strategy_improve~=0));
                strategy_improve=strategy_improve./sum(strategy_improve);
                f=strategy_improve;
                p=(f-sum(p.*f)).*p.*alpha+p;
                p(find(p<=0.02))=0.02;
                p=p./(sum(p));
                rk =cumsum(p);
                success_mem = zeros(1,strategy_num);
                failure_mem = zeros(1,strategy_num);
                strategy_improve=zeros(1,strategy_num);
            end
            
            if fitcount>=max_FES
                break;
            end
            
            % update pbest
            if  result(i)<pbest_val(i)
                pbest_pos(i,:)=pos(i,:);
                pbest_val(i)=result(i);
                obj_func_slope(i)=0;
            else
                obj_func_slope(i)=obj_func_slope(i)+1;
            end
            
            % update gbest
            if  pbest_val(i)<gbest_val 
                gbest_pos=pbest_pos(i,:);
                gbest_val=pbest_val(i);
            end
        end
    end
    
    % Pop2: LDWPSO, UPSO, LIPS, CLPSO
    for i=num_g1+1:popsize
        
        probility=rand;
        
        %LIPS
        if probility<=rk(1)
            strategy_k = 1;
            EU_dist=mydist(pos(i,:),pbest_pos');
            EU_dist(i)=max(EU_dist);
            [min_dist,min_index]=sort(EU_dist);
            fi=(4.1./nsize).*rand(nsize,dimension);
            FIP=sum(fi.*pbest_pos(min_index(1:nsize),:))./sum(fi);
            delta(i,:)=sum(fi).*(FIP-pos(i,:));
            vel(i,:)=0.7298.*(vel(i,:)+delta(i,:));
            
            
        %UPSO update
        elseif probility<=rk(2)
            strategy_k=2;
            [tmp,tmpid]=min(pbest_val(neighbor_3(i,:)));
            aa1(i,:)=c3_1(k).*rand(1,dimension).*(pbest_pos(i,:)-pos(i,:))+c3_2(k).*rand(1,dimension).*(gbest_pos-pos(i,:));
            vel1(i,:)=w3(k).*vel(i,:)+aa1(i,:);
            aa2(i,:)=c3_1(k).*rand(1,dimension).*(pbest_pos(i,:)-pos(i,:))+c3_2(k).*rand(1,dimension).*(pbest_pos(neighbor_3(i,tmpid),:)-pos(i,:));
            vel2(i,:)=w3(k).*vel(i,:)+aa2(i,:);
            r3_6=nor(fitcount,:);
            vel(i,:)=r3_6.*u_3.*vel1(i,:)+(1-u_3).*vel2(i,:);
            
        % LDWPSO
        elseif probility<=rk(3)
            strategy_k = 3;
            delta(i,:)=c2_1(k).*rand(1,dimension).*(pbest_pos(i,:)-pos(i,:))+c2_2(k).*rand(1,dimension).*(gbest_pos_temp(i,:)-pos(i,:));
            vel(i,:)=w2(k).*vel(i,:)+delta(i,:);
            
        %CLPSO
        elseif probility<=rk(4)
            strategy_k = 4;
            delta(i,:)=(c2_1(k).*rand(1,dimension).*(fri_best_pos(i,:)-pos(i,:)));
            vel(i,:)=w2(k)*vel(i,:)+delta(i,:);
            if obj_func_slope(i)>m
                fri_best(i,:)=i*ones(1,dimension);
                friend1=(ceil(popsize*rand(1,dimension)));
                friend2=(ceil(popsize*rand(1,dimension)));
                friend=(pbest_val(friend1)<pbest_val(friend2)).*friend1+(pbest_val(friend1)>=pbest_val(friend2)).*friend2;
                toss=ceil(rand(1,dimension)-Pc(:,i)');
                if toss==ones(1,dimension)
                    temp_index=randperm(dimension);
                    toss(1,temp_index(1))=0;
                    clear temp_index;
                end
                fri_best(i,:)=(1-toss).*friend+toss.*fri_best(i,:);
                for d=1:dimension
                    fri_best_pos(i,d)=pbest_pos(fri_best(i,d),d);
                end
                obj_func_slope(i)=0;
            end
        end
        
        % Update positions
        vel(i,:)=((vel(i,:)<v_min(i,:)).*v_min(i,:))+((vel(i,:)>v_max(i,:)).*v_max(i,:))+(((vel(i,:)<v_max(i,:))&(vel(i,:)>v_min(i,:))).*vel(i,:));
        pos(i,:)=pos(i,:)+vel(i,:);
        
        if (sum(pos(i,:)>range_max(i,:))+sum(pos(i,:)<range_min(i,:))==0)
            result(i) = (cec14_func(pos(i,:)',func_num))';
            fitcount=fitcount+1;
            
            convergence(fitcount)=gbest_val;
            
            % restart
            if mod(fitcount,max_FES/fe)==0
                p=ones(1,strategy_num)*(1/strategy_num);
            end
            
            % calculate the population states
            if mod(fitcount,co)==0
                total = (success_mem+failure_mem);
                total(find(total==0))=1;
                strategy_improve=strategy_improve./total;
                if isequal(strategy_improve,zeros(1,strategy_num))
                    strategy_improve=ones(1,strategy_num);
                end
                strategy_improve(find(strategy_improve==0))=0.1*min(strategy_improve(strategy_improve~=0));
                strategy_improve=strategy_improve./sum(strategy_improve);
                f=strategy_improve;
                p=(f-sum(p.*f)).*p.*alpha+p;
                p(find(p<=0.02))=0.02; 
                p=p./(sum(p));
                rk =cumsum(p);
                success_mem = zeros(1,strategy_num);
                failure_mem = zeros(1,strategy_num);
                strategy_improve=zeros(1,strategy_num);
            end
            
            if fitcount>=max_FES
                break;
            end
            
            % update pbest
            if  result(i)<pbest_val(i) 
                strategy_improve(strategy_k)=strategy_improve(strategy_k)+(pbest_val(i)-result(i))/pbest_val(i);
                pbest_pos(i,:)=pos(i,:);
                pbest_val(i)=result(i);
                success_mem(strategy_k) = success_mem(strategy_k) +1;
            else
                failure_mem(strategy_k) = failure_mem(strategy_k) + 1;
            end
            if result(i)<pbest_val(i)
                stop_num(i)=0;
                if strategy_k==4
                    obj_func_slope(i)=0;
                end
            else
                stop_num(i)=stop_num(i)+1;
                obj_func_slope(i)=obj_func_slope(i)+1;
            end
            
            % update gbest
            if  pbest_val(i)<gbest_val 
                gbest_pos=pbest_pos(i,:);
                gbest_val=pbest_val(i);
            end
        end
    end
    
    if fitcount>=max_FES
        break;
    end
    
    if (k==max_iteration)&&(fitcount<max_FES)
        k=k-1;
    end
    
end

% record the best solution
position=gbest_pos;
value=gbest_val;

end

