clear
clc

%% load data
load('task1.mat');

%% Initialization
gamma = 0.9; % 0.5,0.9 
pattern = 1; % 1,2,3,4

%% main iteration
reach_times = 0;
max_reward = 0;
execution_t = 0;

% run 10 times
for run = 1:10
    fprintf("RUN %d", run);
    tic; % timer start
    Q = zeros(100,4);
    
    % loop walk 3000 times
    for epoch = 1:3000 
        % initialize trial parameters
        k = 1;
        s_k = 1;
        Q_old = Q;
        
        while s_k < 100  
            % initialize state parameters
            [epsilon_k, alpha_k] = updateLR(k,pattern);
            if epsilon_k < 0.005 % no exploration any more
                break
            end
            % select action
            a_k = selectAction(Q(s_k,:),epsilon_k,reward(s_k,:));  % a_k, s_k are index for Q matrix
            % update Q function
            [Q(s_k,a_k),s_k] = UpdateQ(Q,reward,s_k,a_k,alpha_k,gamma); % update s_k, move to the next grid
            % update k
            k = k + 1;
        end
        
        % check convergence
        diff = abs(Q-Q_old);
        sum_val = max(diff);
        if sum_val  < 0.001 
            fprintf(" \n converge! \n");
            break
        end
    end
    
    % reward shaping
    [map_label , reach,policy,policy_reward,state_list,marker,x,y,total_reward] = optimal_pi(Q,reward,gamma);
    % update optical result
    if total_reward > max_reward
        op_map_label = map_label;
        max_reward = total_reward;
        op_x = x;
        op_y = y;
        op_marker = marker;
        op_policy = policy;
        op_policy_reward = policy_reward;
        op_state_list = state_list;
    end
    toc;
    
    % calculate goal reach number
    if reach == 1
        fprintf("Goal reach!! \n ");
        reach_times = reach_times + 1;
        execution_t = execution_t + toc;
    else
        fprintf("Nor reach..... \n ");
    end
    
end

fprintf("No. of goal-reached run: %d \n ",reach_times);
fprintf("Execution time: %.4f s \n ",execution_t / reach_times );

%% Draw max reward policy
figure()
execution_time= round( execution_t / reach_times,3) ;
% tranform arrow marker
op_marker(op_marker == 1) = '<';
op_marker(op_marker == 2) = 'v';
op_marker(op_marker == 3) = '>';
op_marker(op_marker == 4) = '^';

axis ij;
xlim([-1,11]);
ylim([-1,11]);
grid on;
hold on;
title({['\gamma = ',num2str(gamma),' , execution time = ',num2str(execution_time)]});
for i = 1:length(op_x)
    scatter(op_x(i),op_y(i)+1, 55 ,'b', char(op_marker(i)) );
    %terminal
    scatter(10,10,300, 'filled' , 'pr');
end
hold off;


%%
% optimal policy
function [map_label,reach,policy_s,policy_reward,state_list,marker,x,y,total_reward] = optimal_pi(Q,reward,gamma)
    [~,policy_s]=max(Q,[],2);  % get optimal policy at state s policy_s is a column vec contain max ak idx
    step = 1;
    x = []; % coordinate x
    y = [];
    s = 1;
    marker = [];
    map_label = zeros(10,10);
    policy_reward = zeros(10,10);
    state_list = [];

    while s < 100  && policy_reward(mod(s,10)+1,fix(s/10)+1)==0 % mod = vertical, fix = horizontal
        state_list = [state_list, s];
        x = [x,mod(s,10)];
        y = [y,fix(s/10)];
        switch policy_s(s)
            case 1  %left
                map_label(mod(s,10)+1,fix(s/10)+1) = '<';
                marker = [marker,1];
                policy_reward(mod(s,10)+1,fix(s/10)+1) = gamma^(step-1)*reward(s,1);
                s=s-1;
            case 2  %down
                map_label(mod(s,10)+1,fix(s/10)+1) = 'v';
                marker = [marker,2];
                policy_reward(mod(s,10)+1,fix(s/10)+1) = gamma^(step-1)*reward(s,2);
                s=s+10;

            case 3  %right
                map_label(mod(s,10),fix(s/10)+1) = '>';
                marker = [marker,3];
                policy_reward(mod(s,10)+1,fix(s/10)+1) = gamma^(step-1)*reward(s,3);
                s=s+1;
            case 4 % up
                map_label(mod(s,10),fix(s/10)+1) = '^';
                marker = [marker,4];
                policy_reward(mod(s,10)+1,fix(s/10)+1) = gamma^(step-1)*reward(s,4);
                s=s-10;
        end
        step = step + 1;
    end
    if s == 100
        state_list = [state_list s]; 
        reach = 1;
    else
        reach = 0;
    end
    total_reward = sum(policy_reward(:));
end


% update Q function
function [Q_new,s_k_new] = UpdateQ(Q,reward,s,a_k,alpha_k,gamma)
    switch a_k
        case 1  % left
            Q_new=Q(s,a_k)+ alpha_k *(reward(s,a_k)+gamma*max(Q(s-1,:))-Q(s,a_k)); % move left
            s_k_new=s-1;
        case 2 %down
            Q_new=Q(s,a_k)+ alpha_k *(reward(s,a_k)+gamma*max(Q(s+10,:))-Q(s,a_k)); % move down
            s_k_new=s+10;
        case 3 %right
            Q_new=Q(s,a_k)+alpha_k *(reward(s,a_k)+gamma*max(Q(s+1,:))-Q(s,a_k)); % move right
            s_k_new=s+1;
        case 4 %up
            Q_new=Q(s,a_k)+alpha_k*(reward(s,a_k)+gamma*max(Q(s-10,:))-Q(s,a_k)); % move up
            s_k_new=s-10;
    end
end

% select a for current state
function action = selectAction(Q_s,epsilon_k,ak_choice_at_Q)
    
    % disgard margin action a=4:up, a=1:left, a=3:right, a=2:down
    valid_ak=find(ak_choice_at_Q ~=-1);
    
    ak_update_rule = rand;
    if any(Q_s)
        % max Q index
        [~,max_ak_idx] = max(Q_s(valid_ak));
        % Exploitation
        if ak_update_rule>=epsilon_k  
            action=valid_ak(max_ak_idx);
        % Exploration
        else           
            other_idx=find(Q_s(valid_ak)~= max(Q_s(valid_ak)) );
            action_idx = randi(length(other_idx));
            action = valid_ak(other_idx(action_idx));
        end
    else
        initial_idx=randperm(length(valid_ak),1); 
        action=valid_ak(initial_idx);
    end 
end


function [epsilon_k, alpha_k] = updateLR(k,pattern)
    switch pattern
        case 1
            epsilon_k = 1 / k;
        case 2
            epsilon_k = 100 / (100 + k);
        case 3
            epsilon_k = (1 + log(k)) / k;
        case 4
            epsilon_k = (1 + 5 * log(k)) / k;

    end
    alpha_k = epsilon_k;
end