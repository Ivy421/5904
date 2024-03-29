tr_data = load('train.mat');
te_data = load('test.mat');
x_train = tr_data.train_data ; % 57*2000, 57 dimensions and 2000 samples
y_train = tr_data.train_label; % 2000*1
x_test = te_data.test_data;
y_test = te_data.test_label;

% min-max normalization
% x_train_trans = x_train';
% min_val = min( x_train_trans);
% max_val = max( x_train_trans);
% x_train_trans = (x_train_trans-min_val)./(max_val - min_val);
% x_train = x_train_trans';
% x_te_trans = (x_test'-min_val)./(max_val - min_val);
% x_test = x_te_trans';

% std normalization
mean_vals = mean(x_train'); 
std_vals = std(x_train');  
x_train_trns = (x_train' - mean_vals) ./ std_vals;
x_test_trns = (x_test' - mean_vals) ./ std_vals;
x_train = x_train_trns';
x_test = x_test_trns';

n = size(x_train,2);
d = size(x_train,1);
n_test = size(x_test,2);

gram_mat = zeros(n,n);
H = zeros(n,n);
for i = 1:n
    for j = 1:i
        gram_mat(i,j) = (x_train(:,i))' * x_train(:,j);
        H(i,j) = y_train(i) * y_train(j) * gram_mat(i,j);
        gram_mat(j,i) = gram_mat(i,j);
        H(j,i) = H(i,j);
    end
end
        
f = -ones(n,1);
Aeq = y_train';
beq = 0;
options = optimset('LargeScale','off','MaxIter',500);
alpha = quadprog(H, f, [], [], Aeq, beq, zeros(n, 1),ones(n,1)*10^6,[], options);

%%
obj_value =f'*alpha+0.5*alpha'*H*alpha;
w_o = zeros(57,1);
for i = 1:d
    w_o = w_o + alpha(i)*y_train(i)*x_train(:,i);
end

rounded_alpha = find(alpha < 10^(-4) );
alpha(rounded_alpha) = 0;
support_vector_indices = find(alpha > 1/10^4 );

b=0;
for i =1: size(support_vector_indices,1)
    b = b+ (y_train(i) - w_o'* x_train( :, i) );
end
b = b/size(support_vector_indices,1);

%% training accuracy
pred_y = zeros(2000,1);
for i =1:n
    pred_y(i) = (w_o'*x_train(:,i)+b);
end

for i = 1:n
    if pred_y(i) <0
        pred_y(i)=-1;
    elseif pred_y(i) >0
        pred_y(i)=1;
    end
end
tr_acc = sum(pred_y == y_train) / numel(pred_y);

%% test accuracy
pred_y_te = zeros(n_test, 1);
for i =1:n_test
    pred_y_te(i) = (w_o'*x_test(:,i)+b);
    if pred_y_te(i) <0
        pred_y_te(i)=-1;
    elseif pred_y_te(i) >=0
        pred_y_te(i)=1;
    end
end
te_acc = sum(pred_y_te == y_test) / numel(pred_y_te);
fprintf('###  linear kernel with hard margin ### \n training acc: %.4f, test acc: %.4f\n', tr_acc, te_acc);



