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

C = [0.1, 0.6, 1.1, 2.1];   % soft margin
% polynomial kernel training with diff p and c
p=[1, 2 , 3 , 4 , 5];

n = size(x_train,2);
d = size(x_train,1);
n_te = size(x_test,2);
tr_acc_result = zeros(size(p,2),size(C,2));
te_acc_result = zeros(size(p,2),size(C,2));


for i =1:size(p,2)
    for c = 1:size(C,2)
        [H,gram_mat] = Hessian(x_train, y_train, n,p(i));
        [alpha,b_o] = train_model(x_train, y_train,n,H, zeros(n,1) , ones(n,1) * C(c), p(i) ) ;
        fprintf('\n optimal b_o = %.4f \n ',b_o);
        [tr_acc,pred_y] = train_acc(alpha,  y_train, x_train, n,  b_o, p(i));
        [te_acc,pred_y_te] = test_acc( alpha, y_test,y_train, x_test,x_train, n, n_te,  b_o , p(i) ); 
        tr_acc_result(i,c) = tr_acc;
        te_acc_result(i,c) = te_acc;
        fprintf('\n C = %.2f poly p=%.4f, training acc: %.4f , test accuracy : %.4f \n', C(c), p(i), tr_acc, te_acc);
    end
end

% find hessian matrix with kernel
function [H,gram_mat] = Hessian(x_train, y_train,n, p)
    gram_mat = zeros(n,n);
    H = zeros(n,n);
    for i = 1:n
        for j = 1:i
            gram_mat(i,j) = ((x_train(:,i))' * x_train(:,j) + 1)^p;
            H(i,j) = y_train(i) * y_train(j) * gram_mat(i,j) ;
            gram_mat(j,i) = gram_mat(i,j);
            H(j,i) = H(i,j);
        end
    end
end
% train model, solve quadratic problem, get optimal w and b
function [alpha,b_o] = train_model(x_train, y_train,n,H,low_bound,up_bound, p ) 
    f = -ones(n,1);
    Aeq = y_train';
    beq = 0;
    options = optimset('LargeScale','off','MaxIter',500);
    alpha = quadprog(H, f, [], [], Aeq, beq, low_bound,up_bound,[], options);
    
    rounded_alpha = find(alpha < 10^-4 );
    alpha(rounded_alpha) = 0; 
    
    support_vector_indices = find(alpha > 0 & alpha < up_bound(1) );
    fprintf('\n number of support vector:%d \n ',size( support_vector_indices,1 ));
    
    % find parameters b_o
    b=zeros(size(support_vector_indices,1),1);
    for i =1: size(support_vector_indices,1)
        second = 0;
        idx = support_vector_indices(i);
        for j =1:n
            second = second+ alpha(j)*y_train(j)*(x_train(:,idx)'*x_train(:,j)+1)^p;
        end
        b(i) = y_train(idx) - second;
    end
    b_o = mean(b);
    
end

% training accuracy
function [tr_acc,pred_y] = train_acc( alpha, y_train, x_train, n,  b_o , p )
    pred_y = zeros(n,1);
    for i =1:n
        first = 0;
        for j =1:n
            first = first+alpha(j)*y_train(j)*(x_train(:,i)' * x_train(:,j) + 1) ^ p ;
        end
        pred_y(i) = first+b_o;
        if pred_y(i) <=0
            pred_y(i)=-1;
        else
            pred_y(i)=1;
        end        
    end

    tr_acc = sum(pred_y == y_train) / numel(y_train);
end


% test accuracy
function [te_acc,pred_y_te] = test_acc( alpha, y_te,y_train, x_test,x_train, n, n_te,  b_o , p )
    pred_y_te = zeros(n_te,1);
    for i =1:n_te  % number of test set 
        first = 0;
        for j =1:n  % number of training set
            first = first+alpha(j)*y_train(j)*(x_train(:,j)' * x_test(:,i) + 1) ^ p ;
        end
        pred_y_te(i) = first+b_o;
        if pred_y_te(i) <=0
            pred_y_te(i)=-1;
        else
            pred_y_te(i)=1;
        end        
    end

    te_acc = sum(pred_y_te == y_te) / numel(y_te);
end

