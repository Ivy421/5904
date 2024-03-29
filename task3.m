% training data
tr_data = load('train.mat');
x_train = tr_data.train_data ; % 57*2000, 57 dimensions and 2000 samples
y_train = tr_data.train_label; % 2000*1

% eveluation data
data = load('eval.mat');
x = data.eval_data;
y = data.eval_predicted;


% std normalization
mean_vals = mean(x_train'); 
std_vals = std(x_train');  
x_train_trns = (x_train' - mean_vals) ./ std_vals;
x_train = x_train_trns';
x_eval_trns = (x' - mean_vals) ./ std_vals;
x = x_eval_trns';


n = size(x_train,2);
d = size(x_train,1);
n_te = size(x,2);

% polynomial kernel training with diff p and c
C =  0.6 ;  % soft margin
p = 2;
[H,gram_mat] = Hessian(x_train, y_train, n,p);
[alpha,b_o] = train_model(x_train, y_train,n,H, zeros(n,1) , ones(n,1) * C, p ) ;
[eval_acc, eval_pred]  = g( alpha, y  ,y_train, x, x_train, n, n_te, b_o , p );
fprintf('\n optimal b_o = %.4f \n evaluation accuracy:%.4f \n ',b_o,eval_acc);


% find hessian matrix with kernel
function [H,gram_mat] = Hessian(x_train, y_train,n, p)
    gram_mat = zeros(n,n);
    H = zeros(n,n);
    for i = 1:n
        for j = 1:i
            gram_mat(i,j) = ((x_train(:,i))' * x_train(:,j) +1 )^p;
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

% discriminant function

function [te_acc,pred_y_te] = g( alpha, y_te,y_train, x_test,x_train, n, n_te,  b_o , p )
    pred_y_te = zeros(n_te,1);
    for i =1:n_te  % number of test set 
        first = 0;
        for j =1:n  % number of training set
            first = first+alpha(j)*y_train(j)*(x_train(:,j)' * x_test(:,i) +1 ) ^ p ;
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
