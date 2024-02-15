% define the train sample
x_train = linspace(-2, 2, 80)';
y_train = 1.2 * sin(pi * x_train) - 2.4 * cos(pi * x_train);

% Define the test samples
x_test = linspace(-3, 3, 600)';   -3, 3, 600
y_desired = 1.2 * sin(pi * x_test) - 2.4 * cos(pi * x_test);

% Loop over different numbers of neurons in the hidden layer
hidden_neurons = [1, 2, 5, 10, 20, 50, 100];

%%%% MLP BP method %%%%

% for i = 1:length(hidden_neurons)
%     % Create and train the MLP model
%     hidden_size = hidden_neurons(i);
%     net = feedforwardnet(hidden_size, 'trainlm');
%     net.trainParam.epochs = 100;
%     net = train(net, x_train', y_train');
% 
%     % Test the trained model
%     y_pred = net(x_test');
% 
%     % Plot the results
%     figure;
%     plot(x_test, y_desired, 'b-', 'LineWidth', 2);
%     hold on;
%     plot(x_test, y_pred, 'r--', 'LineWidth', 2);
%     title(['MLP Output vs. Desired Output (Hidden Neurons: ' num2str(hidden_size) ')']);
%     xlabel('x');
%     ylabel('y');
%     legend('Desired Output', 'MLP Output');
%     grid on;
%     hold off;
% end

%%%% batch mode with trainlm algorithm %%%%
for i = 1:length(hidden_neurons)
    % Create and configure the MLP model
    hidden_size = hidden_neurons(i);
    net = feedforwardnet(hidden_size, 'trainbr');  % trainlm
    net.trainParam.epochs = 100;
    net.trainParam.batchSize = length(x_train); % Set batch mode
    net = train(net, x_train', y_train');

    % Test the trained model
    y_pred = net(x_test');

    % Plot the results
    figure;
    plot(x_test, y_desired, 'b-', 'LineWidth', 2);
    hold on;
    plot(x_test, y_pred, 'r--', 'LineWidth', 2);
    title(['MLP Output vs. Desired Output (Hidden Neurons: ' num2str(hidden_size) ')']);
    xlabel('x');
    ylabel('y');
    legend('Desired Output', 'MLP Output');
    grid on;
    hold off;
end
