function prob = univ4
clear ; close all; clc
data = load('univ4.txt');
X = data(:, [1 :5]); y = data(:, 6);
[m, n] = size(X);
X = [ones(m, 1) X];
initial_theta = zeros(n + 1, 1);
[cost, grad] = costFunction(initial_theta, X, y);
test_theta = [-24; 0.2; 0.2;0;0;0];
[cost, grad] = costFunction(test_theta, X, y);

options = optimset('GradObj', 'on', 'MaxIter', 400);

[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

prob = sigmoid([1 320 9.3 4 4 20000] * theta);
%fprintf(['For a student with these scores we predict an admission ' ...
%         'probability of %f\n'], prob);

fprintf('\n');


