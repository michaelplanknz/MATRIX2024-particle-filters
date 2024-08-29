function [X, Y] = simulate_SLM(a, b, x0, nSteps, nReps)

% Simulate nReps of the stochastic logistic model for nSteps with parameters a and b, initial
% condition Poisson(x0)
% Output X is a nReps x nSteps matrix containing the hidden states
% Output Y is a nReps x nSteps matrix containing the corresponding observed data

X = zeros(nReps, nSteps);

X(:, 1) = poissrnd(x0, nReps, 1);

for iStep = 1:nSteps-1
    X(:, iStep+1) = poissrnd( a*X(:, iStep) - b*X(:, iStep).^2 );
end

Y = poissrnd(X);
