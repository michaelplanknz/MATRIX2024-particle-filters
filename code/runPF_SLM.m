function X = runPF_SLM(y, a, b, x0, nParticles)

% Run bootstrap particle filter with nParticles to estimate hidden states (X) of the
% stochastic logistic model with parameters a and b and initial condition
% Poisson(x0), based on observed time series data (y).
% Output X is a nParticles x nSteps matrix containing samples from the
% hidden state distribution conditioned on the data

% Number of time steps (length of data time series)
nSteps = length(y);

% Set up an array to store hidden state values
X = zeros(nParticles, nSteps);

% Initialise hidden state
X(:, 1) = poissrnd(x0, nParticles, 1);

for iStep = 1:nSteps-1
    % Simulate dynamic model to calculate hidden state at next time step
    X(:, iStep+1) = poissrnd( a*X(:, iStep) - b*X(:, iStep).^2 );

    % Calculate particle weights according to the likelihood function and
    % observed data for this time step
    weights = poisspdf( y(iStep+1), X(:, iStep+1) );

    % Resample particles
    resampleIndices = randsample(nParticles, nParticles, true, weights);
    X = X(resampleIndices, :);
end

