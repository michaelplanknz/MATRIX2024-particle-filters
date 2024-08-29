function [X, A, B] = runStateAugPF_SLM(y, aMean, aSD, bMean, bSD, paramNoise, x0, nParticles)

% Run state-augmented bootstrap particle filter with nParticles to estimate hidden states (X) and parameters (a and b) of the
% stochastic logistic model with initial condition Poisson(x0), based on observed time series data (y).
% aMean, aSD, bMean and bSD specifiy the means and standard deviations of
% the prior distributions for a and b
% paramNoise specifies the noise in the change per time step in parameter
% values
% Output X is a nParticles x nSteps matrix containing samples from the
% hidden state distribution conditioned on the data
%        A and B are Particles x nSteps matrices containing corresponding
%        samples from the parameter distribution

% Number of time steps (length of data time series)
nSteps = length(y);


% Set up arrays to store hidden state and parameter values
X = zeros(nParticles, nSteps);
A = zeros(nParticles, nSteps);
B = zeros(nParticles, nSteps);

% Initialise hidden state
X(:, 1) = poissrnd(x0, nParticles, 1);

% Initialise parameter values a and b by drawing from the respective priors
[sh, sc] = gamShapeScale(aMean, aSD);
A(:, 1) = gamrnd(sh, sc, nParticles, 1 );

[sh, sc] = gamShapeScale(bMean, bSD);
B(:, 1) = gamrnd(sh, sc, nParticles, 1 );

for iStep = 1:nSteps-1
    % Simulate dynamic model to calculate hidden state at next time step
    % and add noise to the parameter values
    X(:, iStep+1) = poissrnd( max(0, A(:, iStep).*X(:, iStep) - B(:, iStep).*X(:, iStep).^2 ));
    A(:, iStep+1) = A(:, iStep) .* (1 + normrnd(0, paramNoise, nParticles, 1) );
    B(:, iStep+1) = B(:, iStep) .* (1 + normrnd(0, paramNoise, nParticles, 1) );


   % Calculate particle weights according to the likelihood function and
    % observed data for this time step
    weights = poisspdf( y(iStep+1), X(:, iStep+1) );

    % Resample particles
    resampleIndices = randsample(nParticles, nParticles, true, weights);
    X = X(resampleIndices, :);
    A = A(resampleIndices, :); 
    B = B(resampleIndices, :); 
end


