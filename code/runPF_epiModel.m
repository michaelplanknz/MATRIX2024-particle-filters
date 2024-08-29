function [Rt, It] = runPF_epiModel(y, sigmaR, gs, I0, R0_mean, R0_SD, nParticles)

% Run bootstrap particle filter with nParticles to estimate hidden states (Rt and It) of the
% epidemic renewal equation model based on observed time series data (y) for daily incidence of new cases
% Model parameters sigmaR is the reproduction number random walk step SD
%                  gs is a vector containing the probability mass function for the generation time distribution
% The initial number of daily infections is 1+Poisson(I0) 
% The value of Rt is drawn from a gamma distribution with mean R0_mean and SD R0_SD 
%
% Outputs Rt and It are nParticles x nSteps matrices containing samples from the  hidden state distribution conditioned on the data

% Number of time steps (length of data time series)
nSteps = length(y);

% Set up arrays to store hidden state values
Rt = zeros(nParticles, nSteps);
It = zeros(nParticles, nSteps);

% Draw initial Rt values from specified gamma distribution
[sh, sc] = gamShapeScale(R0_mean, R0_SD);
Rt(:, 1) = gamrnd(sh, sc, nParticles, 1);

% Draw initial It values 
It(:, 1) = 1+poissrnd(I0, nParticles, 1);

for iStep = 1:nSteps-1
    % Simulate dynamic model to calculate hidden states Rt and It at next time step
    % Gaussian random walk (truncated to non-negative values) for Rt
    Rt(:, iStep+1) = max(0, Rt(:, iStep) + normrnd(0, sigmaR, nParticles, 1) );
    % Poisson renewal equation for It
    ind = iStep:-1:max(1, iStep-length(gs)+1);
    It(:, iStep+1) = poissrnd(Rt(:, iStep+1) .* sum(It(:, ind).*gs(1:length(ind)), 2) );

    % Calculate particle weights according to the likelihood function and
    % observed data for this time step
    weights = poisspdf(y(iStep+1), It(:, iStep+1)) ;


    % Resample particles    
    resampleIndices = randsample(nParticles, nParticles, true, weights);
    Rt = Rt(resampleIndices, :);
    It = It(resampleIndices, :);
end

