clear
close all

% For reproducibility
rng(5074);

% Number of time steps and particles to use
nSteps = 40;
nParticles = 10;

% Parameter values
a = 1.4;
b = 0.002;
x0 = 10;

% Simuate forward model to generate synthetic data
[xTrue, y] = simulate_SLM(a, b, x0, nSteps, 1);

% Run particle filter model to estimate hidden states from synthetic data
t = 0:nSteps-1;
X = runPF_SLM_stepped(y, a, b, x0, nParticles);



