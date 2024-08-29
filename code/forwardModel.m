clear
close all

% For reproducibility
rng(5074);

% Number of time steps and number of model realisations to run
nSteps = 40;
nReps = 5;

% Parameter values
a = 1.4;
b = 0.002;
x0 = 10;

% Run forward model
t = 0:nSteps-1;
[X, Y] = simulate_SLM(a, b, x0, nSteps, nReps);

% Plot results
col = [0.7 0.7 0.7];

h = figure(1);
h.Position =  [560   617   956   331];
tiledlayout(1, 2);
nexttile;
plot(t, X, 'Color', col, 'LineStyle', '-')
xlabel('t')
ylabel('x(t)')
title('hidden state')

nexttile;
plot(t, Y, 'Color', col, 'LineStyle', '-')
xlabel('t')
ylabel('y(t)')
title('observations')
