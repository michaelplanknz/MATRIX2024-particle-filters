clear
close all

% For reproducibility
rng(5074);

% Number of time steps and particles to use
nSteps = 50;
nParticles = 1e4;

% Parameter values
a = 1.4;
b = 0.002;
x0 = 10;

% Priors for target parameters
aMean = 2;
aSD = 2;
bMean = 0.002;
bSD = 0.002;

% Amount of noise to add to parameters each time step
paramNoise = 0.02;


% Simuate forward model to generate synthetic data
[xTrue, y] = simulate_SLM(a, b, x0, nSteps, 1);

% Run the state-augmented particle filter to jointly estimate hidden states
% and parameter values from the synthetic data
t = 0:nSteps-1;
[X, A, B] = runStateAugPF_SLM(y, aMean, aSD, bMean, bSD, paramNoise, x0, nParticles);


% Plot results
col = [0.7 0.7 0.7];

nPlot = 20;
h = figure(1);
h.Position = [46   564   686   341];
p1 = plot(t, X(1:nPlot, :), 'Color', col, 'LineStyle', '-');
hold on
p2 = plot(t, xTrue, 'b-');
p3 = plot(t, y, 'ro');
xlabel('t')
ylabel('x(t)')
h=gca;
legend( [p3, p2, p1(1)], 'measurements', 'true states', 'estimated states (marginalised over \theta)', 'Location', 'SouthEast' );



h = figure(2);
h.Position =  [560   617   956   331];
tiledlayout(1, 2)
nexttile;
plot(t, A(1:nPlot, :), 'Color', col, 'LineStyle', '-');
yline(a, 'b-');
xlabel('t')
ylabel('\alpha(t)')
nexttile;
plot(t, B(1:nPlot, :), 'Color', col, 'LineStyle', '-');
yline(b, 'b-');
xlabel('t')
ylabel('\beta(t)')

aEst = A(:, end);
bEst = B(:, end);

edges1 = linspace(min(aEst), max(aEst), 50);
edges2 = linspace(min(bEst), max(bEst), 50);
f = histcounts2(aEst, bEst, edges1, edges2);

mid1 = 0.5 * (edges1(1:end-1)+edges1(2:end));
mid2 = 0.5 * (edges2(1:end-1)+edges2(2:end));

figure(3);
title('joint posterior for parameters (\alpha,\beta)')
imagesc(mid1, mid2, f)
xline(a, 'w-');
yline(b, 'w-');
xlabel('a')
ylabel('b')

