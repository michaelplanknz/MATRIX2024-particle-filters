clear
close all

% For reproducibility
rng(5074);

% Number of time steps and particles to use
nSteps = 40;
nParticles = 1e4;

% Parameter values
a = 1.4;
b = 0.002;
x0 = 10;

% Simuate forward model to generate synthetic data
[xTrue, y] = simulate_SLM(a, b, x0, nSteps, 1);

% Run particle filter model to estimate hidden states from synthetic data
t = 0:nSteps-1;
X = runPF_SLM(y, a, b, x0, nParticles);

% Plot results
col = [0.7 0.7 0.7];

nPlot = 20;
h = figure(1);
h.Position = [ 46   564   686   341];
p1 = plot(t, X(1:nPlot, :), 'Color', col, 'LineStyle', '-');
hold on
p2 = plot(t, xTrue, 'b-');
p3 = plot(t, y, 'ro');
xlabel('t')
ylabel('x(t)')
h=gca;
legend( [p3, p2, p1(1)], 'measurements', 'true states', 'estimated states', 'Location', 'SouthEast'  );

