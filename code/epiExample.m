clear 
close all

% For reproducibility
rng(5074);

% Folder and filename for input data
dataFolder = "../data/";
fNameData = "NZ_covid_data.csv";

% Number of particles to use
nParticles = 1e5;

% Model parameter values
sigmaR = 0.2;
I0 = 1;
R0_mean = 2;
R0_SD = 1;

% Generation time distribution
gsMean = 5.05;
gsSD = 1.94;
[sh, sc] = gamShapeScale(gsMean, gsSD);
pdfFn = @(x)(gampdf(x, sh, sc ));
GTmax = 15;

% Make a dicerete version of the GT distribution
gs = discDist( pdfFn, 1, GTmax );

% Read data
data = readtable(dataFolder+fNameData);
y = data.notified_cases;

% Run particle filter
t = 0:length(y)-1;
[Rt, It] = runPF_epiModel(y, sigmaR, gs, I0, R0_mean, R0_SD, nParticles);


% Plot results
greyCol = [0.7 0.7 0.7];
blueCol = [0.7 0.8 1];
q0 = 0.05;
q1 = 0.95;


nPlot = 20;
h = figure(1);
h.Position = [   680   187   724   811];
tiledlayout(2, 1);
nexttile;
p1 = fill([t, fliplr(t)], [quantile(Rt, q0), fliplr(quantile(Rt, q1))], blueCol  );
hold on
p2 = plot(t, Rt(1:nPlot, :), 'Color', greyCol, 'LineStyle', '-');
p3 = plot(t, median(Rt), 'b-');
xlabel('t')
ylabel('R(t)')
legend( [p3, p1, p2(1)], 'median', '90% CI', 'sample trajectories', 'Location', 'NorthEast'  );
nexttile;
p1 = plot(t, It(1:nPlot, :), 'Color', greyCol, 'LineStyle', '-');
hold on
p2 = plot(t, y, 'ro');
xlabel('t')
ylabel('I(t)')
legend('hidden state estimates')
legend( [p1(1), p2], 'sample trajectories', 'data', 'Location', 'NorthEast'  );

