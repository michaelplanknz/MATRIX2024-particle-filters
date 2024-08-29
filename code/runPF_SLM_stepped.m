function X = runPF_SLM_stepped(y, a, b, x0, nParticles)

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

greyCol = [0.7 0.7 0.7];

t = 0:nSteps-1;
h = figure(1);
particleID = (1:nParticles)';
for iStep = 1:nSteps-1
    % Simulate dynamic model to calculate hidden state at next time step
    X(:, iStep+1) = poissrnd( a*X(:, iStep) - b*X(:, iStep).^2 );

    % Calculate particle weights according to the likelihood function and
    % observed data for this time step
    weights = poisspdf( y(iStep+1), X(:, iStep+1) );

    % Plot hidden states at new time step
    plot(t(1:iStep+1), X(:, 1:iStep+1)', 'k-' )
    hold on
    plot(t(1:iStep+1), y(1:iStep+1), 'ro')
    xlabel('t')
    ylabel('x(t)')
    fprintf('Step 1: Propagate particles from t=%i to t=%i\n', t(iStep), t(iStep+1) ) 
    pause

    % Display particle data
    fprintf('Step 2: Calculate likelihood of particles at t=%i:\n', t(iStep+1))
    state = X(:, iStep+1);
    tbl = table(particleID, state, weights);
    disp(tbl);

    % Color particles according to their likelihood
    colArr = [0.9*(1-weights/max(weights)), 0.9*(1-weights/max(weights)), ones(nParticles, 1)];
    for iParticle = 1:nParticles
        plot(t(iStep:iStep+1), X(iParticle, iStep:iStep+1)', 'Color', colArr(iParticle, :), 'LineStyle', '-' )
    end
    plot(t(iStep+1), y(iStep+1), 'ro', 'MarkerFaceColor', 'r')

    pause


    % Resample particles
    resampleIndices = randsample(nParticles, nParticles, true, weights);

    % Plot resampled particles in black and discared particles in grey
    plot(t(1:iStep+1), X(~ismember(1:nParticles, resampleIndices), 1:iStep+1)', 'Color', greyCol, 'LineStyle', '-' )
    hold on
    plot(t(1:iStep+1), X(ismember(1:nParticles, resampleIndices), 1:iStep+1)', 'k-' )
    plot(t(1:iStep+1), y(1:iStep+1), 'ro')
    hold off
    xlabel('t')
    ylabel('x(t)')
    fprintf('Step 3: resample particles (%i,%i,%i,%i,%i,%i,%i,%i,%i,%i)\n\n', sort(resampleIndices) )
    pause

    X = X(resampleIndices, :);

end

