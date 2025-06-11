

optmpara=importdata('prepare_fig3_optmpara_2Dmodel.csv');
 
usedsubj=[1:2,4:10,12:17,19:24];%1:25;%
  
optmpara_used=optmpara(usedsubj,:);

 
mean_log = mean(log(optmpara_used), 1); % Mean vector of log-transformed data
cov_log = cov(log(optmpara_used));      % Covariance matrix of log-transformed data
  
num_samples_per_batch = 5000; % Number of samples to generate per batch
desired_num_samples = 50000; % Target number of valid samples
valid_samples = []; % Initialize an array to store valid samples
rng(250130);
% % Get the column-wise minimum and maximum of optmpara_used
min_vals = min(optmpara_used, [], 1);
max_vals = max(optmpara_used, [], 1);

% Adjust bounds to exclude outliers
max_excludeoutlier = max_vals;
min_excludeoutlier = min_vals;

% Loop until we have the desired number of valid samples
while size(valid_samples, 1) < desired_num_samples
    % Generate samples in log-normal space
    generated_samples_lognormal = mvnrnd(mean_log, cov_log, num_samples_per_batch);
    para_mvlognormal = exp(generated_samples_lognormal);
    
    % Filter samples within bounds
    within_bounds = all(para_mvlognormal >= min_vals & ...
                        para_mvlognormal >= min_excludeoutlier & ...
                        para_mvlognormal <= max_vals & ...
                        para_mvlognormal <= max_excludeoutlier, 2);
    
    % Append valid samples to the array
    valid_samples = [valid_samples; para_mvlognormal(within_bounds, :)];
    disp(['Generated ', num2str(size(valid_samples, 1)), ' valid samples so far...']);
end

% Keep only the desired number of samples
valid_samples = valid_samples(1:desired_num_samples, :);

% Display the final number of valid samples
disp(['Generated ', num2str(size(valid_samples, 1)), ' valid samples.']);

% %%% Save the valid_samples as your sample parameter set.
%%% writematrix(valid_samples,'****PUT THE NAME YOU WANT.csv****');
 
