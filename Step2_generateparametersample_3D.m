
num_samples=5000000;%%% How many sample parameter set you need.


gludata = importdata('25FSIGT_Glu.csv');
FFAdata = importdata('25FSIGT_FFA.csv');    
GIF0 = importdata('25FSIGT_GIF0.csv');

all6para=importdata('optimizedparameters_3Dmodel.csv');
alloptmpara=[all6para,GIF0,gludata(:,1),FFAdata(:,1)];

usedsubj=[1:2,4:10,12:17,19:24];%1:25;%
usedpara=[1:10];
optmpara=alloptmpara(:,[1:7,9:11]);
optmpara_used=optmpara(usedsubj,usedpara);

mean_log = mean(log(optmpara_used), 1); 
cov_log = cov(log(optmpara_used));      
generated_samples_lognormal = mvnrnd(mean_log, cov_log, num_samples);
para_mvlognormal = exp(generated_samples_lognormal) ;
 
%%% Save the para_mvlognormal as your sample parameter set.
%%% writematrix(para_mvlognormal,'****PUT THE NAME YOU WANT.csv****');
