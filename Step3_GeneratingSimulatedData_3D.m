
N1=1;
N2=100;%%% Starting and ending row number of how many samples you want to generate simulated data with;

  
options = odeset('RelTol', 1e-12, 'AbsTol', 1e-12);
tindex=13:28;
t_vec22=[22,24,25,27,30,40,50,60,70,80,90,100,120,140,160,180] ;
tbefore22=[0, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 19, 22];
t_vec0to180=[tbefore22,t_vec22(2:end)];

ParaName='******Put your generated parameter sample set here.csv*****';
datafolder=cd;
SM=importdata(fullfile(datafolder,ParaName));
insfolder=cd;
InsName='*****Put your insulin data generated from GPR here.mat*****';
Ins= importdata(fullfile(insfolder,InsName));

samples_matrix=SM(N1:N2,1: 10) ;
InsUsed=Ins(N1:N2,:);
n_samples=size(samples_matrix,1);

Output_tGF=zeros(n_samples,4,16);
mat_insbefore22=zeros(n_samples,15);

for sample=1:size(samples_matrix,1)
    newinsall =InsUsed(sample,:);
    insfcn= @(t) interp1( t_vec0to180,newinsall, t, 'linear', 'extrap');

    ib=newinsall(1);

    cx=samples_matrix(sample,2);
    rhs_x=@(t,x) cx*(max(insfcn(t)-ib,0)-x);   
    [~,xfwd]=ode45(@(t,x) rhs_x(t, x),[0 22],0,options);
    xt22=xfwd(end);
    gfx22=[samples_matrix(sample,[9,10]),xt22]';
    params=[samples_matrix(sample,1:6),0,samples_matrix(sample,7),ib, samples_matrix(sample,8)];


    newins22 =InsUsed(sample,tindex);
    insfcn= @(t) interp1( t_vec22,newins22, t, 'linear', 'extrap');
    [~,xt]=ode45(@(t,x) rhs_GFX_noI2(t, x, params, insfcn),t_vec22,gfx22,options);
    Output_tGF(sample,:,:)=[t_vec22;xt(:,1)';newins22;xt(:,2)'];
    mat_insbefore22(sample,:)=InsUsed(sample,1 :15);
 
end
    
% Initialize new matrix
ou_pos = zeros(size(Output_tGF));
in_pos= zeros(size(samples_matrix));
insbefore22_pos=zeros(size(mat_insbefore22));

% Iterate through rows and keep only positive values
for i = 1:size(Output_tGF,1)
    if all(Output_tGF(i,:,:) >= 0)
        ou_pos(i,:,:) = Output_tGF(i,:,:);
        in_pos(i,:)=samples_matrix(i,:);
        insbefore22_pos(i,:)=mat_insbefore22(i,:) ;
    end
end

% Remove rows with all zeros
ou_pos = ou_pos(any(any(ou_pos,3),2),:,:);
in_pos = in_pos(any(in_pos,2),:);
insbefore22_pos = insbefore22_pos(any(insbefore22_pos,2),:);

Output_tGF_pos=ou_pos;
Input_para_pos=in_pos;

%%% Save the simulated data
% % % writematrix(Input_para_pos, 'Name of simulated data for the OUTPUT(parameters) of Neural Network.csv');
% % % writematrix(Output_tGF_pos, 'Name of simulated data for the InPUT(GIF) of Neural Network.csv');
% % % writematrix(insbefore22_pos, 'Name of insulin before t=22 needed for reconstruction since cx needs them.csv');


 
 