function Step3_GeneratingSimulatedData_2D(N1,N2) 
    options = odeset('RelTol', 1e-12, 'AbsTol', 1e-12);
    t_vec22=[22,24,25,27,30,40,50,60,70,80,90,100,120,140,160,180] ;
    tbefore22=[0, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 19, 22];
    t_vec0to180=[tbefore22,t_vec22(2:end)];

    %Put your Parameter Sample Set here!
    ParaName='***********.csv';

    %Put your Insulin Generated from GPR_generate_insulin_data.ipynb here!
    InsName='**********.mat';
    
    %Put your Glucose Generated from GPR_generate_insulin_data.ipynb here!
    GluName='**********.mat';
    
    datafolder=cd;
    SM=importdata(fullfile(datafolder,ParaName));
    Ins= importdata(fullfile(datafolder,InsName));
    Glu= importdata(fullfile(datafolder,GluName));

    InsUsed=Ins(N1:N2,:);
    GluUsed=Glu(N1:N2,:);
    samples_matrix=SM(N1:N2,:);

    n_samples=size(samples_matrix,1);

    Output_tGF=zeros(n_samples,4,28);

    for sample=1:size(samples_matrix,1)

        newins =InsUsed(sample,:);
        newglu =GluUsed(sample,:);

        glufcn= @(t) interp1( t_vec0to180,newglu, t, 'linear', 'extrap');
        insfcn= @(t) interp1( t_vec0to180,newins, t, 'linear', 'extrap');
 
        params=samples_matrix(sample,1:5);
    
        fb=params(5);
        ib=newins(1);
        fx0=[fb,0]';%%% all at time 22, no I2
        [~,xt]=ode45(@(t,x) rhs_FX2D_LIP(t, x, params,ib, glufcn, insfcn),t_vec0to180,fx0,options);  
        Output_tGF(sample,:,:)=[t_vec0to180;newglu;newins;xt(:,1)'];
    end

    ou_pos = zeros(size(Output_tGF));
    in_pos= zeros(size(samples_matrix));

    % Iterate through rows and keep only positive values
    for i = 1:size(Output_tGF,1)
        if all(Output_tGF(i,:,:) >= 0)
            ou_pos(i,:,:) = Output_tGF(i,:,:);
            in_pos(i,:)=samples_matrix(i,:);
        end
    end
    
    % Remove rows with all zeros
    ou_pos = ou_pos(any(any(ou_pos,3),2),:,:);
    in_pos = in_pos(any(in_pos,2),:);
 
    Output_tGF_pos=ou_pos;
    Input_para_pos=in_pos;

    filename1 =sprintf('Parameters_for_simulateddata_%dto%d.csv',N1,N2);
    writematrix(Input_para_pos,  filename1 );
 
    filename2 = sprintf('Simulated_tGIF_%dto%d.csv',N1,N2);
    writematrix(Output_tGF_pos, filename2 );

end
 
% function deriv = rhs_FX2D_LIP(t, x, params,ib,glu_, ins_ )
%     Sfal = params(1);
%     Pxa = params(2);
%     Sffb = params(3);
%     PXFCR = params(4);
%     fb = params(5);
%     Sf=Sffb/fb;
%     al=Sfal/Sf;
%     deriv =  [ -Sf * al * glu_(t) * x(1)  + (Sf-x(2)) * fb;...
%                 Pxa * (ins_(t) - ib)- PXFCR * x(2)];
% end

 