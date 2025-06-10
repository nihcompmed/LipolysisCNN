function intgr8_cxcomputext22(N1,N2)%,part)
 
    Ins= importdata('5millionInsGPRsamples25Alltime_Seed241127.mat');

 
    samples_matrix=SM(N1-500000:N2-500000,:);
    InsUsed=Ins(2000000+N1:2000000+N2,:);
 

     t_vec0to180=ste_mat(1:28,2)';
    t_vec22=[22,24,25,27,30,40,50,60,70,80,90,100,120,140,160,180];


    options = odeset('RelTol', 1e-12, 'AbsTol', 1e-12);
    
    n_samples=size(samples_matrix,1);
%     vec_bwdxt0=zeros(n_samples,1);


% %     set_ins_normal=[1,14,15,16];
% %     set_ins_lognorm=[2:13];

    
% %     ntimes22=length(t_vec22);
% %     inspara=zeros(ntimes22,2);
% %     
    
    tic;
    
% % %     for i=1:ntimes22
% % %        % subplot(4,4,i);hold on;
% % %         ffa_I22=ste_mat(ste_mat(:,2)==t_vec22(i),4);
% % %         %disp(ffa_I22);
% % %         %hist(ffa_I22,20);
% % %         if ismember(i,set_ins_normal)==1
% % %             dist = fitdist(ffa_I22, 'Normal');
% % %         else
% % %             dist = fitdist(ffa_I22, 'Lognormal');
% % %         end
% % %         inspara(i,1)=dist.mu;
% % %         inspara(i,2)=dist.sigma;
% % %     end
% % %     
    Output_tGF=zeros(n_samples,4,16);
    
    %%figure;
    for sample=1:size(samples_matrix,1)
        newinsall =InsUsed(sample,:);
        insfcn= @(t) interp1( t_vec0to180,newinsall, t, 'linear', 'extrap');
        ib=samples_matrix(sample,9);
        cx=samples_matrix(sample,2);
        rhs_x=@(t,x) cx*(max(insfcn(t)-ib,0)-x);   
%         indices = find(t_vec22 <= txmax);
%         largest_index = max(indices);
 
    % [tout,xt]=ode113(@(t,x) rhs_x(t, x),[txmax fliplr(t_vec22(1:largest_index))],x_txmax,options);
    % [tout,xt]=ode45(@(t,x) rhs_x(t, x),linspace(txmax,22,2000),x_txmax ,options);
    % -t_vec22(largest_index)+t_vec22(1:largest_index)
        [~,xfwd]=ode45(@(t,x) rhs_x(t, x),[0 22],0,options);
        xt22=xfwd(end);
        samples_matrix(sample,7)=xt22;
        gfx0=[samples_matrix(sample,[11,12]),xt22]';%%% all at time 22, no I2
%           gfx0=samples_matrix(sample,[12,13,7])';%%% all at time 22
     %   gfx0=[samples_matrix(sample,[9,11]),0]';%%% all at time 0
        params=samples_matrix(sample,1:10);

        newins22 =InsUsed(sample,13:28);
        insfcn= @(t) interp1( t_vec22,newins22, t, 'linear', 'extrap');
        [~,xt]=ode45(@(t,x) rhs_GFX_noI2(t, x, params, insfcn),t_vec22,gfx0,options);
     
        
        Output_tGF(sample,:,:)=[t_vec22;xt(:,1)';newins22;xt(:,2)'];
    % % %  
    % %    for l=1:4 
    % %    subplot(2,2,l);hold on;
    % %      plot(t_vec22,xt(:,l),'b-');hold on
    % %  
    % %    end
     
    end
    
%     samples_matrix=[samples_matrix,vec_bwdxt0];
    % Initialize new matrix
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
%     vec_bwdxt0=vec_bwdxt0(any(in_pos,2),:);


    Output_tGF_pos=ou_pos;
    Input_para_pos=in_pos;
    
% % % figure;
% % %  
% % %         subplot(121);hold on;
% % %         for k=1:size(Output_tGF_pos,1)
% % %             if mod(k,20)==0
% % %                 gifmat=reshape(Output_tGF_pos(k,:,:),[4 16]);
% % %                 plot(t_vec22,gifmat(2,:),'b-');hold on
% % %             end
% % %         end
% % %  
% % %          subplot(122);hold on;
% % %         for k=1:size(Output_tGF_pos,1)
% % %             if mod(k,20)==0
% % %                 gifmat=reshape(Output_tGF_pos(k,:,:),[4 16]);
% % %                 plot(t_vec22,gifmat(4,:),'b-');hold on
% % %             end
% % %         end

    savefolder  =fullfile(Folder, '../../../../data/duanx3/GFX3DandNN/GFX3Ddata240328');
    filename1 =sprintf('In240425_GPRIns_CfFb_%dto%d.csv',N1,N2);
    writematrix(Input_para_pos, fullfile(savefolder,filename1));
    filename2 = sprintf('Out240425_GPRIns_CfFb_%dto%d.csv',N1,N2);
    writematrix(Output_tGF_pos, fullfile(savefolder,filename2));
%     filename3 = sprintf('bwdxt0230711_GPRIns_txmax_%dto%d.csv',N1,N2);
%     writematrix(vec_bwdxt0, fullfile(datafolder,filename3));

    toc;
    %%% si, cx, sg, x2, cf, l2, ce, kappa, xt0, l0, i2 , glu, ffa
end
 

function deriv = rhs_GFX_CfFb(t, x, params, ins_ )
    alpha = 2;
    si = params(1);
    cx = params(2);
    sg = params(3);
    x2 = params(4);
    cf = params(5);
    l2 = params(6);
    gb = params(8);
    ib  = params(9);
    CfFb= params(10);
    deriv = [ (sg.*(gb  -x(1))-si.*x(3).*x(1));...
           (-cf.* x(2)+CfFb -l2.*((x(3)./x2).^alpha)./(1.+(x(3)./x2).^alpha));... % l_ is the old (l0 + l2)./cf
          (cx.*(max(ins_(t)-ib,0.0)-x(3)))]; % ins_(t0)


   % ins_ = @(t) interp1(new_vector(1,:), new_vector(2,:), t, 'linear', 'extrap');
% % %     deriv = [ (sg.*(gb.*(1+(max(xt_-ins_(t)+ib,0.0)./i2))-x(1))-si.*xt_.*x(1));
% % %               1000..*(-cf.*(ffa_-l_)-l2.*((xt_./x2).^alpha)./(1.+(xt_./x2).^alpha)); % l_ is the old (l0 + l2)./cf
% % %               (cx.*(max(ins_(t)-ib,0.0)-xt_)); % ins_(t0)
% % %               (-ce.*(ffa_-fb+kappa.*(x(1)-gb)))];
end

 
function deriv = rhs_GFX_noI2(t, x, params, ins_ )
    alpha = 2;
    si = params(1);
    cx = params(2);
    sg = params(3);
    x2 = params(4);
    cf = params(5);
    l2 = params(6);
    gb = params(8);
    ib  = params(9);
    fb = params(10);
    deriv = [ (sg.*(gb  -x(1))-si.*x(3).*x(1));...
          (-cf.*(x(2)-fb)-l2.*((x(3)./x2).^alpha)./(1.+(x(3)./x2).^alpha));... % l_ is the old (l0 + l2)./cf
          (cx.*(max(ins_(t)-ib,0.0)-x(3)))]; % ins_(t0)


   % ins_ = @(t) interp1(new_vector(1,:), new_vector(2,:), t, 'linear', 'extrap');
% % %     deriv = [ (sg.*(gb.*(1+(max(xt_-ins_(t)+ib,0.0)./i2))-x(1))-si.*xt_.*x(1));
% % %               1000..*(-cf.*(ffa_-l_)-l2.*((xt_./x2).^alpha)./(1.+(xt_./x2).^alpha)); % l_ is the old (l0 + l2)./cf
% % %               (cx.*(max(ins_(t)-ib,0.0)-xt_)); % ins_(t0)
% % %               (-ce.*(ffa_-fb+kappa.*(x(1)-gb)))];
end



