function deriv = rhs_GFX_noI2(t, x, params, ins_ )
    alpha = 2;
 


    si = params(1);
    cx = params(2);
    sg = params(3);
    x2 = params(4);
    cf = params(5);
    l2 = params(6);
    xt0 = params(7);
    gb = params(8);
    ib  = params(9);
    fb = params(10);
 
 
    deriv = [ (sg.*(gb -x(1))-si.*x(3).*x(1));...
          (-cf.*(x(2)-fb)-l2.*((x(3)./x2).^alpha)./(1.+(x(3)./x2).^alpha));... % l_ is the old (l0 + l2)./cf
          (cx.*(max(ins_(t)-ib,0.0)-x(3)))]; % ins_(t0)


   % ins_ = @(t) interp1(new_vector(1,:), new_vector(2,:), t, 'linear', 'extrap');
% % %     deriv = [ (sg.*(gb.*(1+(max(xt_-ins_(t)+ib,0.0)./i2))-x(1))-si.*xt_.*x(1));
% % %               1000..*(-cf.*(ffa_-l_)-l2.*((xt_./x2).^alpha)./(1.+(xt_./x2).^alpha)); % l_ is the old (l0 + l2)./cf
% % %               (cx.*(max(ins_(t)-ib,0.0)-xt_)); % ins_(t0)
% % %               (-ce.*(ffa_-fb+kappa.*(x(1)-gb)))];
end