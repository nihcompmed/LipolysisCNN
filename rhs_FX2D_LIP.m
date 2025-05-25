function deriv = rhs_FX2D_LIP(t, x, params,ib,glu_, ins_ )
    Sfal = params(1);
    Pxa = params(2);
    Sffb = params(3);
    PXFCR = params(4);
    fb = params(5);
    Sf=Sffb/fb;
    al=Sfal/Sf;
    deriv =  [ -Sf * al * glu_(t) * x(1)  + (Sf-x(2)) * fb;...
                Pxa * (ins_(t) - ib)- PXFCR * x(2)];
end
