%%% fig
fig2: 
'GFX3DoptmRelGluFFAsimult_25FSIGT_fromdenoiseFFA_01_to_25.csv'
inused=importdata('prepare_fig2_parameterset3DdecF_mvlgnmDirect_first10kdata_of3930694.csv'); 
Para2= importdata('prepare_fig3_optmpara_2Dmodel.csv');

in=importdata('prepare_fig3_parameterset2D_mvlgnmDirectparaphybd_first10kdata_of2877269.csv');

fig4
simudata=importdata('prepare_fig4_3DdecF_mvlgnmDirect_first10kdata_of3930694.csv'); 
optmG=importdata('GFX3D25_optmG.csv');optmF=importdata('GFX3D25_optmF.csv');
fig5
    optmF2D=importdata('FX2D25LIP_optmF_241214.csv');
    simudata=importdata('Out250130FX2D25_mvlognmdirectparaphybd_2877269_first10k.csv');

    fig6b
        paratrain=importdata('inferencefromnetwork/paratrain_24121602.csv');
    parainfertrain=importdata('inferencefromnetwork/parainfertrain_24121602.csv');

fig7b

    paratest=importdata('inferencefromnetwork/paratest_24121602.csv');
    parainfertest=importdata('inferencefromnetwork/parainfertest_24121602.csv');
    paraoptm=importdata('prepare_fig3_optmpara_2Dmodel.csv');
    paratrue=paraoptm(:,1:5);
    parainferoptm=importdata('inferencefromnetwork/parainferfromoptm_Mvlognormal25_24121602.csv');



    RUN:python FX2DNN_twithreciGIF_241216_MvLognormal_noinferGIFb_tanh01.py 500000 1e-3 500 241216 1000 500 24121602


    fig7
Need to generate

inpcsv = pd.read_csv('In250115_GFX3D25_decF_mvlgnmDirect_3930694.csv', header=None)
outcsv = pd.read_csv('Out250115_GFX3D25_decF_mvlgnmDirect_3930694.csv', header=None)



parainfertrain = importdata('inferencefromnetwork/paragiventrain_Mvlognormaldirect_25011504.csv');
paratrain      = importdata('inferencefromnetwork/parainfertrain_Mvlognormaldirect_25011504.csv');

parainfertest  = importdata('inferencefromnetwork/paragiventest_Mvlognormaldirect_25011504.csv');
paratest       = importdata('inferencefromnetwork/parainfertest_Mvlognormaldirect_25011504.csv');

parainferoptm = importdata('inferencefromnetwork/parainferfromoptm_Mvlognormaldirect_25011504.csv');





    figS5
    aa=importdata('prepare_fig2_parameterset3DdecF_mvlgnmDirect_first10kdata_of3930694.csv');






FigS14
denoiseFFA=importdata('FCdenoised_FFA.csv');



%%

%{
Readme:
Step 1:
        1.for parameter estimation from optimization for the 2D model, go to:
        2.for parameter estimation from optimization for the 3D model, go to:
Step 2:
        3.for generating sample parameter set for the 2D model, go to:
        4.for generating sample parameter set for the 3D model, go to:
Step 3:
        5.for getting simulated datasets for the 2D model, go to:
        6.for getting simulated datasets for the 3D model, go to:
Step 4:
        7.for training the neural network(s) for the 2D model, go to:
        8.for training the neural network(s) for the 3D model, go to:
Step 5:
        9.for parameter inference from the trained network for the 2D
        model, go to:Step5_2D_parameter_inference.ipynb

        10.for parameter inference from the trained network for the 3D
        model, go to:Step5_3D_parameter_inference.ipynb
%}