
Readme:
  1. If readers just want to reproduce the figures in the paper, then they may just use Matlab to 
    open the file

    plot_every_figure.m

    and run the following command in Matlab:

    plot_every_figure(fignumber)
    
    where fignumber is from 1 to 11 for figure1 to figure11, and from 31 to 37 for supplementary figures.
    
  2. Since trained network (around 220MB) and simulated datasets (several GBs) are too big, we do not upload them here.
     One can contact Xiaoyu Duan at duanx3@nih.gov if they want to get one for checking.
     
  4. Anyway, here we provide the code from scratch, so that as long as one follow the steps, they can generate simulated datasets and
     trained networks on their own(of course it is recommended to train the networks with GPUs).

All steps from the beginning optimization to the final parameter inference is as follows:
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
          model, go to: Step5_2D_parameter_inference.ipynb
  
          10.for parameter inference from the trained network for the 3D
          model, go to: Step5_3D_parameter_inference.ipynb
