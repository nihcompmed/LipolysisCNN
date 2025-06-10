
Readme:
  1. If readers just want to reproduce the figures in the paper, then they may just use Matlab to 
    open the file

    plot_every_figure.m

    and run the following command in Matlab:

    plot_every_figure(fignumber)
    
    where fignumber is from 1 to 11 for figure1 to figure11, and from 31 to 37 for supplementary figures.
    
  2. Since trained network (around 220MB) and simulated datasets (several GBs) are too big, we do not upload them here.
     One can contact Xiaoyu Duan at duanx3@nih.gov if they want to get one for checking. But we upload the first 10,000
     rows of those datasets in case readers would like to use and check. They are called:

     dataset2D_networkinput_tGIF_first10k.csv
     
     dataset2D_networkoutput_parameters_first10k.csv
     
     
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
  
          5.for getting simulated datasets for the 2D model:
          
            First, open GPR_generate_insulin_data.ipynb and run it to generate an insulin sample matrix.
            
          6.for getting simulated datasets for the 3D model:

            First, open GPR_generate_insulin_data.ipynb and run it to generate an insulin sample matrix.
          
  Step 4:
  
          7 and 8.for training the neural network(s) for the 2D or 3D model, go to the folder 
          
          /Step4_TrainingNetworks
          
          and there are several swarm files for any network needed in this paper. 
          For example, one can run

          python GFX3DNN_twithreciGIF_250203.py 1000 1e-3 10 500 6 241216 2000 1000 relu 250207033

          to train a network for 3D with feature engineering case (twithreciGIF). 
          The other sys inputs can be seen in the file GFX3DNN_twithreciGIF_250203.py in the same folder.

          Notice: These py files are all using the full 2D and 3D datasets, so one need to replace
          the name of .csv dataset by their own dataset to be trained with.
          For example, in the GFX3DNN_twithreciGIF_250203.py, look at the commands:

          inpcsv = pd.read_csv('In250115_GFX3D25_decF_mvlgnmDirect_3930694.csv', header=None)
          inputs = torch.tensor(inpcsv.values[:, :npara ], dtype=torch.float32)
          #print(inputs[0])
          outcsv = pd.read_csv('Out250115_GFX3D25_decF_mvlgnmDirect_3930694.csv', header=None)
          Ou = torch.tensor(outcsv.values, dtype=torch.float32)
          outputs = Ou.reshape(Ou.shape[0], 16, 4).transpose(1, 2)

          Replace it to be:
          
          inpcsv = pd.read_csv('*** PUT YOUR OUTPUT PARAMETER SET FOR TRAINING ****.csv', header=None)
          inputs = torch.tensor(inpcsv.values[:, :npara ], dtype=torch.float32)
          #print(inputs[0])
          outcsv = pd.read_csv('*** PUT YOUR INPUT SIMULATED TRAJETORY SET FOR TRAINING ****.csv', header=None)
          Ou = torch.tensor(outcsv.values, dtype=torch.float32)
          outputs = Ou.reshape(Ou.shape[0], 16, 4).transpose(1, 2)
          
  Step 5:
  
          9.for parameter inference from the trained network for the 2D
          model, go to: Step5_2D_parameter_inference.ipynb
  
          10.for parameter inference from the trained network for the 3D
          model, go to: Step5_3D_parameter_inference.ipynb
