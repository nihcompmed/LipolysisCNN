import torch
import pandas as pd
import numpy as np
import csv
from copy import copy
import sys
import os
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, random_split
#from pytorch_lightning.loggers import TensorBoardLogger
import torch.optim.lr_scheduler as lr_scheduler

torch.set_float32_matmul_precision('high')



if len(sys.argv) < 9:
    print("Usage: python tGV1102.py ntrain")
    sys.exit(1)

ntrain = int(sys.argv[1])
#print(ntrain)
max_lr = float(sys.argv[2])
C1steep = float(sys.argv[3])
bsize_myinput=int(sys.argv[4])
seed=int(sys.argv[5])
max_epochs=int(sys.argv[6])
ep_maxlr=int(sys.argv[7])
finalactfcn = sys.argv[8]  # Add the new input for the activation function
job_id=int(sys.argv[9])



npara=5

inpcsv = pd.read_csv('In250130FX2D25_mvlognmdirectparaphybd_2877269.csv', header=None)
inputs = torch.tensor(inpcsv.values[:, :npara ], dtype=torch.float32)
#print(inputs[0])
outcsv = pd.read_csv('Out250130FX2D25_mvlognmdirectparaphybd_2877269.csv', header=None)
Ou = torch.tensor(outcsv.values, dtype=torch.float32)
outputs = Ou.reshape(Ou.shape[0], 28, 4).transpose(1, 2) 

row_t = outputs[:, 0, :]  # 
row_G = outputs[:, 1, :]  # 
row_I = outputs[:, 2, :]  # 
row_F = outputs[:, 3, :]  # 
row_1onG= torch.reciprocal(row_G)
row_1onI= torch.reciprocal(row_I)
row_1onF= torch.reciprocal(row_F)
  

outputs = torch.cat((row_t.unsqueeze(1),row_G.unsqueeze(1),
                     row_t.unsqueeze(1),row_I.unsqueeze(1),
                     row_t.unsqueeze(1),row_F.unsqueeze(1),
                     row_G.unsqueeze(1),row_I.unsqueeze(1),
                     row_I.unsqueeze(1),row_F.unsqueeze(1),
                     row_G.unsqueeze(1),row_F.unsqueeze(1),
                     row_t.unsqueeze(1),row_1onG.unsqueeze(1),
                     row_t.unsqueeze(1),row_1onI.unsqueeze(1),
                     row_t.unsqueeze(1),row_1onF.unsqueeze(1),
                     row_1onG.unsqueeze(1),row_1onI.unsqueeze(1),
                     row_1onI.unsqueeze(1),row_1onF.unsqueeze(1),
                     row_1onG.unsqueeze(1),row_1onF.unsqueeze(1)),dim=1) #



# Extract the first 9 columns
#min_vals = torch.tensor(min_vals_csv.values[:, :npara], dtype=torch.float32)
#max_vals = torch.tensor(max_vals_csv.values[:, :npara], dtype=torch.float32)

min_vals = torch.min(inputs[:ntrain], dim=0).values  # Column-wise minimums
max_vals = torch.max(inputs[:ntrain], dim=0).values  # Column-wise maximums

#print(min_vals)
#print(max_vals)

# Perform min-max normalization using the loaded min and max values
inputs_rescaled = (inputs - min_vals) / (max_vals - min_vals)

min_per_column = torch.min(inputs_rescaled, dim=0).values

print(min_per_column)

#mean_rescale = torch.mean(inputs, dim=0)
#std_rescale=torch.std(inputs,dim=0)
#mean_diff = torch.mean(inputs[:, 6] - 22)
#inputs_rescaled = (inputs - mean_rescale) / (3*std_rescale)
#inputs_rescaled2 = (inputs[:, 6] - 22) / mean_diff
#inputs_rescaled = torch.cat((inputs_rescaled1, inputs_rescaled2.unsqueeze(1)), dim=1)

X = outputs[:ntrain].unsqueeze(1)
X_test = outputs[ntrain:].unsqueeze(1)
Y = inputs_rescaled[:ntrain]
Y_test = inputs_rescaled[ntrain:]

conv3size1=outputs.size()[1]//2



n_conv1outchannels=512
n_conv2outchannels=512
n_conv3outchannels=1024
n_dense1=1024
n_dense2=1024

bsize=bsize_myinput
dataset = torch.utils.data.TensorDataset(X, Y)
val_size = int(len(dataset) * 0.2) # 80% of the data

train_dataset, val_dataset = random_split(dataset, [len(dataset)-val_size, val_size], generator=torch.Generator().manual_seed(seed))
train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True,num_workers=8, generator=torch.Generator().manual_seed(seed))
val_loader = DataLoader(val_dataset, batch_size=bsize, shuffle=False,num_workers=8)



class NN_relu(pl.LightningModule):
    def __init__(self, batch_size,max_lr):#learning_rate):
        super().__init__()
        #self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_lr=max_lr
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_conv1outchannels, kernel_size=(2,2),stride=(2,1))# 
        self.conv2 = nn.Conv2d(in_channels=n_conv1outchannels, out_channels=n_conv2outchannels, kernel_size=(3,2),stride=(3,1))#
        self.padding=(0,1,0,0)
        self.replication_pad = nn.ReplicationPad2d((0, 1, 0, 0))

        self.pool1 = nn.MaxPool2d(kernel_size=(1,2))
        
        self.conv3 = nn.Conv2d(in_channels=n_conv2outchannels, out_channels=n_conv3outchannels, kernel_size=(conv3size1,2))

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(6*n_conv3outchannels, n_dense1)
        self.relu = nn.ReLU()
        self.logsig=nn.LogSigmoid()
        self.dense2 = nn.Linear(n_dense1,n_dense2)
        self.dense3 = nn.Linear(n_dense2, npara)
        self.softplus = nn.Softplus()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.train_losses = []
        self.val_losses = []
 
        

    def forward(self, x):       
        x_conv1 = self.conv1(x) 
        x_pad = self.replication_pad(x_conv1) 
        x_rep = x.repeat(1, n_conv2outchannels, 1, 1)
        x_cat= torch.cat([x_rep[:, :, i:i+2, :] if i % 2 == 0 else x_pad[:, :, i//2:i//2+1, :]
               for i in range(x_rep.size(2) + x_pad.size(2))], dim=2) #
        x = self.conv2(x_cat) #12-27
        x = self.pool1(x) #12-by-13,512 channels
        
        x = self.conv3(x) #1-by-12,512 channels
        x = self.pool1(x) #1-by-6,512 channels    
        x = self.flatten(x)
        x = self.dense1(x)
        #x = self.relu(x)
        x = torch.tanh(x)
        x = self.dense2(x)
        #x = self.relu(x)
        x = torch.tanh(x)
        x = self.dense3(x)
        x = self.relu(x)
        #x = (torch.tanh(x)+1)/2
        #x = torch.tanh(x)
        #x = -self.logsig(-x)
        return x
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.MSELoss()(outputs, labels)

        self.log('train_loss', loss)
        #batch_dictionary={"loss": loss,"log": logs, "total": total}  
        self.logger.experiment.add_scalar("Loss/Train",loss,self.current_epoch)
        self.logger.experiment.add_scalar("Epoch", self.current_epoch, self.current_epoch)
        self.training_step_outputs.append(loss)
        return loss

    
    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=self.batch_size)# | self.hparams.batch_size)

    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        val_loss = nn.MSELoss()(outputs, labels)
        self.log('val_loss', val_loss)
        self.logger.experiment.add_scalar("Loss/Valid",val_loss,self.current_epoch)
        self.logger.experiment.add_scalar("Epoch", self.current_epoch, self.current_epoch)
        self.validation_step_outputs.append(val_loss)
        return val_loss


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.max_lr)
        
        def lr_lambda(epoch):
            steep1 = C1steep / ep_maxlr
            steep2 = C1steep / max_epochs
            if epoch <= ep_maxlr:
                constant1 = 1 + np.exp(-steep1 * (ep_maxlr / 2))
                lr1 = self.max_lr * constant1 / (1 + np.exp(-steep1 * (epoch - ep_maxlr / 2)))
                return lr1
            else:
                constant2 = 1 + np.exp(-steep2 * (max_epochs / 4*3 - ep_maxlr))
                lr2 = self.max_lr * constant2 / (1 + np.exp(-steep2 * (max_epochs / 4*3 - epoch)))
                return lr2

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [scheduler]
    

    def loss(self, y_hat, y):
        return nn.MSELoss()(y_hat, y)

    def on_train_epoch_end(self):
        train_loss_mean = torch.stack(self.training_step_outputs).mean().item()
        val_loss_mean = torch.stack(self.validation_step_outputs).mean().item()
        self.train_losses.append(train_loss_mean)
        self.val_losses.append(val_loss_mean)

        self.training_step_outputs.clear()
        self.validation_step_outputs.clear()
        print(f"Epoch {self.current_epoch + 1}, Train Loss: {train_loss_mean:.4f}, Val Loss: {val_loss_mean:.4f}")

    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        # log the test loss
        self.log('test_loss', loss)
        return loss    

 

    def on_train_end(self):
        # Define the folder where you want to save the files
        save_folder = 'traininglosses'
        
        # Ensure the folder exists (this is optional if you know the folder already exists)
        os.makedirs(save_folder, exist_ok=True)
        
        # Save the train and validation losses in the specified folder
        np.savetxt(os.path.join(save_folder, f'{job_id}_train_losses.csv'), np.array(self.train_losses), delimiter=',')
        np.savetxt(os.path.join(save_folder, f'{job_id}_val_losses.csv'), np.array(self.val_losses), delimiter=',')


class NN_tanh01(pl.LightningModule):
    def __init__(self, batch_size,max_lr):#learning_rate):
        super().__init__()
        #self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_lr=max_lr
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_conv1outchannels, kernel_size=(2,2),stride=(2,1))# 
        self.conv2 = nn.Conv2d(in_channels=n_conv1outchannels, out_channels=n_conv2outchannels, kernel_size=(3,2),stride=(3,1))#
        self.padding=(0,1,0,0)
        self.replication_pad = nn.ReplicationPad2d((0, 1, 0, 0))

        self.pool1 = nn.MaxPool2d(kernel_size=(1,2))
        
        self.conv3 = nn.Conv2d(in_channels=n_conv2outchannels, out_channels=n_conv3outchannels, kernel_size=(conv3size1,2))

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(6*n_conv3outchannels, n_dense1)
        self.relu = nn.ReLU()
        self.logsig=nn.LogSigmoid()
        self.dense2 = nn.Linear(n_dense1,n_dense2)
        self.dense3 = nn.Linear(n_dense2, npara)
        self.softplus = nn.Softplus()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.train_losses = []
        self.val_losses = []
 
        

    def forward(self, x):       
        x_conv1 = self.conv1(x) 
        x_pad = self.replication_pad(x_conv1) 
        x_rep = x.repeat(1, n_conv2outchannels, 1, 1)
        x_cat= torch.cat([x_rep[:, :, i:i+2, :] if i % 2 == 0 else x_pad[:, :, i//2:i//2+1, :]
               for i in range(x_rep.size(2) + x_pad.size(2))], dim=2) #
        x = self.conv2(x_cat) #12-27
        x = self.pool1(x) #12-by-13,512 channels
        
        x = self.conv3(x) #1-by-12,512 channels
        x = self.pool1(x) #1-by-6,512 channels    
        x = self.flatten(x)
        x = self.dense1(x)
        #x = self.relu(x)
        x = torch.tanh(x)
        x = self.dense2(x)
        #x = self.relu(x)
        x = torch.tanh(x)
        x = self.dense3(x)
        #x = self.relu(x)
        x = (torch.tanh(x)+1)/2
        #x = torch.tanh(x)
        #x = -self.logsig(-x)
        return x
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.MSELoss()(outputs, labels)

        self.log('train_loss', loss)
        #batch_dictionary={"loss": loss,"log": logs, "total": total}  
        self.logger.experiment.add_scalar("Loss/Train",loss,self.current_epoch)
        self.logger.experiment.add_scalar("Epoch", self.current_epoch, self.current_epoch)
        self.training_step_outputs.append(loss)
        return loss

    
    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=self.batch_size)# | self.hparams.batch_size)

    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        val_loss = nn.MSELoss()(outputs, labels)
        self.log('val_loss', val_loss)
        self.logger.experiment.add_scalar("Loss/Valid",val_loss,self.current_epoch)
        self.logger.experiment.add_scalar("Epoch", self.current_epoch, self.current_epoch)
        self.validation_step_outputs.append(val_loss)
        return val_loss


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.max_lr)
        
        def lr_lambda(epoch):
            steep1 = C1steep / ep_maxlr
            steep2 = C1steep / max_epochs
            if epoch <= ep_maxlr:
                constant1 = 1 + np.exp(-steep1 * (ep_maxlr / 2))
                lr1 = self.max_lr * constant1 / (1 + np.exp(-steep1 * (epoch - ep_maxlr / 2)))
                return lr1
            else:
                constant2 = 1 + np.exp(-steep2 * (max_epochs / 4*3 - ep_maxlr))
                lr2 = self.max_lr * constant2 / (1 + np.exp(-steep2 * (max_epochs / 4*3 - epoch)))
                return lr2

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [scheduler]
    

    def loss(self, y_hat, y):
        return nn.MSELoss()(y_hat, y)

    def on_train_epoch_end(self):
        train_loss_mean = torch.stack(self.training_step_outputs).mean().item()
        val_loss_mean = torch.stack(self.validation_step_outputs).mean().item()
        self.train_losses.append(train_loss_mean)
        self.val_losses.append(val_loss_mean)

        self.training_step_outputs.clear()
        self.validation_step_outputs.clear()
        print(f"Epoch {self.current_epoch + 1}, Train Loss: {train_loss_mean:.4f}, Val Loss: {val_loss_mean:.4f}")

    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        # log the test loss
        self.log('test_loss', loss)
        return loss    

 

    def on_train_end(self):
        # Define the folder where you want to save the files
        save_folder = 'traininglosses'
        
        # Ensure the folder exists (this is optional if you know the folder already exists)
        os.makedirs(save_folder, exist_ok=True)
        
        # Save the train and validation losses in the specified folder
        np.savetxt(os.path.join(save_folder, f'{job_id}_train_losses.csv'), np.array(self.train_losses), delimiter=',')
        np.savetxt(os.path.join(save_folder, f'{job_id}_val_losses.csv'), np.array(self.val_losses), delimiter=',')




if finalactfcn == "relu":
    model_1 = NN_relu(batch_size=bsize_myinput, max_lr=max_lr)
    filename_template = f"model_FX2DNN_fullreci_mvlognmdirect_relu_{seed}_{job_id}_best{{epoch}}-{{val_loss:.5f}}"

elif finalactfcn == "tanh01":
    model_1 = NN_tanh01(batch_size=bsize_myinput, max_lr=max_lr)
    filename_template = f"model_FX2DNN_fullreci_mvlognmdirect_tanh01_{seed}_{job_id}_best{{epoch}}-{{val_loss:.5f}}"

else:
    raise ValueError("Invalid value for finalactfcn. Must be 'relu' or 'tanh01'.")


 


checkpoint_callback = ModelCheckpoint(
    filename=filename_template,#"model_GFX3D_cxcomputext22_treciGIF_240409_test_best{epoch}-{val_loss:.4f}",
    save_top_k=1,  
    monitor='val_loss',
    mode='min',  
)

trainer_1 = pl.Trainer(
    max_epochs=max_epochs,
    callbacks=[checkpoint_callback],
    gradient_clip_val=0.5,
    gradient_clip_algorithm="norm"
)



# # Train the model
trainer_1.fit(model_1, train_loader, val_loader)





