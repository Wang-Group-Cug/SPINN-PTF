import numpy as np;import pandas as pd ;import torch;from torch import nn;from torch import optim;import os;from torch.utils.data import DataLoader,TensorDataset ;
device='cpu'
torch.set_default_dtype(torch.float64)
import argparse

def process_data(input_path, output_path):

    Texture=np.array(pd.read_csv(input_path)) # input

    class FNN_test(nn.Module):
        def __init__(self,input_size,hidden_size,output_size,layers,**kwargs) -> None:
            super(FNN_test,self).__init__(**kwargs)
            self.input_size=input_size
            self.hideen_size=hidden_size
            self.seq=nn.Sequential()
            for i in range(layers):
                input_size=self.input_size if i ==0 else hidden_size
                self.seq.append(nn.Linear(input_size,self.hideen_size))
                self.seq.append(nn.ReLU())
            self.seq.append(nn.Linear(self.hideen_size,output_size))
            self.seq.append(nn.Sigmoid())
        
        def forward(self,x):
            '''x[batch_size,x_num]'''
            out=self.seq(x)
            out2=torch.zeros(out.shape)
            out2[:,[0]]=-3+2.3010*out[:,[0]]  
            out2[:,[1]]=0.0043+1.17176988527304*out[:,[1]] 
            out2[:,[2]]=-2+2.17609125905568*out[:,[2]] 
            out2[:,[3]]=-0.698970004336019+0.628388930050312*out[:,[3]]
            return out2
    FNN=FNN_test(input_size=4,hidden_size=374,output_size=4,layers=3)
    FNN.to(device=device)    

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"M3_H3_SWRC.pth")
    FNN.load_state_dict(torch.load(model_path))

    Tex_list=torch.tensor(np.array(Texture),dtype=torch.float64)  
    with torch.no_grad():  # Predict
        FNN.eval()
        out_para=FNN(Tex_list)
    out_para_SWRC=pd.DataFrame(10**out_para)

    class FNN_test(nn.Module):
        def __init__(self,input_size,hidden_size,output_size,layers,**kwargs) -> None:
            super(FNN_test,self).__init__(**kwargs)
            self.input_size=input_size
            self.hideen_size=hidden_size
            self.seq=nn.Sequential()
            for i in range(layers):
                input_size=self.input_size if i ==0 else hidden_size
                self.seq.append(nn.Linear(input_size,self.hideen_size))
                self.seq.append(nn.ReLU())
            self.seq.append(nn.Linear(self.hideen_size,output_size))
            self.seq.append(nn.Sigmoid())
        
        def forward(self,x):
            '''x[batch_size,x_num]'''
            out=self.seq(x)
            out2=torch.zeros(out.shape)
            out2[:, [0]] = -4 + 8 * out[:, [0]]  
            out2[:, [1]] = -4 + 8.5 * out[:, [1]] 
            return out2
    FNN=FNN_test(input_size=4,hidden_size=374,output_size=2,layers=3)
    FNN.to(device=device)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"M3_H3_K.pth")
    FNN.load_state_dict(torch.load(model_path))
    Tex_list=torch.tensor(np.array(Texture),dtype=torch.float64)  
    with torch.no_grad():  # Predict
        FNN.eval()
        out_para=FNN(Tex_list)
    out_para_K=pd.DataFrame(10**out_para)

    pd.concat([pd.DataFrame(out_para_SWRC),pd.DataFrame(out_para_K)],axis=1).to_csv(output_path,header=['alpha','n','m','ths','Kha','Ks']) # Save

# other
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="csv")
    parser.add_argument("--output", required=True, help="csv")
    args = parser.parse_args()
    
    process_data(args.input, args.output)