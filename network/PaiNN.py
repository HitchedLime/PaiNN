import torch 
import torch.nn as nn
import torch.nn.functional as F
import math


# look into paper for formula 
def cos_cut(edge_dis,cutoff):
    return torch.where(edge_dis<cutoff,0.5*torch.cos(edge_dis/cutoff)+1
                       ,torch.tensor(0.0,dtype=edge_dis.dtype))
#look for paper for formula 
def rbf(edge_dis,rbf_features,rbf_unique_atoms,cutoff):

    n= torch.arange(rbf_features,device= edge_dis.device)+1
    inner_part =(n*torch.pi/cutoff)*edge_dis.unsqueeze(-1)
    return torch.sin(inner_part)/edge_dis.unsqueeze(-1)


class Message(nn.Module):
    def __init__(self,num_features,edge_size,cutoff) -> None:
        super().__init__()
        #node size should be number of  features 
        self.edge_size = edge_size
        self.num_features = num_features
        self.cutoff = cutoff

        #node size 3*128 = 384 as output 
        self.scalar_msg = nn.Sequential(nn.Linear(self.num_features,self.num_features),
                                        nn.SiLU(),
                                        nn.Linear(num_features,num_features*3))
        self.filter= nn.Linear(edge_size, num_features * 3)
    def forward(self,node_s,node_vec,edge,edge_difference,edge_dis):
        #filter  its marked as W in the paper 
        filter_W =self.filter(rbf,edge_dis,self.edge_size,self.cutoff)
        filter_W  =filter_W *  cos_cut(edge_dis,self.cutoff).unsqueeze(-1)
        s_output = self.scalar_msg(node_s)
        filer_output = filter_W * s_output[edge[:, 1]]

        gate_state_vector, gate_edge_vector, message_scalar = torch.split(
            filer_output, 
            self.num_features,
            dim = 1,
        )

        # the arrow from r_ij  hamadar with split 
        message_vec = node_vec[edge[:,1]] * gate_state_vector

        #the aroorw from v_i after split 
        edge_vec = gate_edge_vector.unsqueeze(-1) *(edge_difference/edge_dis.unsqueeze(-1)).unsqueeze(-1)
        

        temp_s = torch.zeros_like(node_s)
        temp_vec = torch.zeros_like(node_vec)

        #solved my problem when 
        # temp_s.index_add_(0, edge[:, 0], message_scalar)
        # temp_vec.index_add_(0, edge[:, 0], message_vec)
        
      

        temp_s.scatter_add_(0, edge[:, 0].unsqueeze(1).expand(-1, self.num_features), message_scalar)
        temp_vec.scatter_add_(0, edge[:, 0].unsqueeze(1).expand(-1, message_vec.size(1)), message_vec)


        delta_node_scalar = node_s + temp_s
        delta_node_vector = node_vec + temp_vec





        #delta V_i vec and delga s_i
        return delta_node_vector,delta_node_scalar  

class Update(nn.Module):
    def __init__(self,num_features) -> None:
        super().__init__()
        self.U_dot = nn.Linear(num_features,num_features )
        self.V_dot =  nn.Linear(num_features,num_features)
        
        self.update_mlp = nn.Sequential(nn.Linear(num_features*2,num_features),
                                        nn.SiLU(),
                                        nn.Linear(num_features,3*num_features))
    def forward(self,node_s,node_vec):
        Uv= self.U_dot(node_vec)
        Vv =self.V_dot(node_s) 
class Pain(nn.Module):

    def __init__(
        self,
        num_message_passing_layers: int = 3,
        num_features: int = 128,
        num_outputs: int = 1,
        num_rbf_features: int = 20,
        num_unique_atoms: int = 100,
        cutoff_dist: float = 5.0,
    ) -> None:
        super().__init__()


 