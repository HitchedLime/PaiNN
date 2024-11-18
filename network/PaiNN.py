import torch 
import torch.nn as nn
from sympy import print_tree
from torch_geometric.nn import radius_graph


# def cos_cut(distances, cutoff):
    
#     mask = (distances < cutoff).float()  # Only values within cutoff contribute 
#     cutoff_values = 0.5 * (torch.cos(distances * (torch.pi / cutoff)) + 1.0)
#     return cutoff_values * mask



# look into paper for formula 
def cos_cut(edge_dis,cutoff):
    return torch.where(edge_dis<=cutoff,0.5*(torch.cos(torch.pi*edge_dis/cutoff)+1)
                       ,torch.tensor(0.0,dtype=edge_dis.dtype))
#look for paper for formula 
def rbf(edge_dis,num_rbf_features,cutoff):

    n= torch.arange(num_rbf_features,device = edge_dis.device)+1
    
    inner_part =(n*torch.pi/cutoff)*edge_dis.unsqueeze(-1)
    return torch.sin(inner_part)/edge_dis.unsqueeze(-1)


class Message(nn.Module):
    def __init__(self,num_features,edge_size,cutoff) -> None:
        super().__init__()
        #node size should be number of  features 
        self.edge_size = edge_size
        self.num_features = num_features
        self.cutoff = cutoff
        self.num_atoms = 20

        #node size 3*128 = 384 as output 
        self.scalar_msg = nn.Sequential(nn.Linear(self.num_features,self.num_features),
                                        nn.SiLU(),
                                        nn.Linear(num_features,num_features*3))
        self.filter= nn.Linear(self.num_atoms, self.num_features*3)


    def forward(self,node_s,node_vec,edge,edge_difference,edge_dis):
        #filter  its marked as W in the paper 
        filter_W =self.filter(rbf(edge_dis,self.num_atoms,self.cutoff))[edge[:, 1]]
        cos_cut_var =  cos_cut(edge_dis,self.cutoff).unsqueeze(-1)[edge[:, 1]]


        filter_W  =filter_W * cos_cut_var

        s_output = self.scalar_msg(node_s)

        # print(filter_W.shape)
        # print(cos_cut_var.shape)
        # print(s_output.shape)
        # print(s_output[edge[:, 1]].shape)
        # print(edge.shape)






        filer_output = filter_W * s_output[edge[:, 1]]

        gate_state_vector, gate_edge_vector, message_scalar = torch.split(
            filer_output, 
            self.num_features,
            dim = 1,
        )
        # print("Gate_state", gate_state_vector.shape)
        # print("gate_edge",gate_edge_vector.shape)
        # print("message_scalar",message_scalar.shape)
        # print("node_ves",node_vec[edge[:,1]].shape)
        # the arrow from r_ij  hamadar with split 
        message_vec = node_vec[edge[:,1]] * gate_state_vector.unsqueeze(2)

        # print("Gate edge vector",gate_edge_vector.unsqueeze(-1).shape)
        # print("edge diff",edge_dis.unsqueeze(-1).shape)
        # print("edge_ve",(edge_difference/edge_dis.unsqueeze(-1)).unsqueeze(-1).shape)
        #the aroorw from v_i after split 
        edge_vec = gate_edge_vector.unsqueeze(1) *(edge_difference[edge[:,1]]/edge_dis[edge[:,1]].unsqueeze(-1)).unsqueeze(-1)
        

        temp_s = torch.zeros_like(node_s)
        temp_vec = torch.zeros_like(node_vec)

        #solved my problem when 
        temp_s.index_add_(0, edge[:, 0], message_scalar)
        temp_vec.index_add_(0, edge[:, 0], message_vec)
        
      

        # temp_s.scatter_add_(0, edge[:, 0].unsqueeze(1).expand(-1, self.num_features), message_scalar)
        # temp_vec.scatter_add_(0, edge[:, 0].unsqueeze(1).expand(-1, message_vec.size(1)), message_vec)


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
        Vv_norm = torch.linalg(Vv,dim= 1)


        mlp_input = torch.cat((Vv_norm, node_s), dim=1)
        mlp_output = self.update_mlp(mlp_input)


        a_vv, a_sv, a_ss = torch.split(
            mlp_output,                                        
            node_vec.shape[-1],      # split it threaa wayws                                   
            dim = 1,
        )
        
        delta_v = a_vv.unsqueeze(1) * Uv
        #not sure about this one 
        dot_prod = torch.sum(Uv * Vv, dim=1)
        delta_s = a_sv * dot_prod + a_ss
        
        return node_s + delta_s, node_vec + delta_v


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
        num_atoms = 100
        self.num_features = num_features
        self.cutoff = cutoff_dist 
        self.embedding  = nn.Embedding(num_unique_atoms,num_features)
        self.num_layers =3
        self.num_unique_atoms=num_unique_atoms
        #### Architecture making the block first thinng in figure 2 
        
        self.message_layers = nn.ModuleList(
            [
                Message(self.num_features, num_rbf_features, self.cutoff)
                for _ in range(self.num_layers)
            ]
        )
        self.update_layers = nn.ModuleList(
            [
                Update(self.num_features)
                for _ in range(self.num_layers)
            ]            
        )
        self.last_layer = nn.Sequential(nn.Linear(self.num_features,self.num_features),
        nn.SiLU(),
        nn.Linear(self.num_features,num_outputs))

        normalization=True
        atom_normalization =True
        target_mean=[0.0]
        target_stddev=[1.0]
        self.normalization = torch.nn.Parameter(
            torch.tensor(normalization), requires_grad=False
        )
        self.atomwise_normalization = torch.nn.Parameter(
            torch.tensor(atom_normalization), requires_grad=False
        )
        self.normalize_stddev = torch.nn.Parameter(
            torch.tensor(target_stddev[0]), requires_grad=False
        )
        self.normalize_mean = torch.nn.Parameter(
            torch.tensor(target_mean[0]), requires_grad=False

        )
    def forward(self, input_dict, compute_forces=True):
    # Extract relevant inputs from input_dict
        x = input_dict['x']            # Atom features
        pos = input_dict['pos']         # Atom positions
        z = input_dict['z']             # Atomic numbers
        edge_index = input_dict['edge_index']    # Edge indices
        edge_attr = input_dict['edge_attr']      # Edge attributes
        batch = input_dict['batch']     # Batch assignments
        num_atoms = torch.bincount(batch)        # Number of atoms per molecule
        edge_index =radius_graph(x, r=1.5, batch=batch, loop=False)

        edge_diff = pos[edge_index[1]] - pos[edge_index[0]]  
        if compute_forces:
            edge_diff.requires_grad_()  
        
        edge_dist = torch.linalg.norm(edge_diff, dim=1)

        # Initialize scalar and vectorial node features
        node_scalar = self.embedding(z)  
        node_vector = torch.zeros((pos.shape[0], self.num_features, 3), device=pos.device)

        # Message passing layers
        for message_layer, update_layer in zip(self.message_layers, self.update_layers):
            node_scalar, node_vector = message_layer(node_scalar, node_vector, edge_index, edge_diff, edge_dist)
            node_scalar, node_vector = update_layer(node_scalar, node_vector)

        # Fully connected layers to predict energy contributions
        x0 = node_scalar  
        z1 = self.linear_1(x0)
        z1.retain_grad()
        x1 = self.silu(z1)
        node_scalar = self.linear_2(x1)  

        node_scalar.squeeze_()

        # Aggregate atomic contributions into molecular energy
        image_idx = torch.arange(num_atoms.shape[0], device=edge_index.device)
        image_idx = torch.repeat_interleave(image_idx, num_atoms)

        energy = torch.zeros_like(num_atoms).float()
        energy.index_add_(0, image_idx, node_scalar)  

        result_dict = {'energy': energy}

        if compute_forces:
            dE_ddiff = torch.autograd.grad(
                energy,
                edge_diff,
                grad_outputs=torch.ones_like(energy),
                retain_graph=True,
                create_graph=True,
            )[0]

            i_forces = torch.zeros_like(pos).index_add(0, edge_index[0], dE_ddiff)
            j_forces = torch.zeros_like(pos).index_add(0, edge_index[1], -dE_ddiff)
            forces = i_forces + j_forces

            result_dict['forces'] = forces

        return result_dict
