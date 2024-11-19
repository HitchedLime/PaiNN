import torch
import torch.nn as nn


class Message(nn.Module):
    def __init__(self,
                 num_features: int = 128,
                 num_rbf_features: int = 20,
                 cutoff_dist: float = 5.0,
                 v: torch.Tensor,
                 s: torch.Tensor,
                 distances: torch.Tensor
                 ) -> None:
        super().__init__()

        self.num_features = num_features
        self.num_rbf_features = num_rbf_features
        self.cutoff_dist = cutoff_dist
        self.v = v
        self.s = s
        self.distances = distances

        # Actual block
        phi = nn.Sequential(nn.Linear(self.num_features, self.num_features),
                            nn.SiLU(),
                            nn.Linear(self.num_features, 3 * self.num_features))
        Filter = nn.Sequential(self.rbf(),
                               nn.Linear(self.num_rbf_features, 3 * self.num_features),
                               self.cos_cutoff())
        



    def rbf(self, r_dist: torch.Tensor) -> torch.Tensor:
        n = torch.arange(1, self.num_rbf_features+1)
        return torch.sin(n*torch.pi*torch.norm(r_dist) / self.cutoff_dist) / self.cutoff_dist
    
    def cos_cutoff(self, r_dist: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.cos(torch.pi * r_dist / self.cutoff_dist) + 1)
    

    def forward(self, s, v, )
    
        
        
        
        split1, split2, split3 = torch.split(phi * Filter, 3)

        delta_v = v * split1
        delta_v = torch.sum(delta_v + split3 * (self.distances/torch.norm(self.distances)), axis=0)

        delta_s = torch.sum(split2, axis=0)

        return delta_v, delta_s



    #raise NotImplementedError


# class Update(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#     raise NotImplementedError


class PaiNN(nn.Module):
    """
    Polarizable Atom Interaction Neural Network with PyTorch.
    """
    def __init__(
        self,
        num_message_passing_layers: int = 3,
        num_features: int = 128,
        num_outputs: int = 1,
        num_rbf_features: int = 20,
        num_unique_atoms: int = 100,
        cutoff_dist: float = 5.0,
    ) -> None:
        """
        Args:
            num_message_passing_layers: Number of message passing layers in
                the PaiNN model.
            num_features: Size of the node embeddings (scalar features) and
                vector features.
            num_outputs: Number of model outputs. In most cases 1.
            num_rbf_features: Number of radial basis functions to represent
                distances.
            num_unique_atoms: Number of unique atoms in the data that we want
                to learn embeddings for.
            cutoff_dist: Euclidean distance threshold for determining whether 
                two nodes (atoms) are neighbours.
        """
        super().__init__()
        
        self.num_message_passing_layers = num_message_passing_layers
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.num_rbf_features = num_rbf_features
        self.num_unique_atoms = num_unique_atoms
        self.cutoff_dist = cutoff_dist
        self.embedding = nn.Embedding(self.num_unique_atoms, self.num_features)

        raise NotImplementedError
    def message_passing(self) -> None:
        pass
    
    def message_update(self)  -> None:
        pass

    def forward(
        self,
        atoms: torch.LongTensor,
        atom_positions: torch.FloatTensor,
        graph_indexes: torch.LongTensor,
    ) -> torch.FloatTensor:
        """
        Forward pass of PaiNN. Includes the readout network highlighted in blue
        in Figure 2 in (Schütt et al., 2021) with normal linear layers which is
        used for predicting properties as sums of atomic contributions. The
        post-processing and final sum is perfomed with
        src.models.AtomwisePostProcessing.

        Args:
            atoms: torch.LongTensor of size [num_nodes] with atom type of each
                node in the graph.
            atom_positions: torch.FloatTensor of size [num_nodes, 3] with
                euclidean coordinates of each node / atom.
            graph_indexes: torch.LongTensor of size [num_nodes] with the graph 
                index each node belongs to.

        Returns:
            A torch.FloatTensor of size [num_nodes, num_outputs] with atomic
            contributions to the overall molecular property prediction.
        """
        raise NotImplementedError