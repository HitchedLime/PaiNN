import torch.nn.functional as F
from pytorch_lightning import seed_everything
from tqdm import trange

from QM9DataModule.QM9 import *
from network.PaiNN import Pain
from utils.PostProcessing import AtomwisePostProcessing
from utils.parameters import cli

args = [] # Specify non-default arguments in this list
args = cli(args)
seed_everything(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dm = QM9DataModule(
    target=args.target,
    data_dir=args.data_dir,
    batch_size_train=args.batch_size_train,
    batch_size_inference=args.batch_size_inference,
    num_workers=args.num_workers,
    splits=args.splits,
    seed=args.seed,
    subset_size=args.subset_size,
)
dm.prepare_data()
dm.setup()
y_mean, y_std, atom_refs = dm.get_target_stats(
    remove_atom_refs=True, divide_by_atoms=True
)

painn = Pain(
    num_message_passing_layers=args.num_message_passing_layers,
    num_features=args.num_features,
    num_outputs=args.num_outputs, 
    num_rbf_features=args.num_rbf_features,
    num_unique_atoms=args.num_unique_atoms,
    cutoff_dist=args.cutoff_dist,
)
post_processing = AtomwisePostProcessing(
    args.num_outputs, y_mean, y_std, atom_refs
)

painn.to(device)
post_processing.to(device)

optimizer = torch.optim.AdamW(
    painn.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,
)

painn.train()
pbar = trange(args.num_epochs)
for epoch in pbar:

    loss_epoch = 0.
    for batch in dm.train_dataloader():
        batch = batch.to(device)
        # print(f"This is z {batch.z}")
        # print(f"This is pos {batch.pos}")
        # print(f"This is batch {batch.batch}")
        atomic_contributions =painn(batch)
        # atomic_contributions = painn(
        #     atoms=batch.z,
        #     atom_positions=batch.pos,
        #     graph_indexes=batch.batch
        # )
        preds = post_processing(
            atoms=batch.z,
            graph_indexes=batch.batch,
            atomic_contributions=atomic_contributions,
        )
        loss_step = F.mse_loss(preds, batch.y, reduction='sum')

        loss = loss_step / len(batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_epoch += loss_step.detach().item()
    loss_epoch /= len(dm.data_train)
    pbar.set_postfix_str(f'Train loss: {loss_epoch:.3e}')

mae = 0
painn.eval()
with torch.no_grad():
    for batch in dm.test_dataloader():
        batch = batch.to(device)

        atomic_contributions = painn(
            atoms=batch.z,
            atom_positions=batch.pos,
            graph_indexes=batch.batch,
        )
        preds = post_processing(
            atoms=batch.z,
            graph_indexes=batch.batch,
            atomic_contributions=atomic_contributions,
        )
        mae += F.l1_loss(preds, batch.y, reduction='sum')

mae /= len(dm.data_test)
unit_conversion = dm.unit_conversion[args.target]
print(f'Test MAE: {unit_conversion(mae):.3f}')