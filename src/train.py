import torch
from model.CancerGATE import *
from preprocessing.Preprocess import load_preprocess_results
import torch_geometric.transforms as trans
from torch_geometric.data import Data


input_dict = load_preprocess_results()
edge_index = torch.tensor(input_dict['edge_index'], dtype=torch.long)
feature_dict = input_dict['subtype_x']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = trans.Compose([
    trans.ToDevice(device),
    trans.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=False),
])

data_dict = dict()
for subtype in feature_dict:
    data = Data(x=torch.tensor(feature_dict[subtype].values, dtype=torch.float32), edge_index=edge_index)
    train_data, val_data, test_data = transform(data)
    break

data = Data(x=torch.tensor(feature_dict['Tumor'].values, dtype=torch.float32), edge_index=edge_index)
train_data, val_data, test_data = transform(data)

encoder = GATEncoder([13, 64, 32])
feat_decoder = FeatDecoder(encoder)
model = GATE(encoder, feat_decoder)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)


def train(input_data):
    model.train()
    optimizer.zero_grad()
    z = model.encode(input_data.x, input_data.pos_edge_label_index)
    loss = model.total_loss(input_data.x, z, data.edge_index)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(input_data):
    model.eval()
    z = model.encode(train_data.x, train_data.edge_index)
    return model.test(z, input_data.pos_edge_label_index, input_data.neg_edge_label_index)

# from torch_geometric.datasets import KarateClub
# data = KarateClub()[0]
# train_data, val_data, test_data = transform(data)


print(test_data)
for epoch in range(1, 150 + 1):
    loss = train(train_data)
    auc, ap = test(test_data)
    print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}, Loss: {loss: 4f}')
