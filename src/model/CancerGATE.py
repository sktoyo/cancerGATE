import torch
from torch.autograd import Variable
from torch_geometric.nn import GAE, GCNConv, GATConv


class GATE(GAE):
    def __init__(self, encoder, feat_decoder):
        super().__init__(encoder)
        self.feat_decoder = feat_decoder
        self.feat_loss = torch.nn.MSELoss()

    def feature_loss(self, x, z):
        x_recon = self.feat_decoder(z)
        features_loss = self.feat_loss(x, x_recon)
        return features_loss

    def structure_loss(self, z, edge_index):
        structure_loss = self.recon_loss(z, edge_index)
        return structure_loss

    def total_loss(self, x, z, edge_index, alpha=1):
        feature_loss = self.feature_loss(x, z)
        structure_loss = self.structure_loss(z, edge_index)
        return structure_loss + alpha * feature_loss

    def get_attention(self, x, edge_index):
        attention = self.encoder.conv1(x, edge_index)[1]
        attention += self.encoder.conv2(x, edge_index)[1]
        return attention


class GCNEncoder(torch.nn.Module):
    def __init__(self, dim_list):
        super().__init__()
        self.conv1 = GCNConv(dim_list[0], dim_list[1])
        self.conv2 = GCNConv(dim_list[1], dim_list[2])

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class GATEncoder(torch.nn.Module):
    def __init__(self, dim_list):
        super().__init__()
        self.conv1 = GATConv(dim_list[0], dim_list[1])
        self.conv2 = GATConv(dim_list[1], dim_list[2])

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class FeatDecoder(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.weight_list = self.get_decoder_weights()

    def get_decoder_weights(self):
        decode_weight1 = self.encoder.conv2.lin_src.weight
        decode_weight2 = self.encoder.conv1.lin_src.weight
        return [decode_weight1, decode_weight2]

    def forward(self, z):
        h = torch.matmul(z, self.weight_list[0])
        recon_x = torch.matmul(h, self.weight_list[1])
        return recon_x


if __name__ == "__main__":
    print(torch.ones((3,4)))

    encoder = GATEncoder([13, 64, 32])
    print(encoder.conv1.lin_src.weight.shape)
    print(encoder.conv2.lin_src.weight.shape)
    feat_decoder = FeatDecoder(encoder)
    model = GATE(encoder, feat_decoder)

    import numpy as np
    test_x = np.random.rand(100, 13)
    test_network = np.random.randint(99, size=(2, 1000))

    test_x = torch.Tensor(test_x)
    test_network = torch.LongTensor(test_network)
    print(model.total_loss(test_x, test_network, 1))
    print(model.feat_decoder(model.encoder(test_x,test_network)).shape)
