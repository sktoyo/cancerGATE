import tensorflow as tf
from dgl.nn.tensorflow import GATConv


class MultiModalAttentionGAE(tf.keras.Model):
    def __init__(self,
                 g,
                 in_dims,
                 dim_hiddens,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(MultiModalAttentionGAE, self).__init__()
        self.g = g
        self.dim_list = dim_hiddens
        self.embedding_layer_list = []
        self.decoding_layer_list = []
        self.layer_list = []
        self.num_layers = len(dim_hiddens) - 1
        self.num_features = len(in_dims)

        # input embedding layers
        for i in range(self.num_features):
            self.embedding_layer_list.append(
                tf.keras.layers.Dense(self.dim_list[0], activation='relu', input_shape=(in_dims[i],)))

        # output decoding layers
        for i in range(self.num_features):
            self.decoding_layer_list.append(tf.keras.layers.Dense(in_dims[i], activation='sigmoid'))

        # hidden layers
        for i in range(self.num_layers):
            if i == 0:
                self.layer_list.append(GATConv(
                    self.dim_list[i] * self.num_features, self.dim_list[i + 1], heads[i],
                    feat_drop, attn_drop, negative_slope, residual, activation))
            elif i != self.num_layers - 1:
                self.layer_list.append(GATConv(
                    self.dim_list[i], self.dim_list[i + 1], heads[i],
                    feat_drop, attn_drop, negative_slope, residual, activation))
            # output layer
            else:
                self.layer_list.append(GATConv(
                    self.dim_list[i], self.dim_list[i + 1], heads[i],
                    feat_drop, attn_drop, negative_slope, residual, activation=lambda x: x))

        for i in range(self.num_layers, 0, -1):
            # if i != 1:
            self.layer_list.append(GATConv(
                self.dim_list[i], self.dim_list[i - 1], heads[i - 1],
                feat_drop, attn_drop, negative_slope, residual, activation))

    def embedding(self, features):
        h = list()
        for i, feature in enumerate(features):
            h.append(self.embedding_layer_list[i](feature))

        h = tf.keras.layers.Concatenate()(h)

        for i, layer in enumerate(self.layer_list):
            if i == self.num_layers:
                break
            h = layer(self.g, h)
            if i != self.num_layers - 1:
                h = tf.reshape(h, (h.shape[0], -1))
        logits = tf.reduce_mean(h, axis=1)
        return logits

    def decoding(self, features):
        h = features
        h = tf.keras.activations.relu(h)
        for i in range(self.num_layers, len(self.layer_list)):
            h = self.layer_list[i](self.g, h)
            if i != len(self.layer_list) - 1:
                h = tf.reshape(h, (h.shape[0], -1))

        h = tf.reduce_mean(h, axis=1)

        result = list()
        for i in range(self.num_features):
            result.append(self.decoding_layer_list[i](h))
        return result

    def get_reconstructed(self, features):
        h = features
        logits_h = self.embedding(h)
        x = tf.transpose(logits_h)
        logits_st = tf.matmul(logits_h, x)
        return logits_st

    def get_reconstructed_edge(self, features, edges):
        h = features
        logits_h = self.embedding(h)
        edge_0 = [edge[0] for edge in edges]
        edge_1 = [edge[1] for edge in edges]
        logits_0 = tf.gather(logits_h, edge_0)
        logits_1 = tf.gather(logits_h, edge_1)
        logits_st = tf.math.multiply(logits_0, logits_1)
        logits_st = tf.reduce_sum(logits_st, axis=1)
        return logits_st

    def call(self, features):
        h = features
        h = self.embedding(h)
        feature_re = self.decoding(h)

        return feature_re


class GATE(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(GATE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, g, feature):
        return self.encoder(g, feature)

    def feat_decode(self, g, z):
        return self.decoder(g, z)

    def structure_decode(self, g, feature):
        z = self.encode(g, feature)
        x = tf.transpose(z)
        logits_st = tf.matmul(z, x)
        return logits_st

    def get_attention(self, g, feature):
        return self.encoder.get_sum_attention(g, feature)

    def call(self, g, feature):
        z = self.encode(g, feature)
        feature_re = self.feat_decode(g, z)
        return feature_re

    def get_reconstructed_edge(self, g, feature, edges):
        h = self.encode(g, feature)
        edge_0 = [edge[0] for edge in edges]
        edge_1 = [edge[1] for edge in edges]
        logits_0 = tf.gather(h, edge_0)
        logits_1 = tf.gather(h, edge_1)
        logits_st = tf.math.multiply(logits_0, logits_1)
        logits_st = tf.reduce_sum(logits_st, axis=1)
        return logits_st


class Encoder(tf.keras.Model):
    def __init__(self, dim_list, head_list, activation):
        super().__init__()
        self.conv1 = GATConv(dim_list[0], dim_list[1], head_list[0], activation=activation)
        self.conv2 = GATConv(dim_list[1], dim_list[2], head_list[1], activation=lambda x: x)

    def call(self, g, feature):
        h = self.conv1(g, feature)
        h = tf.reshape(h, (h.shape[0], -1))
        h = self.conv2(g, h)
        h = tf.reduce_mean(h, axis=1)
        return h

    def get_sum_attention(self, g, feature):
        h, attention1 = self.get_attention(self.conv1, g, feature)
        h = tf.reshape(h, (h.shape[0], -1))
        attention1 = self.reshape_attention(attention1)
        h, attention2 = self.get_attention(self.conv2, g, h)
        attention2 = self.reshape_attention(attention2)
        total_attention = attention1 + attention2
        return total_attention

    def get_attention(self, layer, feature, g):
        h, attention = layer(g, feature, True)
        attention = tf.reduce_sum(attention, axis=1)
        return h, attention

    def reshape_attention(self, attention):
        attention = tf.reshape(attention, [attention.shape[0]])
        return attention


class Decoder(tf.keras.Model):
    def __init__(self, dim_list, head_list, activation):
        super().__init__()
        self.conv1 = GATConv(dim_list[2], dim_list[1], head_list[1], activation=activation)
        self.conv2 = GATConv(dim_list[1], dim_list[0], head_list[0], activation=tf.keras.activations.relu)

    def call(self, g, feature):
        h = self.conv1(g, feature)
        h = tf.reshape(h, (h.shape[0], -1))
        h = self.conv2(g, h)
        h = tf.reduce_mean(h, axis=1)
        return h


class AttentionGAE_structure_only(tf.keras.Model):
    def __init__(self,
                 g,
                 in_dims,
                 dim_hiddens,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(AttentionGAE_structure_only, self).__init__()
        self.g = g
        self.dim_list = dim_hiddens
        self.embedding_layer_list = []
        self.decoding_layer_list = []
        self.layer_list = []
        self.num_layers = len(dim_hiddens) - 1
        self.num_features = len(in_dims)

        # input embedding layers
        for i in range(self.num_features):
            self.embedding_layer_list.append(
                tf.keras.layers.Dense(self.dim_list[0], activation='relu', input_shape=(in_dims[i],)))

        # hidden layers
        for i in range(self.num_layers):
            if i == 0:
                self.layer_list.append(GATConv(
                    self.dim_list[i] * self.num_features, self.dim_list[i + 1], heads[i],
                    feat_drop, attn_drop, negative_slope, residual, activation))
            elif i != self.num_layers - 1:
                self.layer_list.append(GATConv(
                    self.dim_list[i], self.dim_list[i + 1], heads[i],
                    feat_drop, attn_drop, negative_slope, residual, activation))
            # output layer
            else:
                self.layer_list.append(GATConv(
                    self.dim_list[i], self.dim_list[i + 1], heads[i],
                    feat_drop, attn_drop, negative_slope, residual, activation=lambda x: x))

    def embedding(self, features):
        h = list()
        for i, feature in enumerate(features):
            h.append(self.embedding_layer_list[i](feature))

        h = tf.keras.layers.Concatenate()(h)

        for i, layer in enumerate(self.layer_list):
            if i == self.num_layers:
                break
            h = layer(self.g, h)
            if i != self.num_layers - 1:
                h = tf.reshape(h, (h.shape[0], -1))
        logits = tf.reduce_mean(h, axis=1)
        return logits

    def get_reconstructed(self, features):
        h = features
        logits_h = self.embedding(h)
        x = tf.transpose(logits_h)
        logits_st = tf.matmul(logits_h, x)
        return logits_st

    def get_reconstructed_edge(self, features, edges):
        h = features
        logits_h = self.embedding(h)
        edge_0 = [edge[0] for edge in edges]
        edge_1 = [edge[1] for edge in edges]
        logits_0 = tf.gather(logits_h, edge_0)
        logits_1 = tf.gather(logits_h, edge_1)
        logits_st = tf.math.multiply(logits_0, logits_1)
        logits_st = tf.reduce_sum(logits_st, axis=1)
        return logits_st
