import pickle
from tqdm import tqdm
from sklearn.model_selection import  KFold
from gatev2_utils import *
from gatev2_models import *


mse = tf.keras.losses.MeanSquaredError()
wce = tf.nn.weighted_cross_entropy_with_logits
total_epoch = 150

with open('../data/input_data.pkl', 'rb') as f:
    input_dict = pickle.load(f)

feature_dict = input_dict['subtype_x']
for subtype in feature_dict:
    feature_dict[subtype] = tf.convert_to_tensor(feature_dict[subtype], dtype=tf.float32)
humannet_edges_node1, humannet_edges_node2 = input_dict['edge_index']

# create network
humannet_dgl = dgl.graph((humannet_edges_node1, humannet_edges_node2))
humannet_dgl = humannet_dgl.to('/gpu:0')

humannet_adj = sp.csr_matrix((np.ones(len(humannet_edges_node1)), (humannet_edges_node1, humannet_edges_node2)), shape=(len(set(humannet_edges_node1)), len(set(humannet_edges_node1))))
# total adj as label
labels = tf.sparse.to_dense(convert_sparse_matrix_to_sparse_tensor(humannet_adj))


loss_batch_size = 2048
batch_count = labels.shape[0] // loss_batch_size
dim_list = [13, 128, 64]
subtype_list = list(feature_dict.keys())

pos_weight = float(humannet_adj.shape[0] * humannet_adj.shape[0] - humannet_adj.sum()) / humannet_adj.sum()
norm = humannet_adj.shape[0] * humannet_adj.shape[0] / float((humannet_adj.shape[0] * humannet_adj.shape[0] - humannet_adj.sum()) * 2)

head_list = [8, 8]
activation = tf.keras.activations.relu
model_name = 'GATE'

positive_set, negative_set, train_edges, valid_pos, valid_neg, test_pos, test_neg = data_split(humannet_adj, 0.1, 0.1)


def create_model(dim_list, head_list, activation):
    encoder = Encoder(dim_list, head_list, activation)
    decoder = Decoder(dim_list, head_list, activation)
    model = GATE(encoder, decoder)
    return model


def train(model, g, feature):
    for batch_idx in range(batch_count):
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_weights)
            feature_loss, structure_loss, loss = batch_loss(model, batch_idx, g, feature)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients((grad, var) for (grad, var) in zip(grads, model.trainable_variables) if
                                  grad is not None)  ### suppress warning
    return feature_loss


def batch_loss(model, batch_idx, g, feature):
    structure_decode = model.structure_decode(g, feature)
    feature_decode = model(g, feature)
    if batch_idx == batch_count - 1:
        structure_loss = tf.reduce_sum(
            wce(labels[batch_idx * loss_batch_size:], structure_decode[batch_idx * loss_batch_size:],
                pos_weight=pos_weight))
        feature_loss = mse(feature[batch_idx * loss_batch_size:], feature_decode[batch_idx * loss_batch_size:])
    else:
        structure_loss = tf.reduce_sum(wce(labels[batch_idx * loss_batch_size:(batch_idx + 1) * loss_batch_size],
                                           structure_decode[batch_idx * loss_batch_size:(batch_idx + 1) * loss_batch_size],
                                           pos_weight=pos_weight))
        feature_loss = mse(feature[batch_idx * loss_batch_size:(batch_idx + 1) * loss_batch_size],
                           feature_decode[batch_idx * loss_batch_size:(batch_idx + 1) * loss_batch_size])

    structure_loss = structure_loss / (labels.shape[0] ** 2) * norm
    loss = tf.add(feature_loss, structure_loss)
    return feature_loss, structure_loss, loss


def print_performacne(model, g, feature, train_pos, train_neg, feature_loss):
    roc_curr, ap_curr = get_roc_score2(model, g, feature, train_pos, train_neg)
    print(subtype,
          "Feature loss {:.4f} | AUROC {:.4f} | AP {:.4f}".format(feature_loss.numpy(), roc_curr, ap_curr))
    return roc_curr, ap_curr

from collections import defaultdict
result_dict = defaultdict(list)

for subtype in ['Normal', 'Tumor']:
    model = create_model(dim_list, head_list, activation)
    kf = KFold(n_splits=5, shuffle=True)
    cv = 0
    for train_index, test_index in kf.split(positive_set):
        cv += 1
        train_pos, valid_pos = positive_set[train_index], positive_set[test_index]
        train_neg, valid_neg = negative_set[train_index], negative_set[test_index]
        train_g = create_input_network(train_pos)
        optimizer = tf.keras.optimizers.Adam(1e-2)

        for epoch in tqdm(range(total_epoch)):
            feature_loss = train(model, train_g, feature_dict[subtype])
            if epoch % 20 == 0:  # print first loss
                print_performacne(model, train_g, feature_dict[subtype], train_pos, train_neg, feature_loss)

        print_performacne(model, train_g, feature_dict[subtype], train_pos, train_neg, feature_loss)
        print("Validation")
        roc_curr, ap_curr = print_performacne(model, train_g, feature_dict[subtype], valid_pos, valid_neg, feature_loss)

        model.save_weights(
            "../result/checkpoints/CV{}_{}_{}_{}_{}_full checkpoints/model.ckpt".format(str(cv), model_name,
                                                                                                subtype, total_epoch,
                                                                                                ' '.join(
                                                                                                    [str(dim) for dim in
                                                                                                     dim_list])))
        print(
            "Saving model CV{}_{}_{}_{}_{}_full.ckpt".format(str(cv), model_name, subtype, total_epoch,
                                                             ' '.join([str(dim) for dim in dim_list])))

        result_dict['subtype'].append(subtype)
        result_dict['cv'].append(cv)
        result_dict['roc_curr'].append(roc_curr)
        result_dict['ap_curr'].append(ap_curr)

import pandas as pd
pd.DataFrame(result_dict).to_csv("../result/performance_record_{}.csv".format(' '.join([str(dim) for dim in dim_list])))













# print()
#
# def main_cv(input_argue):
#     ### main training code
#     subtype_list = ['Tumor', 'Normal']
#     for subtype in subtype_list:
#         print(subtype)
#         optimizer = tf.keras.optimizers.Adam(1e-2)
#         # bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#         mse = tf.keras.losses.MeanSquaredError()
#         wce = tf.nn.weighted_cross_entropy_with_logits
#
#         exp_input = subtype_expression_input_dict[subtype]
#         mut_input = subtype_mutation_input_dict[subtype]
#
#         in_dims, train_input = check_input_argue(input_argue, exp_input, mut_input)
#
#         num_features = len(in_dims)
#         model = MultiModalAttentionGAE(g, in_dims, dim_hiddens, head_list, activation, feat_drop, attn_drop, negative_slope,
#                                        residual)
#
#         batch_count = labels.shape[0] // loss_batch_size
#
#         for epoch in tqdm(range(total_epoch)):
#             for i in range(batch_count):
#                 with tf.GradientTape() as tape:
#                     tape.watch(model.trainable_weights)
#                     logits_st = model.get_reconstructed(train_input)
#                     if i == batch_count - 1:
#                         loss_value_st = tf.reduce_sum(
#                             wce(labels[i * loss_batch_size:], logits_st[i * loss_batch_size:], pos_weight=pos_weight))
#                     else:
#                         loss_value_st = tf.reduce_sum(wce(labels[i * loss_batch_size:(i + 1) * loss_batch_size],
#                                                           logits_st[i * loss_batch_size:(i + 1) * loss_batch_size],
#                                                           pos_weight=pos_weight))
#                     loss_value_st = loss_value_st / (labels.shape[0] ** 2) * norm
#
#                     # add feature loss batch
#                     logits_ft = model(train_input, training=True)
#                     loss_ft = tf.zeros(1)
#                     for j in range(num_features):
#                         if i == batch_count - 1:
#                             loss_ft = tf.add(loss_ft, mse(train_input[j][i * loss_batch_size:],
#                                                           logits_ft[j][i * loss_batch_size:]))
#                         else:
#                             loss_ft = tf.add(loss_ft, mse(train_input[j][i * loss_batch_size:(i + 1) * loss_batch_size],
#                                                           logits_ft[j][i * loss_batch_size:(i + 1) * loss_batch_size]))
#
#                     loss = tf.add(loss_ft, loss_value_st)
#                 grads = tape.gradient(loss, model.trainable_weights)
#                 optimizer.apply_gradients((grad, var) for (grad, var) in zip(grads, model.trainable_variables) if
#                                           grad is not None)  ### suppress warning
#
#             if epoch % 20 == 0:  # print first loss
#                 train_roc_curr, train_ap_curr = get_roc_score(model, train_input, positive_set, negative_set)
#                 print(subtype, dim_hiddens,
#                       "Feature loss {:.4f} | Train AUROC {:.4f} | Train AP {:.4f}".format(loss_ft.numpy()[0],
#                                                                                           train_roc_curr,
#                                                                                           train_ap_curr))
#
#         train_roc_curr, train_ap_curr = get_roc_score(model, train_input, train_pos, train_neg)
#         print(subtype, dim_hiddens,
#               "Feature loss {:.4f} | Train AUROC {:.4f} | Train AP {:.4f}".format(loss_ft.numpy()[0], train_roc_curr,
#                                                                                   train_ap_curr))
#
#         test_roc_curr, test_ap_curr = get_roc_score(model, train_input, valid_pos, valid_neg)
#         print(subtype, dim_hiddens,
#               "Feature loss {:.4f} | Test AUROC {:.4f} | Test AP {:.4f}".format(loss_ft.numpy()[0], test_roc_curr,
#                                                                                 test_ap_curr))
#
#         model.save_weights(
#             "./checkpoints/CV{}_{}_{}_{}_{}_{}_full checkpoints/model.ckpt".format(str(cv), input_argue, model_name,
#                                                                                    subtype, total_epoch, ' '.join(
#                     [str(dim) for dim in dim_hiddens])))
#         print(
#             "Saving model CV{}_{}_{}_{}_{}_{}_full.ckpt".format(str(cv), input_argue, model_name, subtype, total_epoch,
#                                                                 ' '.join([str(dim) for dim in dim_hiddens])))
#         print()