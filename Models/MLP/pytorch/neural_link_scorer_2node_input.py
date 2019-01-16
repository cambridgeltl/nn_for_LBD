import argparse
import sys
import random
import csv
import ujson
import re
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import wordvecdata as wvd
from sklearn.metrics import average_precision_score

import pdb

COLUMNS = ["node1", "node2"]
LABEL_COLUMN = "label"

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_classes = 2
batch_size = 100
learning_rate = 0.0001

PATH = "saved_models/best_model.pth"
CUR_PATH = "saved_models/cur_model.pth"

#Torch Dataset class to hold data
class LinksDataset(Dataset):

    def __init__(self, features_array, labels_array, transform=torch.from_numpy):
        """
        Args:
            features_array:
            labels_array:
            transform:
        """
        self.link_features = features_array
        self.labels = labels_array
        self.transform = transform

    def __len__(self):
        return len(self.link_features)

    def __getitem__(self, idx):
        link_features = self.link_features[idx]
        label = self.labels[idx]

        if self.transform:
            link_features = self.transform(link_features)

        return link_features, label

# model
class MLP(nn.Module):
    def __init__(self, num_classes, input_dim=300):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 100)
        self.layer1_nonlinearity = nn.ReLU()
        self.output_layer = nn.Linear(100, 2)
        self.softplus = nn.Softplus()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer1_nonlinearity(out)
        out = out.reshape(out.size(0), -1)
        out = self.output_layer(out)
        out = self.softplus(out)
        return out

def get_input(df, embeddings,index_map, combination_method='hadamard', data_purpose='train'):
    """Build model inputs."""

    # Converts the label column into a constant Tensor.
    label_values = [0 if val_ == 0.0 else 1 for val_ in df[LABEL_COLUMN].values]
    original_labels = np.array(label_values)

    feature_cols = {}
    column_tensors = []
    col_keys = []
    for i in COLUMNS:
        if i == 'node1':
            col_weight = 10.0
        elif i == 'node2':
            col_weight = 10.0

        words = [value for value in df[i].values]
        col_keys.append(words)
        ids = [index_map[word] for word in words]
        column_tensors.append([np.multiply(np.array(embeddings[id_]), col_weight) for id_ in ids])

    keys = []
    for entity1, entity2 in zip(col_keys[0], col_keys[1]):
      keys.append("%s::%s" % (entity1, entity2))

    assert(combination_method in ['hadamard','average', 'weighted_l1', 'weighted_l2', 'concatenate']), "Invalid combination Method %s" % combination_method
    print("Combining with {}.".format(combination_method))

    features = column_tensors[0]
    for i in range(1, len(column_tensors)):
      if combination_method == 'hadamard':
          features = np.multiply(features, column_tensors[i])
      elif combination_method == 'average':
          features = np.mean(np.array([ features, column_tensors[i] ]), axis=0)
      elif combination_method == 'weighted_l1':
          features = np.absolute(np.subtract(features, column_tensors[i]))
      elif combination_method == 'weighted_l2':
          features = np.square(np.absolute(np.subtract(features, column_tensors[i])))
      elif combination_method == 'concatenate':
          features = np.concatenate([features, column_tensors[i]], 1)

    return features, np.array(label_values), keys

def build_model(input_dim):
    """Build model."""

    model = MLP(num_classes, input_dim).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, criterion, optimizer

def train_and_eval(train_epochs, train_data, test_data, train_embeddings_file_name, test_embeddings_file_name, eval_filename,
    unformed_filename, positive_labels, combination_method, method, c_lst, lbd_type, experiment_name, a, c, gold_b):
    """Train and evaluate the model."""

    index_map, weights = wvd.load(test_embeddings_file_name)
    #Get positive labels
    positive_labels = positive_labels.split(',')

    print("reading training data...")
    train_file_name = train_data
    df_train = pd.read_table(train_file_name, dtype={'train_nodes':str})
    df_train = df_train.sample(frac=1)

    # remove NaN elements
    df_train = df_train.dropna(how='any', axis=0)

    #Get inputs
    train_x, labels, _ = get_input(df_train, weights, index_map, combination_method)
    train_loader = torch.utils.data.DataLoader(dataset=LinksDataset(train_x, labels), batch_size=batch_size, shuffle=True)

    pos_labels = [l_ for l_ in labels if l_ != 0]

    #Start loading evaluation data (same as test data for cancer cases)
    print("reading eval data...")
    test_file_name = test_data
    df_test = pd.read_table(test_file_name, dtype={'train_nodes':str})

    # remove NaN elements
    df_test = df_test.dropna(how='any', axis=0)

    c_dict = {}
    if c_lst:
        c_file = open(c_lst, 'r')
        c_ = c_file.readline().strip(' \n\t')
        while c_:
            c_dict[c_] = 1
            c_ = c_file.readline().strip(' \n\t')

    test_x, test_labels, test_original_x = get_input(df_test, weights, index_map, combination_method, data_purpose='test')
    test_loader = torch.utils.data.DataLoader(dataset=LinksDataset(test_x, test_labels), batch_size=batch_size, shuffle=False)
    #End loading evaluation data

    print("\nBuilding model...")
    feature_dim = train_x[0].shape[0]
    model, criterion, optimizer = build_model(feature_dim)

    # Train the model
    print("\nTraining model...")
    total_step = len(train_loader)
    best_info = {'best_rank':1000000000}
    evaluate_every = 5
    for epoch in range(train_epochs):
        for i, (train_x, labels) in enumerate(train_loader):
            labels = labels.type(torch.LongTensor)
            #labels = labels.view(-1, 1)
            links = train_x.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(links)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 500 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, train_epochs, i+1, total_step, loss.item()))

        if (epoch + 1) % evaluate_every == 0:
            #Save the current model
            torch.save(model, CUR_PATH)
            #Load the last saved best model
            lmodel = torch.load(CUR_PATH)
            lmodel.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

            # Test the model
            with torch.no_grad():
                predictions = []
                for test_links, _ in test_loader:
                    test_links = test_links.to(device)
                    outputs = lmodel(test_links)
                    predicted, _ = torch.max(outputs.data, 1)
                    predictions.extend([tensor.item() for tensor in predicted])

            if lbd_type == 'closed_discovery':
                ranks, ties, output = do_case_cd_evaluations(a, c, gold_b, [p for p in predictions], [x for x in test_original_x], experiment_name)
                for ind, tie in enumerate(ties):
                    if tie > 10:
                        ranks.pop(ind)
                if len(ranks) > 0:
                    best_rank = min(ranks)
                    if best_rank < best_info['best_rank']:
                        print("Saving because {} < {}".format(best_rank, best_info['best_rank']))
                        torch.save(model, PATH)
                        best_info['case_name'] = experiment_name
                        best_info['best_rank'] = best_rank
                        best_info['loss_at_best'] = loss.item()
                        best_info['epoch'] = epoch + 1
                        best_info['output'] = output

                        fil = open("{}.txt".format(experiment_name), 'w')
                        fil.write("{}\n\n{}".format(str(best_info), best_info['output']))
                        fil.close()
            elif lbd_type == 'open_discovery':
                ranks, ties, output = do_case_od_evaluations(a, gold_b, c, c_dict, [p for p in predictions], [x for x in test_original_x], experiment_name)
                for ind, tie in enumerate(ties):
                    if tie > 10:
                        ranks.pop(ind)
                if len(ranks) > 0:
                    best_rank = min(ranks)
                    if best_rank < best_info['best_rank']:
                        print("Saving because {} < {}".format(best_rank, best_info['best_rank']))
                        torch.save(model, PATH)
                        best_info['case_name'] = experiment_name
                        best_info['best_rank'] = best_rank
                        best_info['loss_at_best'] = loss.item()
                        best_info['epoch'] = epoch + 1
                        best_info['output'] = output

                        fil = open("{}.txt".format(experiment_name), 'w')
                        fil.write("{}\n\n{}".format(str(best_info), best_info['output']))
                        fil.close()
            else:
                print("ERROR: Invalid lbd_type: {}".format(lbd_type))

def do_case_cd_evaluations(a, c, gold_b, predictions, test_x, case_name):
    scores_dict = {}
    for x_link, score in zip(test_x, predictions):
        entity1 = x_link.split('::')[0]
        entity2 = x_link.split('::')[1]
        if entity1 in [a, c]:
            b = entity2
        else:
            b = entity1
        if b in scores_dict:
            scores_dict[b].append(score)
        else:
            scores_dict[b] = [score]

    gold_scores = scores_dict[gold_b]
    max_rank = 1
    avg_rank = 1
    min_rank = 1
    max_ties = 0
    avg_ties = 0
    min_ties = 0
    gold_max_score = max(gold_scores)
    gold_min_score = min(gold_scores)
    gold_avg_score = sum(gold_scores)/float(len(gold_scores))
    for b, scores in scores_dict.iteritems():
        if b == gold_b:
            continue

        max_score = max(scores)
        min_score = min(scores)
        avg_score = sum(scores)/float(len(scores))

        if max_score > gold_max_score:
            max_rank += 1
        elif max_score == gold_max_score:
            max_ties += 1
        if min_score > gold_min_score:
            min_rank += 1
        elif min_score == gold_min_score:
            min_ties += 1
        if avg_score > gold_avg_score:
            avg_rank += 1
        elif avg_score == gold_avg_score:
            avg_ties += 1

    output = "Of {} Bs, gold B {} (scores: {}) ranks: Max: {} [ties: {}] ({}), Min: {} [ties: {}] ({}), Avg: {} [ties: {}] ({}).".format(len(scores_dict), gold_b, gold_scores, max_rank, max_ties, gold_max_score, min_rank, min_ties, gold_min_score, avg_rank, avg_ties, gold_avg_score)
    print(output)
    ranks = [max_rank, avg_rank, min_rank]
    ties = [max_ties, avg_ties, min_ties]
    return ranks, ties, output

def do_case_od_evaluations(a, gold_b, gold_c, all_cs, predictions, test_x, case_name):
    scores_dict = {}
    entities = []
    for x_link, score in zip(test_x, predictions):

        entity1 = x_link.split('::')[0]
        entity2 = x_link.split('::')[1]
        entities.append(entity1)
        entities.append(entity2)

        if entity1 in scores_dict:
            scores_dict[entity1][entity2] = score
        else:
            scores_dict[entity1] = {entity2: score}

        if entity2 in scores_dict:
            scores_dict[entity2][entity1] = score
        else:
            scores_dict[entity2] = {entity1: score}

    entities = list(set(entities))

    max_rank = 1
    avg_rank = 1
    min_rank = 1
    max_c = {}
    avg_c = {}
    min_c = {}
    for c in all_cs:
        max_c[c] = []
        avg_c[c] = []
        min_c[c] = []
        c_scores= scores_dict[c]
        for b in c_scores.keys():
            b_scores = scores_dict[b]
            ab_score = b_scores[a]
            bc_score = c_scores[b]
            scores = [ab_score, bc_score]
            max_score = max(scores)
            min_score = min(scores)
            avg_score = sum(scores)/float(len(scores))

            max_c[c].append(max_score)
            avg_c[c].append(avg_score)
            min_c[c].append(min_score)

    max_max_rank = 1
    max_sum_rank = 1
    avg_max_rank = 1
    avg_sum_rank = 1
    min_max_rank = 1
    min_sum_rank = 1

    max_max_rank_ties = 0
    max_sum_rank_ties = 0
    avg_max_rank_ties = 0
    avg_sum_rank_ties = 0
    min_max_rank_ties = 0
    min_sum_rank_ties = 0
    #max accumulator for max aggregator
    gold_max_max = max(max_c[gold_c])
    for maxc, maxc_scores in max_c.iteritems():
        cur_max = max(maxc_scores)
        if cur_max > gold_max_max:
            max_max_rank += 1
        elif cur_max == gold_max_max and maxc != gold_c:
            max_max_rank_ties += 1

    #sum accumulator for max aggregator
    gold_sum_max = sum(max_c[gold_c])
    for maxc, maxc_scores in max_c.iteritems():
        cur_sum = sum(maxc_scores)
        if cur_sum > gold_sum_max:
            max_sum_rank += 1
        elif cur_sum == gold_sum_max and maxc != gold_c:
            max_sum_rank_ties += 1

    #max accumulator for avg aggregator
    gold_max_avg = max(avg_c[gold_c])
    for avgc, avgc_scores in avg_c.iteritems():
        cur_max = max(avgc_scores)
        if cur_max > gold_max_avg:
            avg_max_rank += 1
        elif cur_max == gold_max_avg and avgc != gold_c:
            avg_max_rank_ties += 1

    #sum accumulator for avg aggregator
    gold_sum_avg = sum(avg_c[gold_c])
    for avgc, avgc_scores in avg_c.iteritems():
        cur_sum = sum(avgc_scores)
        if cur_sum > gold_sum_avg:
            avg_sum_rank += 1
        elif cur_sum == gold_sum_avg and avgc != gold_c:
            avg_sum_rank_ties += 1

    #max accumulator for min aggregator
    gold_max_min = max(min_c[gold_c])
    for minc, minc_scores in min_c.iteritems():
        cur_max = max(minc_scores)
        if cur_max > gold_max_min:
            min_max_rank += 1
        elif cur_max == gold_max_min and minc != gold_c:
            min_max_rank_ties += 1

    #sum accumulator for min aggregator
    gold_sum_min = sum(min_c[gold_c])
    for minc, minc_scores in min_c.iteritems():
        cur_sum = sum(minc_scores)
        if cur_sum > gold_sum_min:
            min_sum_rank += 1
        elif cur_sum == gold_sum_min and minc != gold_c:
            min_sum_rank_ties += 1

    output = "Of {} ABCs, gold ABC {}:{}:{}\n Max accumulator: \nRanks: Max: {}[ties: {}]({}), Min: {}[ties: {}]({}), Avg: {}[ties: {}]({}).\n \
        Sum accumulator:\nRanks: Max: {}[ties: {}]({}), Min: {}[ties: {}]({}), Avg: {}[ties: {}]({})".format(
        len(all_cs), a, gold_b, gold_c, max_max_rank, max_max_rank_ties, gold_max_max, min_max_rank, min_max_rank_ties, gold_max_min, avg_max_rank, avg_max_rank_ties, gold_max_avg, max_sum_rank, max_sum_rank_ties, gold_sum_max, min_sum_rank, min_sum_rank_ties, gold_sum_min, avg_sum_rank, avg_sum_rank_ties, gold_sum_avg)
    print(output)

    ranks = [max_max_rank, min_max_rank, avg_max_rank, max_sum_rank, min_sum_rank, avg_sum_rank]
    ties = [max_max_rank_ties, min_max_rank_ties, avg_max_rank_ties, max_sum_rank_ties, min_sum_rank_ties, avg_sum_rank_ties]
    return ranks, ties, output

FLAGS = None
def main(_):
    train_and_eval(FLAGS.train_epochs, FLAGS.train_data, FLAGS.test_data, FLAGS.train_embeddings_data,
        FLAGS.test_embeddings_data, FLAGS.eval_filename, FLAGS.unformed_filename, FLAGS.positive_labels,
        FLAGS.combination_method, FLAGS.method, FLAGS.c_list, FLAGS.lbd_type, FLAGS.experiment_name,
        FLAGS.a_node, FLAGS.c_node, FLAGS.goldb_node)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")

  parser.add_argument(
      "--train_epochs",
      type=int,
      default=10,
      help="Number of training epochs."
  )
  parser.add_argument(
      "--experiment_name",
      type=str,
      default="",
      help="Name of this experiment instance."
  )
  parser.add_argument(
      "--train_data",
      type=str,
      default="",
      help="Path to training examples."
  )
  parser.add_argument(
      "--test_data",
      type=str,
      default="",
      help="Path to the test data."
  )
  parser.add_argument(
      "--train_embeddings_data",
      type=str,
      default="",
      help="Path to the pre-trained embeddings file for training."
  )
  parser.add_argument(
      "--test_embeddings_data",
      type=str,
      default="",
      help="Path to the pre-trained embeddings file for testing."
  )
  parser.add_argument(
      "--eval_filename",
      type=str,
      default="",
      help="Path to file with evalution data."
  )
  parser.add_argument(
      "--unformed_filename",
      type=str,
      default="",
      help="Path to .json file with unformed edges."
  )
  parser.add_argument(
      "--positive_labels",
      type=str,
      default="I-LINK",
      help="Label of positive classes in data, separated by comma."
  )
  parser.add_argument(
      "--combination_method",
      type=str,
      default="concatenate",
      help="How the features should be combined by the model."
  )
  parser.add_argument(
      "--graph_bipartite",
      type=str,
      default=False,
      help="Process graph as bipartitie or not for Common Neighbours."
  )
  parser.add_argument(
      "--method",
      type=str,
      default="",
      help="Method used to create embeddings."
  )
  parser.add_argument(
      "--c_list",
      type=str,
      default="",
      help="Path to the list of Cs for open discovery."
  )
  parser.add_argument(
      "--b_list",
      type=str,
      default="",
      help="Path to the list of Bs for closed discovery."
  )
  parser.add_argument(
      "--lbd_type",
      type=str,
      default="",
      help="The type of discovery for LBD."
  )
  parser.add_argument(
        "--a_node",
        type=str,
        default="",
        help="name of 'A' node."
  )
  parser.add_argument(
        "--c_node",
        type=str,
        default="",
        help="name of 'C' node."
  )
  parser.add_argument(
        "--goldb_node",
        type=str,
        default="",
        help="name of 'B' node deemed correct for case."
  )
  parser.add_argument(
        "--case_name",
        type=str,
        default="",
        help="name of closed discovery case."
  )
  FLAGS, unparsed = parser.parse_known_args()
  main(sys.argv[0])
