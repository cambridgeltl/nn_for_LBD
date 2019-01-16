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
from datetime import datetime

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

    if data_purpose != 'eval':
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

def train_and_eval(train_epochs, train_data, devel_data, test_data, test_embeddings_file_name,
    devel_filename, eval_filename, devel_unformed_filename, unformed_filename, positive_labels, combination_method, method,
    lbd_type, experiment_name):
    """Train and evaluate the model."""

    index_map, weights = wvd.load(test_embeddings_file_name)
    #Get positive labels
    positive_labels = positive_labels.split(',')

    print("reading training data...")
    train_file_name = train_data
    df_train = pd.read_table(train_file_name, dtype={'node1':str, 'node2':str})
    df_train = df_train.sample(frac=1)

    # remove NaN elements
    df_train = df_train.dropna(how='any', axis=0)

    #Get inputs
    train_x, labels, _ = get_input(df_train, weights, index_map, combination_method)
    train_loader = torch.utils.data.DataLoader(dataset=LinksDataset(train_x, labels), batch_size=batch_size, shuffle=True)

    pos_labels = [l_ for l_ in labels if l_ != 0]

    #Start Loading devel set
    print("reading devel data...")
    devel_file_name = devel_data
    df_devel = pd.read_table(devel_file_name, dtype={'node1':str, 'node2':str})
    df_devel = df_devel.sample(frac=1)
    # remove NaN elements
    df_devel = df_devel.dropna(how='any', axis=0)

    #Get inputs
    devel_x, dev_labels, _ = get_input(df_devel, weights, index_map, combination_method)
    devel_loader = torch.utils.data.DataLoader(dataset=LinksDataset(devel_x, dev_labels), batch_size=batch_size, shuffle=True)
    dev_pos_labels = [l_ for l_ in dev_labels if l_ != 0]
    #End Loading devel set

    #Start prepping dev data
    #Initial testing will select 1st 1000 egs in devel/test data instead of random selection. Read these lines
    with open(devel_filename) as inp_file:
        chosen_As = {}
        for ind, line in enumerate(csv.reader(inp_file, delimiter='\t')): #quoting=csv.QUOTE_NONE - If req to make data work, examine data
            a = line[0].replace(' ', '_').replace('-', '_')
            cs = line[1].split(';')
            chosen_As[a] = cs

            if ind >= 999:
                break
    print("In dev, there were {} chosen As".format(len(chosen_As)))

    a_dfs = {}
    added_links = {}
    negative_c_lst = {}
    for a in chosen_As.keys():
        a_dfs[a] = {'node1': [], 'node2': [], 'label':[]}
        added_links[a] = {}
        negative_c_lst[a] = []

    #Filter test for chosen As which formed (True positives)
    for ind, row in enumerate(df_devel.itertuples()):
        node1 = row.node1
        node2 = row.node2
        if node1 in chosen_As.keys():
            a_dfs[node1]['node1'].append(row.node1)
            a_dfs[node1]['node2'].append(row.node2)
            a_dfs[node1]['label'].append(row.label)
            added_links[node1]["{}::{}".format(row.node1,row.node2)] = 1
        if node2 in chosen_As.keys():
            a_dfs[node2]['node1'].append(row.node1)
            a_dfs[node2]['node2'].append(row.node2)
            a_dfs[node2]['label'].append(row.label)
            added_links[node2]["{}::{}".format(row.node1,row.node2)] = 1

    #Add unformed edges for the chosen As (True negatives)
    a_c_regex = r"'(.*?)'"
    unformed_edges = 0
    if 'json' in devel_unformed_filename:
        with open(devel_unformed_filename) as uf:
            data = ujson.loads(uf.read())
            for ac, b_lst in data.iteritems():
                ac_extract = re.findall(a_c_regex, ac)
                a = ac_extract[0].replace(' ', '_').replace('-', '_')
                c = ac_extract[1].replace(' ', '_').replace('-', '_')

                if a in negative_c_lst:
                    negative_c_lst[a].append(c)

                if a in chosen_As.keys():
                    for b in b_lst:
                        if "{}::{}".format(a,b) not in added_links[a]:
                            a_dfs[a]['node1'].append(a)
                            a_dfs[a]['node2'].append(b)
                            a_dfs[a]['label'].append(0) #label is irrelevant for test
                            added_links[a]["{}::{}".format(a, b)] = 1
                        if "{}::{}".format(b,c) not in added_links[a]:
                            a_dfs[a]['node1'].append(b)
                            a_dfs[a]['node2'].append(c)
                            a_dfs[a]['label'].append(0) #label is irrelevant for test

    for a, a_dict in a_dfs.iteritems():
        a_dfs[a] = pd.DataFrame(a_dict)
    #End prepping dev data

    print("\nBuilding model...")
    feature_dim = train_x[0].shape[0]
    model, criterion, optimizer = build_model(feature_dim)

    # Train the model
    print("\nTraining model...")
    total_step = len(train_loader)
    best_info = {'max_mmrr': 0.0}
    evaluate_every = 15
    for epoch in range(train_epochs):
        for i, (train_x, labels) in enumerate(train_loader):
            labels = labels.type(torch.LongTensor)
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
            ###Start Evaluate on the dev set
            print("Evaluating on devel...")
            map_output = ""

            #Save the current model
            torch.save(model, CUR_PATH)
            #Load the last saved best model
            lmodel = torch.load(CUR_PATH)
            # Test the model
            lmodel.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

            mrr_total = {'max': {}, 'min': {}, 'avg': {}}
            for agg in ['max', 'min', 'avg']:
                for acc in ['sum', 'max']:
                    mrr_total[agg][acc] = 0.0

            for a_ind, (a, df_test_a) in enumerate(a_dfs.iteritems()):
                test_x, test_labels, original_labels = get_input(df_test_a, weights, index_map, combination_method, data_purpose='eval')
                test_loader = torch.utils.data.DataLoader(dataset=LinksDataset(test_x, test_labels), batch_size=batch_size, shuffle=False)

                with torch.no_grad():
                    predictions = []
                    link_score_lst = {}
                    for test_links, test_labels in test_loader:
                        test_links = test_links.to(device)
                        outputs = lmodel(test_links)
                        predicted, _ = torch.max(outputs.data, 1)
                        predictions.extend([tensor.item() for tensor in predicted])

                        for link, pred in zip(original_labels, predictions):
                            link_score_lst[link] = pred

                c_scores = {}
                devel_cs = set(chosen_As[a] + negative_c_lst[a])
                for c in devel_cs:
                    c_scores[c] = {}

                    c_scores[c]['max'] = {}
                    c_scores[c]['min'] = {}
                    c_scores[c]['avg'] = {}

                    c_scores[c]['max']['sum'] = 0.0
                    c_scores[c]['min']['sum'] = 0.0
                    c_scores[c]['avg']['sum'] = 0.0
                    c_scores[c]['max']['max'] = 0.0
                    c_scores[c]['min']['max'] = 0.0
                    c_scores[c]['avg']['max'] = 0.0

                ab_lst = {}
                bc_lst = {}
                for index, row in df_test_a.iterrows():
                    if row['node1'] == a or row['node2'] == a:
                        ab_lst["{}::{}".format(row['node1'], row['node2'])] = 1
                    else:
                        bc_lst["{}::{}".format(row['node1'], row['node2'])] = 1
                for a_lnk in ab_lst:
                    #a-b links
                    ab = a_lnk.split('::')
                    b = ab[1] if ab[1] != a else ab[0]
                    ab_score = link_score_lst[a_lnk] if a_lnk in link_score_lst else link_score_lst["{}::{}".format(ab[1], ab[0])]
                    for b_c in bc_lst:
                        if b in b_c:
                            #b-c links
                            bc = b_c.split('::')
                            c = bc[1] if bc[1] != b else bc[0]
                            bc_score = link_score_lst[b_c] if b_c in link_score_lst else link_score_lst["{}::{}".format(bc[1], bc[0])]

                            #aggregators
                            max_agg = max(ab_score, bc_score)
                            min_agg = min(ab_score, bc_score)
                            avg_agg = sum([ab_score, bc_score])/2.0

                            #accumulators
                            if c in c_scores:
                                c_scores[c]['max']['sum'] += max_agg
                                c_scores[c]['min']['sum'] += min_agg
                                c_scores[c]['avg']['sum'] += avg_agg

                                c_scores[c]['max']['max'] = max_agg if max_agg > c_scores[c]['max']['max'] else c_scores[c]['max']['max']
                                c_scores[c]['min']['max'] = min_agg if min_agg > c_scores[c]['min']['max'] else c_scores[c]['min']['max']
                                c_scores[c]['avg']['max'] = avg_agg if avg_agg > c_scores[c]['avg']['max'] else c_scores[c]['avg']['max']

                y_true = [1.0 if c_ in chosen_As[a] else 0.0 for c_ in devel_cs] #gold
                for agg in ['max', 'min', 'avg']:
                    for acc in ['sum', 'max']:
                        scores = [c_scores[c_][agg][acc] for c_ in devel_cs]
                        score_label_lst = []
                        for score, label in zip(scores, y_true):
                            score_label_lst.append((score, label))

                        sorted_score_label_lst = sorted(score_label_lst, key=lambda x: x[0], reverse=True)
                        y_scores = [tup[0] for tup in sorted_score_label_lst] #predictions

                        true_inds = [ind for ind in range(len(y_true)) if y_true[ind] == 1.0]
                        true_scores = [y_scores[ind] for ind in true_inds]
                        sorted_scores = sorted(y_scores, reverse=True)

                        true_ranks = []
                        true_c_lst = chosen_As[a]
                        for tc, ts in zip(true_c_lst, true_scores):
                            true_ranks.append((sorted_scores.index(ts) + 1, ts, tc))

                        mrr = np.mean([1.0/tr_[0] for tr_ in true_ranks])
                        mrr_total[agg][acc] += mrr
                if a_ind % 500 == 0:
                    print("{} devel completed.".format(a_ind))

            for agg in ['max', 'min', 'avg']:
                for acc in ['sum', 'max']:
                    mean_mrr = mrr_total[agg][acc]/len(a_dfs.keys())

                    map_o = "MRR {}-{}: {}.".format(agg, acc, mean_mrr)
                    map_output = "{}\n{}\n\n{}".format(experiment_name, map_o, map_output)
                    print(map_o)

                    if mean_mrr > best_info['max_mmrr']:
                        print("Saving because {} > {}".format(mean_mrr, best_info['max_mmrr']))
                        torch.save(model, PATH)
                        best_info['experiment_name'] = experiment_name
                        best_info['max_mmrr'] = mean_mrr
                        best_info['loss_at_best'] = loss.item()
                        best_info['epoch'] = epoch + 1
                        best_info['config'] = "ACC: {}, AGG: {}".format(acc, agg)
            ###End Evaluate on the dev set

    print("\nTrain complete. Best info: {}".format(best_info))
    train_x = None
    train_loader = None
    print("\nTesting model...")
    index_map, weights = wvd.load(test_embeddings_file_name)

    print("Reading data...")
    test_file_name = test_data
    df_test = pd.read_table(test_file_name, dtype={'train_nodes':str})

    # remove NaN elements
    df_test = df_test.dropna(how='any', axis=0)

    #Initial testing will select 1st 100 egs in devel/test data instead of random selection. Read these lines
    with open(eval_filename) as inp_file:
        chosen_As = {}
        for ind, line in enumerate(csv.reader(inp_file, delimiter='\t')): #quoting=csv.QUOTE_NONE - If req to make data work, examine data
            a = line[0].replace(' ', '_').replace('-', '_')
            cs = line[1].split(';')
            chosen_As[a] = cs

            if ind >= 999:
                break
    print("There were {} chosen As".format(len(chosen_As)))

    a_dfs = {}
    added_links = {}
    negative_c_lst = {}
    for a in chosen_As.keys():
        a_dfs[a] = {'node1': [], 'node2': [], 'label':[]}
        added_links[a] = {}
        negative_c_lst[a] = []
    #Filter test for chosen As which formed (True positives)

    for ind, row in enumerate(df_test.itertuples()):
        node1 = row.node1
        node2 = row.node2
        if node1 in chosen_As.keys():
            a_dfs[node1]['node1'].append(row.node1)
            a_dfs[node1]['node2'].append(row.node2)
            a_dfs[node1]['label'].append(row.label)
            added_links[node1]["{}::{}".format(row.node1,row.node2)] = 1
        if node2 in chosen_As.keys():
            a_dfs[node2]['node1'].append(row.node1)
            a_dfs[node2]['node2'].append(row.node2)
            a_dfs[node2]['label'].append(row.label)
            added_links[node2]["{}::{}".format(row.node1,row.node2)] = 1

    #Add unformed edges for the chosen As (True negatives)
    #negative_c_lst = {}
    a_c_regex = r"'(.*?)'"
    unformed_edges = 0
    if 'json' in unformed_filename:
        with open(unformed_filename) as uf:
            data = ujson.loads(uf.read())
            for ac, b_lst in data.iteritems():
                ac_extract = re.findall(a_c_regex, ac)
                a = ac_extract[0].replace(' ', '_').replace('-', '_')
                c = ac_extract[1].replace(' ', '_').replace('-', '_')

                if a in negative_c_lst:
                    negative_c_lst[a].append(c)

                if a in chosen_As.keys():
                    for b in b_lst:
                        if "{}::{}".format(a,b) not in added_links[a]:
                            a_dfs[a]['node1'].append(a)
                            a_dfs[a]['node2'].append(b)
                            a_dfs[a]['label'].append(0) #label is irrelevant for test
                            added_links[a]["{}::{}".format(a, b)] = 1
                        if "{}::{}".format(b,c) not in added_links[a]:
                            a_dfs[a]['node1'].append(b)
                            a_dfs[a]['node2'].append(c)
                            a_dfs[a]['label'].append(0) #label is irrelevant for test

    for a, a_dict in a_dfs.iteritems():
        a_dfs[a] = pd.DataFrame(a_dict)
    #End prepping test data

    print("\nLoading best model...")
    #Load the saved best model
    print("Loading model with loss: {}".format(best_info['loss_at_best']))
    model = torch.load(PATH)
    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

    mrr_total = {'max': {}, 'min': {}, 'avg': {}}
    map_total = {'max': {}, 'min': {}, 'avg': {}}
    mr_total = {'max': {}, 'min': {}, 'avg': {}}
    h_at_r_total = {'max': {}, 'min': {}, 'avg': {}}
    a_ap = {'max': {}, 'min': {}, 'avg': {}}
    for agg in ['max', 'min', 'avg']:
        for acc in ['sum', 'max']:
            mrr_total[agg][acc] = 0.0
            map_total[agg][acc] = 0.0
            mr_total[agg][acc] = 0.0
            h_at_r_total[agg][acc] = 0.0
            a_ap[agg][acc] = ""

    for a_ind, (a, df_test_a) in enumerate(a_dfs.iteritems()):
        test_x, test_labels, original_labels = get_input(df_test_a, weights, index_map, combination_method, data_purpose='test')
        test_loader = torch.utils.data.DataLoader(dataset=LinksDataset(test_x, test_labels), batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            predictions = []
            link_score_lst = {}
            for test_links, test_labels in test_loader:
                test_links = test_links.to(device)
                outputs = model(test_links)
                predicted, _ = torch.max(outputs.data, 1)
                predictions.extend([tensor.item() for tensor in predicted])

                for link, pred in zip(original_labels, predictions):
                    link_score_lst[link] = pred

        c_scores = {}
        test_cs = set(chosen_As[a] + negative_c_lst[a])
        for c in test_cs:
            c_scores[c] = {}

            c_scores[c]['max'] = {}
            c_scores[c]['min'] = {}
            c_scores[c]['avg'] = {}

            c_scores[c]['max']['sum'] = 0.0
            c_scores[c]['min']['sum'] = 0.0
            c_scores[c]['avg']['sum'] = 0.0
            c_scores[c]['max']['max'] = 0.0
            c_scores[c]['min']['max'] = 0.0
            c_scores[c]['avg']['max'] = 0.0

        ab_lst = {}
        bc_lst = {}
        for index, row in df_test_a.iterrows():
            if row['node1'] == a or row['node2'] == a:
                ab_lst["{}::{}".format(row['node1'], row['node2'])] = 1
            else:
                bc_lst["{}::{}".format(row['node1'], row['node2'])] = 1
        for a_lnk in ab_lst:
            #a-b links
            ab = a_lnk.split('::')
            b = ab[1] if ab[1] != a else ab[0]
            ab_score = link_score_lst[a_lnk] if a_lnk in link_score_lst else link_score_lst["{}::{}".format(ab[1], ab[0])]
            for b_c in bc_lst:
                if b in b_c:
                    #b-c links
                    bc = b_c.split('::')
                    c = bc[1] if bc[1] != b else bc[0]
                    bc_score = link_score_lst[b_c] if b_c in link_score_lst else link_score_lst["{}::{}".format(bc[1], bc[0])]

                    #aggregators
                    max_agg = max(ab_score, bc_score)
                    min_agg = min(ab_score, bc_score)
                    avg_agg = sum([ab_score, bc_score])/2.0

                    #accumulators
                    if c in c_scores:
                        c_scores[c]['max']['sum'] += max_agg
                        c_scores[c]['min']['sum'] += min_agg
                        c_scores[c]['avg']['sum'] += avg_agg

                        c_scores[c]['max']['max'] = max_agg if max_agg > c_scores[c]['max']['max'] else c_scores[c]['max']['max']
                        c_scores[c]['min']['max'] = min_agg if min_agg > c_scores[c]['min']['max'] else c_scores[c]['min']['max']
                        c_scores[c]['avg']['max'] = avg_agg if avg_agg > c_scores[c]['avg']['max'] else c_scores[c]['avg']['max']

        y_true = [1.0 if c_ in chosen_As[a] else 0.0 for c_ in test_cs] #gold
        for agg in ['max', 'min', 'avg']:
            for acc in ['sum', 'max']:
                scores = [c_scores[c_][agg][acc] for c_ in test_cs]
                score_label_lst = []
                for score, label in zip(scores, y_true):
                    score_label_lst.append((score, label))

                sorted_score_label_lst = sorted(score_label_lst, key=lambda x: x[0], reverse=True)
                y_scores = [tup[0] for tup in sorted_score_label_lst] #predictions

                true_inds = [ind for ind in range(len(y_true)) if y_true[ind] == 1.0]
                true_scores = [y_scores[ind] for ind in true_inds]
                sorted_scores = sorted(y_scores, reverse=True)

                true_ranks = []
                true_c_lst = chosen_As[a]
                for tc, ts in zip(true_c_lst, true_scores):
                    true_ranks.append((sorted_scores.index(ts) + 1, ts, tc))

                ap = average_precision_score(y_true, y_scores, average='micro')
                mr = np.mean([tr_[0] for tr_ in true_ranks])
                mrr = np.mean([1.0/tr_[0] for tr_ in true_ranks])
                hits_at_R = len([tup_[0] for tup_ in sorted(true_ranks, key= lambda x: x[0]) if tup_[0] <= len(true_ranks)])/float(len(true_ranks))

                map_total[agg][acc] += ap
                mr_total[agg][acc] += mr
                mrr_total[agg][acc] += mrr
                h_at_r_total[agg][acc] += hits_at_R

                tp = len([x for x in y_true if x > 0.0])
                a_ap[agg][acc] += "{}. A: {}, AP: {}. MR: {}. MRR: {}. Hits at R: {}. TP: {}/{}. \nRanks: {}\n\n".format(a_ind + 1, a, ap, mr, mrr, hits_at_R, tp, len(predictions), sorted(true_ranks, key= lambda x: x[0]))
        if a_ind % 500 == 0:
            print("{} test completed at {}.".format(a_ind, datetime.now()))

    for agg in ['max', 'min', 'avg']:
        for acc in ['sum', 'max']:
            mean_map = map_total[agg][acc]/len(a_dfs.keys())
            mean_mr = mr_total[agg][acc]/len(a_dfs.keys())
            mean_mrr = mrr_total[agg][acc]/len(a_dfs.keys())
            mean_hits_at_r = h_at_r_total[agg][acc]/len(a_dfs.keys())

            map_o = "AGG: {}\tACC: {}\nMean MAP was: {}. Mean mean-rank was: {}. MRR: {}. Mean Hits at R: {}".format(agg, acc, mean_map, mean_mr, mean_mrr, mean_hits_at_r)
            map_output = "{}\n{}\n{}\n\n{}".format(experiment_name, best_info, map_o, a_ap[agg][acc])
            print(map_o)

            with open("Eval-Scores-{}-{}-{}.txt".format(experiment_name, agg, acc), 'w') as fil:
                fil.write(map_output)
FLAGS = None
def main(_):
    train_and_eval(FLAGS.train_epochs, FLAGS.train_data, FLAGS.devel_data, FLAGS.test_data,
        FLAGS.test_embeddings_data, FLAGS.devel_filename, FLAGS.eval_filename, FLAGS.devel_unformed_filename, FLAGS.unformed_filename, FLAGS.positive_labels,
        FLAGS.combination_method, FLAGS.method, FLAGS.lbd_type, FLAGS.experiment_name)


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
      "--devel_data",
      type=str,
      default="",
      help="Path to the devel data."
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
       "--devel_filename",
       type=str,
       default="",
       help="Path to file with development data."
   )
  parser.add_argument(
      "--eval_filename",
      type=str,
      default="",
      help="Path to file with evalution data."
  )
  parser.add_argument(
       "--devel_unformed_filename",
       type=str,
       default="",
       help="Path to .json file with devel unformed edges."
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
      "--method",
      type=str,
      default="",
      help="Method used to create embeddings."
  )
  parser.add_argument(
      "--lbd_type",
      type=str,
      default="",
      help="The type of discovery for LBD."
  )
  FLAGS, unparsed = parser.parse_known_args()
  main(sys.argv[0])
