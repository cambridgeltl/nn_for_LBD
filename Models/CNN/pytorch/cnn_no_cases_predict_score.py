import argparse
import sys
import random
import csv
import ujson
import re
import pandas as pd
import numpy as np

from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import wordvecdata as wvd
from sklearn.metrics import average_precision_score

COLUMNS = ["node1", "node2", "node3"]
LABEL_COLUMN = "label"

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
PATH = "saved_models/best_biogrid_model.pth"
CUR_PATH = "saved_models/cur_biogrid_model.pth"
#TODO pass these from command line
num_classes = 2
batch_size = 100
learning_rate = 0.0001
frame_link_amt = 50
conv_height = 7

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

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes, conv_height, conv_width=300):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(conv_height, conv_width), stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1))
            )
        self.fc = nn.Linear(2816, 1)
        self.softmax = nn.Softplus()

    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.softmax(out)
        return out

def get_input(df, embeddings,index_map, combination_method='hadamard', data_purpose='train'):
    """Build model inputs."""

    dim_size = embeddings.shape[1]
    features = []
    # Converts the label column into a constant Tensor.
    label_values = df[LABEL_COLUMN].values

    assert(combination_method in ['hadamard','average', 'weighted_l1', 'weighted_l2', 'concatenate']), "Invalid combination Method %s" % combination_method
    padding = np.array([0.0] * dim_size, dtype='float32')

    print("Combining with {}.".format(combination_method))

    total_real_links = 0
    max_real_links = 0
    min_real_links = 10000
    total_rows = 0
    c_lst = [] #ordered list of the Cs (for evaluation)
    c_lst_dict = {} #dict to support fast searching of above. TODO: investigate ordered dicts

    col_keys = []
    for ind, row in enumerate(df.itertuples()):
        input_str = row.train_nodes
        conv_rows = input_str.split('-')

        c = conv_rows[0].split('::')[2]
        if c not in c_lst_dict:
            c_lst.append(c)
            c_lst_dict[c] = 1

        conv_rows_cnt = len(conv_rows)
        total_real_links += conv_rows_cnt
        if conv_rows_cnt > max_real_links:
            max_real_links = conv_rows_cnt
        if conv_rows_cnt < min_real_links:
            min_real_links = conv_rows_cnt

        if conv_rows_cnt < frame_link_amt:
            needed = frame_link_amt - len(conv_rows)
            for i in range(needed):
                if i % 2 == 0:
                    conv_rows.append('PAD::PAD::PAD')
                else:
                    conv_rows = ['PAD::PAD::PAD'] + conv_rows
        frame = {'node1':[], 'node2':[], 'node3':[],}
        for crow in conv_rows:
            node1, node2, node3 = crow.split('::')
            frame['node1'].append(node1)
            frame['node2'].append(node2)
            frame['node3'].append(node3)
        instance_df = pd.DataFrame(frame)

        feature_cols = {}
        column_tensors = []
        for i in COLUMNS:
            if i == 'node1':
                col_weight = 10.0
            elif i == 'node2':
                col_weight = 10.0
            elif i == 'node3':
                col_weight = 10.0

            words = [value for value in instance_df[i].values]
            col_keys.append([w_ for w_ in words if value != 'PAD'])
            ids = [index_map[word] if word != 'PAD' else -1 for word in words]
            column_tensors.append([np.multiply(np.array(embeddings[id_]), col_weight) if id_ != -1 else padding for id_ in ids])

        instance_features = np.array(column_tensors[0])
        no_output = ['map']
        for i in range(1, len(column_tensors)):
            if combination_method == 'hadamard':
                instance_features = np.multiply(instance_features, column_tensors[i])
            elif combination_method == 'average':
                instance_features = np.mean(np.array([ instance_features, column_tensors[i] ]), axis=0)
            elif combination_method == 'weighted_l1':
                instance_features = np.absolute(np.subtract(instance_features, column_tensors[i]))
            elif combination_method == 'weighted_l2':
                instance_features = np.square(np.absolute(np.subtract(instance_features, column_tensors[i])))
            elif combination_method == 'concatenate':
                instance_features = np.concatenate([instance_features, column_tensors[i]], 1)

        #Combine all feature vectors into the conv window size instead of simply truncating
        updated_instance_features = instance_features[:frame_link_amt]
        start_row = frame_link_amt
        end_row = frame_link_amt * 2
        while end_row <= len(conv_rows):
            updated_instance_features = np.add(updated_instance_features, instance_features[start_row:end_row])
            start_row = end_row
            end_row += frame_link_amt

        #sum the remaining rows
        #TODO: can perhaps be done more elegantly with modulus of len(conv_rows) and frame_link_amt
        if start_row < len(conv_rows):
            padding_needed = end_row - len(conv_rows)
            padded_instance_features = np.concatenate([instance_features[start_row:], np.ones((padding_needed, instance_features.shape[1]),dtype='float32')])
            updated_instance_features = np.add(updated_instance_features, padded_instance_features)

        features.append(np.expand_dims(updated_instance_features, axis=0))
        total_rows = ind

    print("\nReal links in conv window stats: Range from {}-{} with a mean of {}.".format(min_real_links, max_real_links,
                                                            total_real_links/max(total_rows, 1)))

    if data_purpose == 'test':
        return features, np.array([1 if val > 0 else 0 for val in label_values if val != -1]), label_values, c_lst

    return features, np.array([val * 100 for val in label_values if val != -1])

def label_func(x, positive_labels):
    for ind, label in enumerate(positive_labels, 1):
        if label in x:
            return ind
    return 0

def build_model(conv_width):
    """Build model."""

    model = ConvNet(num_classes, conv_height, conv_width).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, criterion, optimizer

def train_and_eval(train_epochs, train_data, devel_data, test_data, train_embeddings_file_name, test_embeddings_file_name,
    devel_filename, eval_filename, devel_unformed_filename, unformed_filename, positive_labels, combination_method, method,
    c_lst, lbd_type, experiment_name, all_data, neighbours_data, cutoff_year):
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
    train_x, labels = get_input(df_train, weights, index_map, combination_method)
    train_loader = torch.utils.data.DataLoader(dataset=LinksDataset(train_x, labels), batch_size=batch_size, shuffle=True)

    pos_labels = [l_ for l_ in labels if l_ != 0]

    #Start Loading devel set
    print("reading devel data...")
    devel_file_name = devel_data
    df_devel = pd.read_table(devel_file_name, dtype={'train_nodes':str})
    df_devel = df_devel.sample(frac=1)
    # remove NaN elements
    df_devel = df_devel.dropna(how='any', axis=0)
    #Get inputs
    devel_x, dev_labels = get_input(df_devel, weights, index_map, combination_method)
    devel_loader = torch.utils.data.DataLoader(dataset=LinksDataset(devel_x, dev_labels), batch_size=batch_size, shuffle=True)
    dev_pos_labels = [l_ for l_ in dev_labels if l_ != 0]
    #End Loading devel set
    #Start prepping dev data
    #Initial testing will select 1st 100 egs in devel/test data instead of random selection. Read these lines
    with open(devel_filename) as inp_file:
        chosen_As = {}
        for ind, line in enumerate(csv.reader(inp_file, delimiter='\t')): #quoting=csv.QUOTE_NONE - If req to make data work, examine data
            a = line[0].replace(' ', '_').replace('-', '_')
            cs = line[1].split(';')
            chosen_As[a] = 1

            if ind >= 999:
                break
    print("In dev, there were {} chosen As".format(len(chosen_As)))

    a_dfs = {}
    for a in chosen_As.keys():
        a_dfs[a] = {'train_nodes': [], 'label':[]}
    #Filter test for chosen As which formed (True positives)
    formed_c = {}
    for ind, row in enumerate(df_devel.itertuples()):
        input_str = row.train_nodes
        conv_rows = input_str.split('-')
        link_lst = conv_rows[0].split('::')
        a = link_lst[0]
        c = link_lst[2]
        if a in chosen_As.keys():
            a_dfs[a]['train_nodes'].append(row.train_nodes)
            a_dfs[a]['label'].append(row.label)
            formed_c["{}::{}".format(a,c)] = 1

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

                if a in chosen_As.keys():
                    if "{}::{}".format(a,c) not in formed_c:
                        conv_frame = ""
                        for ind, b in enumerate(b_lst):
                            conv_frame += "{}::{}::{}".format(a, b.replace(' ', '_').replace('-', '_'), c)
                            if ind < len(b_lst) - 1:
                                conv_frame += "-"
                        a_dfs[a]['train_nodes'].append(conv_frame)
                        a_dfs[a]['label'].append(0)
                        unformed_edges += 1

    print("There were {} unformed edges added.".format(unformed_edges))
    for a, a_dict in a_dfs.iteritems():
        a_dfs[a] = pd.DataFrame(a_dict)
    #End prepping dev data

    print("\nBuilding model...")
    feature_dim = train_x[0].shape[2]
    model, criterion, optimizer = build_model(feature_dim)

    # Train the model
    print("\nTraining model...")
    total_step = len(train_loader)
    best_info = {'max_mmrr':0}
    evaluate_every = 5
    for epoch in range(train_epochs):
        for i, (train_x, labels) in enumerate(train_loader):
            labels = labels.type(torch.FloatTensor)
            labels = labels.view(-1, 1)

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
            mrr_total = 0.0

            #Save the current model
            torch.save(model, CUR_PATH)
            #Load the last saved best model
            lmodel = torch.load(CUR_PATH)
            # Test the model
            lmodel.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

            for a_ind, (a, df_test_a) in enumerate(a_dfs.iteritems()):
                test_x, test_labels, original_labels, c_lst = get_input(df_test_a, weights, index_map, combination_method, data_purpose='test')
                test_loader = torch.utils.data.DataLoader(dataset=LinksDataset(test_x, test_labels), batch_size=batch_size, shuffle=False)

                with torch.no_grad():
                    predictions = []
                    pred_lab_lst = []
                    for test_links, test_labels in test_loader:
                        test_links = test_links.to(device)
                        outputs = lmodel(test_links)
                        predicted, _ = torch.max(outputs.data, 1)
                        predictions.extend([tensor.item() for tensor in predicted])

                        for pred, lab in zip(predictions, [tensor.item() for tensor in test_labels]):
                            pred_lab_lst.append((pred, lab))

                sorted_pred_lab_lst = sorted(pred_lab_lst, key=lambda x: x[0], reverse=True)

                y_true = [tup[1] for tup in sorted_pred_lab_lst] #gold
                y_scores = [tup[0] for tup in sorted_pred_lab_lst] #predictions

                true_inds = [ind for ind in range(len(y_true)) if y_true[ind] == 1]
                true_scores = [y_scores[ind] for ind in true_inds]
                sorted_scores = sorted(y_scores, reverse=True)
                true_ranks = []

                for tc, ts in zip(c_lst, true_scores):
                    true_ranks.append((sorted_scores.index(ts) + 1, ts, tc))

                mrr = np.mean([1.0/tr_[0] for tr_ in true_ranks])
                mrr_total += mrr

                tp = len([x for x in y_true if x > 0.0])
                if a_ind % 500 == 0:
                    print("{} devel completed.".format(a_ind))
            mean_mrr = mrr_total/len(a_dfs.keys())

            map_o = "MRR: {}.".format(mean_mrr)
            map_output = "{}\n{}\n\n{}".format(experiment_name, map_o, map_output)
            print(map_o)

            if mean_mrr > best_info['max_mmrr']:
                print("Saving because {} > {}".format(mean_mrr, best_info['max_mmrr']))
                torch.save(model, PATH)
                best_info['experiment_name'] = experiment_name
                best_info['max_mmrr'] = mean_mrr
                best_info['loss_at_best'] = loss.item()
                best_info['epoch'] = epoch + 1
            ###End Evaluate on the dev set

    print("\nTrain complete. Best info: {}".format(best_info))
    train_x = None
    train_loader = None
    print("\nTesting model...")
    index_map, weights = wvd.load(test_embeddings_file_name)

    print("reading data...")
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
            chosen_As[a] = 1

            if ind >= 999:
                break
    print("There were {} chosen As".format(len(chosen_As)))

    a_dfs = {}
    for a in chosen_As.keys():
        a_dfs[a] = {'train_nodes': [], 'label':[]}
    #Filter test for chosen As which formed (True positives)
    formed_c = {}
    for ind, row in enumerate(df_test.itertuples()):
        input_str = row.train_nodes
        conv_rows = input_str.split('-')
        link_lst = conv_rows[0].split('::')
        a = link_lst[0]
        c = link_lst[2]
        if a in chosen_As.keys():
            a_dfs[a]['train_nodes'].append(row.train_nodes)
            a_dfs[a]['label'].append(row.label)
            formed_c["{}::{}".format(a,c)] = 1

    #Add unformed edges for the chosen As (True negatives)
    a_c_regex = r"'(.*?)'"
    unformed_edges = 0
    if 'json' in unformed_filename:
        with open(unformed_filename) as uf:
            data = ujson.loads(uf.read())
            for ac, b_lst in data.iteritems():
                ac_extract = re.findall(a_c_regex, ac)
                a = ac_extract[0].replace(' ', '_').replace('-', '_')
                c = ac_extract[1].replace(' ', '_').replace('-', '_')

                if a in chosen_As.keys():
                    if "{}::{}".format(a,c) not in formed_c:
                        conv_frame = ""
                        for ind, b in enumerate(b_lst):
                            conv_frame += "{}::{}::{}".format(a, b.replace(' ', '_').replace('-', '_'), c)
                            if ind < len(b_lst) - 1:
                                conv_frame += "-"
                        a_dfs[a]['train_nodes'].append(conv_frame)
                        a_dfs[a]['label'].append(0)
                        unformed_edges += 1

    print("There were {} unformed edges added.".format(unformed_edges))
    for a, a_dict in a_dfs.iteritems():
        a_dfs[a] = pd.DataFrame(a_dict)

    map_output = ""
    map_total  = 0.0
    mr_total = 0.0
    mrr_total = 0.0
    h_at_r_total = 0.0

    #Load the saved best model
    print("Loading model with loss: {}".format(best_info['loss_at_best']))
    model = torch.load(PATH)
    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

    for a_ind, (a, df_test_a) in enumerate(a_dfs.iteritems()):
        test_x, test_labels, original_labels, c_lst = get_input(df_test_a, weights, index_map, combination_method, data_purpose='test')
        test_loader = torch.utils.data.DataLoader(dataset=LinksDataset(test_x, test_labels), batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            predictions = []
            pred_lab_lst = []
            for test_links, test_labels in test_loader:
                test_links = test_links.to(device)
                outputs = model(test_links)
                predicted, _ = torch.max(outputs.data, 1)
                predictions.extend([tensor.item() for tensor in predicted])

                for pred, lab in zip(predictions, [tensor.item() for tensor in test_labels]):
                    pred_lab_lst.append((pred, lab))

        sorted_pred_lab_lst = sorted(pred_lab_lst, key=lambda x: x[0], reverse=True)

        y_true = [tup[1] for tup in sorted_pred_lab_lst] #gold
        y_scores = [tup[0] for tup in sorted_pred_lab_lst] #predictions

        true_inds = [ind for ind in range(len(y_true)) if y_true[ind] == 1]
        true_scores = [y_scores[ind] for ind in true_inds]
        sorted_scores = sorted(y_scores, reverse=True)
        true_ranks = []

        for tc, ts in zip(c_lst, true_scores):
            true_ranks.append((sorted_scores.index(ts) + 1, ts, tc))

        ap = average_precision_score(y_true, y_scores, average='micro')
        mr = np.mean([tr_[0] for tr_ in true_ranks])
        mrr = np.mean([1.0/tr_[0] for tr_ in true_ranks])
        hits_at_R = len([tup_[0] for tup_ in sorted(true_ranks, key= lambda x: x[0]) if tup_[0] <= len(true_ranks)])/float(len(true_ranks))

        map_total += ap
        mr_total += mr
        mrr_total += mrr
        h_at_r_total += hits_at_R

        tp = len([x for x in y_true if x > 0.0])
        a_ap = "{}. A: {}, AP: {}. MR: {}. MRR: {}. Hits at R: {}. TP: {}/{}. \nRanks: {}\n\n".format(a_ind + 1, a, ap, mr, mrr, hits_at_R, tp, len(predictions), sorted(true_ranks, key= lambda x: x[0]))
        print(a_ap)
        map_output += a_ap

    mean_map = map_total/len(a_dfs.keys())
    mean_mr = mr_total/len(a_dfs.keys())
    mean_mrr = mrr_total/len(a_dfs.keys())
    mean_hits_at_r = h_at_r_total/len(a_dfs.keys())

    map_o = "Mean MAP was: {}. Mean mean-rank was: {}. MRR: {}. Mean Hits at R: {}".format(mean_map, mean_mr, mean_mrr, mean_hits_at_r)
    map_output = "{}\n{}\n{}\n\n{}".format(experiment_name, best_info, map_o, map_output)
    print(map_o)

    fil = open("Eval-Scores-{}.txt".format(experiment_name), 'w')
    fil.write(map_output)
    fil.close()

def calculate_scores(a, neighbours_data, all_data, cutoff_year, metric, agg, acc):
    a_neighbours = neighbours_data[a]
    c_scores = {}
    for b in a_neighbours:
        cs = neighbours_data[b]
        for c in cs:
            if c not in a_neighbours and c != a:
                if "{}::{}".format(a,b) in all_data:
                    ab_key = "{}::{}".format(a,b)
                elif "{}::{}".format(b,a) in all_data:
                    ab_key = "{}::{}".format(b,a)
                else:
                    print("ERROR: {} and {} not found in data.".format(a, b))

                if "{}::{}".format(b,c) in all_data:
                    bc_key = "{}::{}".format(b,c)
                elif "{}::{}".format(c,b) in all_data:
                    bc_key = "{}::{}".format(c,b)
                else:
                    print("ERROR: {} and {} not found in data.".format(b,c))

                ab_score = all_data[ab_key]
                bc_score = all_data[bc_key]

                if agg == 'avg':
                    agg_score = sum([ab_score, bc_score])/2.0
                if agg == 'min':
                    agg_score = min(ab_score, bc_score)

                if c in c_scores:
                    if acc == 'sum':
                        c_scores[c] += agg_score
                else:
                    c_scores[c] = agg_score

    for key,value in c_scores.iteritems():
        c_scores[key] = round(value, 8)
    return c_scores

def read_original_data(input_data_file, col_labels, col_indices, cutoff_year):
    col_indices_vals = col_indices.split(':')
    col_labels_vals = col_labels.split(',')
    assert(len(col_indices_vals) == len(col_labels_vals)), "Lenghts of Labels ({}) and Indices ({}) do not match.".format(col_labels, col_indices)
    indices = {}
    for label, index in zip(col_labels_vals, col_indices_vals):
        indices[label] = index

    split_indices = col_indices.split(':')
    assert(len(split_indices) >= 2), "Incorrect length for indices: {}".format(len(split_indices))
    attribute_indices = {}
    for num, index in enumerate(split_indices):
        if num == 0:
            entity1_index = int(index)
        elif num == 1:
            entity2_index = int(index)
        elif num == 2:
            year_index = int(index)
        elif num == 3:
            score_index = int(index)

    with open(input_data_file) as tsv:
        entity1_lst = []
        entity2_lst = []
        entity_neighbours = {}
        future_entity_neighbours = {}
        future_links_cnt = 0

        self_referential_edges = 0
        data = {}
        for ind, line in enumerate(csv.reader(tsv)): #quoting=csv.QUOTE_NONE - If req to make data work, examine data
            if ind == 0:
                continue #skip header
            attribute_values = {}
            entity1 = line[entity1_index].replace(' ', '_').replace('-', '_')
            entity2 = line[entity2_index].replace(' ', '_').replace('-', '_')
            year = line[year_index].replace(' ', '_').replace('-', '_')
            score = process_score(line[score_index], int(year), int(cutoff_year))

            if entity1 == entity2:
                self_referential_edges += 1

            key1 = "%s::%s" % (entity1, entity2)
            key2 = "%s::%s" % (entity2, entity1)

            if key1 not in data and key2 not in data:
                #Shuffle node order in key
                if random.choice([0,1]) == 1:
                    data[key1] = float(score) #attribute_values
                else:
                    data[key2] = float(score) #attribute_values
                entity1_lst.append(entity1)
                entity2_lst.append(entity2)

                if int(year) <= cutoff_year:
                    if entity1 in entity_neighbours:
                        entity_neighbours[entity1].append(entity2)
                    else:
                        entity_neighbours[entity1] = [entity2]
                    if entity2 in entity_neighbours:
                        entity_neighbours[entity2].append(entity1)
                    else:
                        entity_neighbours[entity2] = [entity1]
                else:
                    future_links_cnt += 1
                    if entity1 in future_entity_neighbours:
                        future_entity_neighbours[entity1].append(entity2)
                    else:
                        future_entity_neighbours[entity1] = [entity2]
                    if entity2 in future_entity_neighbours:
                        future_entity_neighbours[entity2].append(entity1)
                    else:
                        future_entity_neighbours[entity2] = [entity1]

    print("Read data complete.")
    return data, entity_neighbours, future_entity_neighbours

def process_score(score_str, link_year, cutoff_year):
    score_arr = score_str.split(';')
    year_diff = cutoff_year - link_year
    if year_diff >= 0:
        return score_arr[year_diff]
    return 0


FLAGS = None
def main(_):
    all_data, neighbours_data, future_neighbours_data = read_original_data(FLAGS.input_train_file, FLAGS.col_labels, FLAGS.col_indices, int(FLAGS.cutoff_year))
    train_and_eval(FLAGS.train_epochs, FLAGS.train_data, FLAGS.devel_data, FLAGS.test_data, FLAGS.train_embeddings_data, FLAGS.test_embeddings_data, FLAGS.devel_filename, FLAGS.eval_filename,
        FLAGS.devel_unformed_filename, FLAGS.unformed_filename, FLAGS.positive_labels, FLAGS.combination_method, FLAGS.method, FLAGS.c_list, FLAGS.lbd_type, FLAGS.experiment_name,
        all_data, neighbours_data, int(FLAGS.cutoff_year))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")

  parser.add_argument(
      '-if',
      "--input_train_file",
      type=str,
      default="",
      help="Path to input data."
  )
  parser.add_argument(
      '-ci',
      '--col_indices',
      type=str,
      default='0:1',
      help='Index of tsv file where information on entities and attributes are (default 0:1)'
  )
  parser.add_argument(
      '-cl',
      '--col_labels',
      type=str,
      default='entity1,entity2',
      help='Labels of the data in the tsv file where entities and attributes are (default 0:1)'
  )
  parser.add_argument(
      '-cy',
      '--cutoff_year',
      type=str,
      help='Year at which to use data until. Ignore all links after that year.'
  )
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
  FLAGS, unparsed = parser.parse_known_args()
  main(sys.argv[0])
