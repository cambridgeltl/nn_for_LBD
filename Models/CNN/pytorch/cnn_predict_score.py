import argparse
import sys
import random
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import wordvecdata as wvd

COLUMNS = ["node1", "node2", "node3"]
LABEL_COLUMN = "label"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
batch_size = 100 #TODO pass this from command line
learning_rate = 0.00001
frame_link_amt = 50
conv_height = 7

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

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, conv_height, conv_width=300):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(conv_height, conv_width), stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1))
            )
        self.fc = nn.Linear(2816, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.softplus(out)
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

    col_keys = []
    for ind, row in enumerate(df.itertuples()):
        input_str = row[1]
        conv_rows = input_str.split('-')

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

        #combine all feature vectors into the conv window size instead of simply truncating
        updated_instance_features = instance_features[:frame_link_amt]
        start_row = frame_link_amt
        end_row = frame_link_amt * 2
        while end_row <= len(conv_rows):
            updated_instance_features = np.add(updated_instance_features, instance_features[start_row:end_row])
            start_row = end_row
            end_row += frame_link_amt

        #sum the remaining rows
        #TODO: can perhaps be done more elegantly
        if start_row < len(conv_rows):
            padding_needed = end_row - len(conv_rows)
            padded_instance_features = np.concatenate([instance_features[start_row:], np.ones((padding_needed, instance_features.shape[1]),dtype='float32')])
            updated_instance_features = np.add(updated_instance_features, padded_instance_features)

        features.append(np.expand_dims(updated_instance_features, axis=0))
        total_rows = ind

    print("\nReal links in conv window stats: Range from {}-{} with a mean of {}.".format(min_real_links, max_real_links,
                                                            total_real_links/max(total_rows, 1)))

    return features, np.array([val * 100 for val in label_values if val != -1])

def build_model(conv_width):
    """Build model."""

    model = ConvNet(conv_height, conv_width).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, criterion, optimizer

def train_and_eval(train_epochs, train_data, test_data, train_embeddings_file_name, test_embeddings_file_name, combination_method, a, c, gold_b, case_name, c_lst, lbd_type):
    """Train and evaluate the model."""

    index_map, weights = wvd.load(test_embeddings_file_name)

    print("reading training data...")
    train_file_name = train_data
    df_train = pd.read_table(train_file_name, dtype={'train_nodes':str})
    df_train = df_train.sample(frac=1)

    # remove NaN elements
    df_train = df_train.dropna(how='any', axis=0)

    #Get inputs
    train_x, labels = get_input(df_train, weights, index_map, combination_method)
    train_loader = torch.utils.data.DataLoader(dataset=LinksDataset(train_x, labels),
                                               batch_size=batch_size, shuffle=True)

    pos_labels = [l_ for l_ in labels if l_ != 0]
    print("Train labels stats: {} - {} with mean of {}. All positive means: {}.".format(min(labels), max(labels), sum(labels)/len(labels), sum(pos_labels)/len(pos_labels)))

    #Start loading evaluation data (same as test data for cancer cases)
    print("reading eval data...")
    test_file_name = test_data
    df_test = pd.read_table(test_file_name, dtype={'train_nodes':str})

    #Bit of a hack to get the original AC keys
    test_keys = []
    for row in df_test.itertuples():
        single_link = row[1].split('-')[0]
        a_,b_,c_ = single_link.split('::')
        test_keys.append("%s::%s" % (a_, c_))

    # remove NaN elements
    df_test = df_test.dropna(how='any', axis=0)

    c_dict = {}
    if c_lst:
        c_file = open(c_lst, 'r')
        c_ = c_file.readline().strip(' \n\t')
        while c_:
            c_dict[c_] = 1
            c_ = c_file.readline().strip(' \n\t')

    test_x, test_labels = get_input(df_test, weights, index_map, combination_method, data_purpose='test')
    test_loader = torch.utils.data.DataLoader(dataset=LinksDataset(test_x, test_labels), batch_size=batch_size, shuffle=False)
    #End loading evaluation data

    print("\nBuilding model...")
    feature_dim = train_x[0].shape[2]
    model, criterion, optimizer = build_model(feature_dim)

    # Train the model
    print("\nTraining model...")
    total_step = len(train_loader)
    best_info = {'best_rank':1000000000}
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
            print("Evaluating at epoch {}".format(epoch + 1))

            #Save the current model
            torch.save(model, CUR_PATH)
            #Load the last saved best model
            lmodel = torch.load(CUR_PATH)
            # Test the model
            lmodel.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

            # Test the model
            with torch.no_grad():
                predictions = []
                for test_links, _ in test_loader:
                    test_links = test_links.to(device)
                    outputs = lmodel(test_links)
                    predicted, _ = torch.max(outputs.data, 1)
                    predictions.extend([tensor.item() for tensor in predicted])

            if lbd_type == 'open_discovery_ac':
                rank, ties, output = do_case_od_evaluations_ac(a, c, c_dict, [p for p in predictions], test_keys, case_name)
                if rank < best_info['best_rank'] and ties < 11:
                    print("Saving because {} < {}".format(rank, best_info['best_rank']))
                    torch.save(model, PATH)
                    best_info['case_name'] = case_name
                    best_info['best_rank'] = rank
                    best_info['loss_at_best'] = loss.item()
                    best_info['epoch'] = epoch + 1
                    best_info['output'] = output

                    fil = open("{}.txt".format(case_name), 'w')
                    fil.write(str(best_info))
                    fil.close()
            else:
                print("ERROR: Invalid lbd_type: {}".format(lbd_type))

def do_case_od_evaluations_ac(a, gold_c, all_cs, predictions, test_x, case_name):
    for x_link, score in zip(test_x, predictions):
        entity1 = x_link.split('::')[0]
        entity2 = x_link.split('::')[1]
        if entity1 == a:
            c = entity2
        elif entity2 == a:
            c = entity1
        else:
            print("Error: A ({}) not in link {}.".format(a, x_link))
        if c == gold_c:
            gold_c_score = score
            break

    rank = 1
    ties = 0

    min_score = 10000
    max_score = 0
    total_score = 0
    for x_link, score in zip(test_x, predictions):
        entity1 = x_link.split('::')[0]
        entity2 = x_link.split('::')[1]
        if entity1 == a:
            c = entity2
        elif entity2 == a:
            c = entity1
        else:
            print("Error: A ({}) not in link {}.".format(a, x_link))

        if score > gold_c_score:
            rank += 1
        if score == gold_c_score and c != gold_c:
            ties += 1

        if score > max_score:
            max_score = score
        if score < min_score:
            min_score = score
        total_score += score

    print("Scores range: {}-{} with a mean of {}.".format(min_score, max_score, total_score/len(test_x)))
    output = "Of {} ACs, gold AC {}:{}\n Rank: {}, Ties with gold: {}. Score: {}.".format(len(all_cs), a, gold_c, rank, ties, gold_c_score)
    print(output)
    return rank, ties, output

FLAGS = None
def main(_):
    train_and_eval(FLAGS.train_epochs, FLAGS.train_data, FLAGS.test_data, FLAGS.train_embeddings_data, FLAGS.test_embeddings_data, FLAGS.combination_method,
                        FLAGS.a_node, FLAGS.c_node, FLAGS.goldb_node, FLAGS.experiment_name, FLAGS.c_list, FLAGS.lbd_type)


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
      "--combination_method",
      type=str,
      default="concatenate",
      help="How the features should be combined by the model."
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
      "--experiment_name",
      type=str,
      default="",
      help="name of closed discovery case."
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
