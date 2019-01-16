import sys
import csv
import random
import multiprocessing
import networkx as nx

from datetime import datetime

def argparser():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--input-file', help='Input .tsv file')
    ap.add_argument('-n', '--negative-ratio', default=1, help='Ratio of negative to positive examples to create (default 1)')
    ap.add_argument('-a', '--a_node', help='Name of "A" node.')
    ap.add_argument('-c', '--c_node', default=None, help='Name of "C" node.')
    ap.add_argument('-cy', '--cutoff_year', help='Year at which to use data until. Ignore all links after that year.')
    ap.add_argument('-ts', '--train_set_size', help='Amount of edges in train set.')
    ap.add_argument('-cd', '--closed_discovery', default=False, help='Whether to do closed discovery or not.')
    ap.add_argument('-od', '--open_discovery', default=False, help='Whether to do open discovery or not.')

    ap.add_argument('-tf', '--train-filename', default='train.tsv', help='name of file for training data (default train.tsv)')
    ap.add_argument('-df', '--devel-filename', default='devel.tsv', help='name of file for development data (default devel.tsv)')
    ap.add_argument('-tef', '--test-filename', default='test.tsv', help='name of file for testing data (default test.tsv)')

    ap.add_argument('-vf', '--vertices-filename', default='vertices.txt', help='name of file containing mapping of all vertex number to name (default vertices.txt)')
    ap.add_argument('-tegf', '--test-graph-filename', default='test_adj_mat.adjlist', help='name of file to store graph for representation induction for testing (default test_adj_mat.adjlist)')
    ap.add_argument('-bf', '--B-filename', default='b.txt', help='name of file to store all the Bs in Closed Discovery (default b.txt)')
    ap.add_argument('-cf', '--C-filename', default='Cs.txt', help='name of file to store all the Cs in Open Discovery (default Cs.txt)')

    ap.add_argument('-v', '--values', default='0:1', help='Values for labels (default 0:1)')
    ap.add_argument('-l', '--labels', default='O:I-LINK', help='Labels for values (default O:I-LINK)')

    ap.add_argument('-x', '--indices', default='0:1', help='Index of tsv file where entities can be found (default 0:1)')
    ap.add_argument('-ci', '--col_indices', default='0:1', help='Index of tsv file where information on entities and attributes are (default 0:1)')
    ap.add_argument('-cl', '--col_labels', default='entity1,entity2', help='Labels of the data in the tsv file where entities and attributes are (default 0:1)')

    ap.add_argument('-lbdm', '--lbd_method', help='Type of LBd to prepare the data for.')

    return ap

def read_data(input_data_file, col_labels, col_indices, cutoff_year):
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
        for ind, line in enumerate(csv.reader(tsv)):
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
                    data[key1] = score #attribute_values
                else:
                    data[key2] = score #attribute_values
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

    print("\n{} nodes read. ({} entity neighbours) {} edges read.".format(len(set(entity1_lst + entity2_lst)), len(entity_neighbours), len(data)))
    print("\n{} future entity neighbour nodes {} future edges read.".format(len(future_entity_neighbours), future_links_cnt))
    print("{}/{:,} ({}%) edges were self-referential.".format(self_referential_edges, len(data), (self_referential_edges/float(len(data)))*100 ))

    return data, entity_neighbours, future_entity_neighbours

def process_score(score_str, link_year, cutoff_year):
    score_arr = score_str.split(';')
    year_diff = cutoff_year - link_year
    if year_diff >= 0:
        return score_arr[year_diff]
    return 0

def create_adjmat(neighbours_data, vertices_filename, test_graph_filename, all_data=None):
    graph_format = test_graph_filename.split('.')[-1]

    test_output = ""
    test_output_cnt = 0

    vertices = {}
    node_index = 0
    for key, value in neighbours_data.iteritems():
        if key not in vertices:
            node_index += 1
            vertices[key] = str(node_index)
    print("\n{} vertices read.".format(len(vertices)))

    if graph_format == 'edgelist':
        #format of node2vec
        for key, value in neighbours_data.iteritems():
            if len(value) > 0:
                for node_edge in value:
                    key1 = "{}::{}".format(key, node_edge)
                    key2 = "{}::{}".format(node_edge, key)
                    if key1 in all_data:
                        test_output += "{} {} {}\n".format(vertices[key], vertices[node_edge], float(all_data[key1]))
                    elif key2 in all_data:
                        test_output += "{} {} {}\n".format(vertices[key], vertices[node_edge], float(all_data[key2]))
                    else:
                        print("ERROR: {} nor {} not in data!".format(key1, key2))
            else:
                test_output += "{} {}\n".format(key, key, 0.0) #Hack to create an edge for nodes which have no edges as node2vec does not ceate embeddings for them.
    elif graph_format == 'line':
        #format of LINE
        for key, value in neighbours_data.iteritems():
            if len(value) > 0:
                for node_edge in value:
                    key1 = "{}::{}" .format(key, node_edge)
                    key2 = "{}::{}".format(node_edge, key)
                    if key1 in all_data:
                        test_output += "{} {} {}\n".format(key, node_edge, float(all_data[key1]) * 100)
                    elif key2 in all_data:
                        test_output += "{} {} {}\n".format(key, node_edge, float(all_data[key2]) * 100)
                    else:
                        print("ERROR: {} nor {} not in data!".format(key1, key2))
            else:
                test_output += "{} {} {}\n".format(key, key, 0.0)

    test_graph = open(test_graph_filename, 'w')
    test_graph.write(test_output)
    test_graph.close()

    voutput = ""
    fil3 = open(vertices_filename, 'w')
    for vertex, index in vertices.iteritems():
        voutput += "{} {}\n".format(index, vertex)
    fil3.write(voutput)

    return vertices

def get_closed_discovery_Bs(neighbours_data, a, c):
    a_set = set(neighbours_data[a])
    c_set = set(neighbours_data[c])
    common_neighbours = a_set.intersection(c_set)
    print("{} Bs found.".format(len(common_neighbours)))
    return common_neighbours

def create_closed_discovery_test(b_set, a, c, neighbours_data, test_filename):
    test_hops = {}
    output = 'node1\tnode2\tlabel\n'
    for b in b_set:
        #Calculate score for AB label
        #TODO: only uses Jaccard; add others
        score = 0.0
        if a in neighbours_data and b in neighbours_data:
            a_neighbours = set(neighbours_data[a])
            b_neighbours = set(neighbours_data[b])
            score = len(a_neighbours.intersection(b_neighbours)) / float(len(a_neighbours.union(b_neighbours)))
        output += "{}\t{}\t{}\n".format(a,b,score)

        #Calculate score for BC label
        #TODO: only uses Jaccard; add others
        score = 0.0
        if c in neighbours_data and b in neighbours_data:
            b_neighbours = set(neighbours_data[b])
            c_neighbours = set(neighbours_data[c])
            score = len(b_neighbours.intersection(c_neighbours)) / float(len(b_neighbours.union(c_neighbours)))
        output += "{}\t{}\t{}\n".format(b,c,score)

        test_hops["{}::{}".format(a,b)] = 1
        test_hops["{}::{}".format(b,a)] = 1
        test_hops["{}::{}".format(b,c)] = 1
        test_hops["{}::{}".format(c,b)] = 1

    test_file = open(test_filename, 'w')
    test_file.write(output)
    test_file.close()

    return test_hops

def create_closed_discovery_test_abc(b_set, a, c, neighbours_data, test_filename):
    test_hops = {}
    output = 'node1\tnode2\tnode3\tlabel\n'

    #Calculate score for label
    #TODO: only uses Jaccard; add others
    score = 0.0
    if a in neighbours_data and c in neighbours_data:
        a_neighbours = set(neighbours_data[a])
        c_neighbours = set(neighbours_data[c])
        score = len(a_neighbours.intersection(c_neighbours)) / float(len(a_neighbours.union(c_neighbours)))

    for b in b_set:
        output += "{}\t{}\t{}\t{}\n".format(a,b,c,score)

        #All permutations of the link are considered equal
        test_hops["{}::{}::{}".format(a,b,c)] = 1
        test_hops["{}::{}::{}".format(a,c,b)] = 1
        test_hops["{}::{}::{}".format(b,c,a)] = 1
        test_hops["{}::{}::{}".format(b,a,c)] = 1
        test_hops["{}::{}::{}".format(c,b,a)] = 1
        test_hops["{}::{}::{}".format(c,a,b)] = 1

    test_file = open(test_filename, 'w')
    test_file.write(output)
    test_file.close()

    return test_hops

def create_closed_discovery_train(neighbours_data, test_hops, size, neg_ratio, train_filename):
    #TODO: add some check of size to make sure function doesn't get stuck in infinite loop

    negative_cnt = neg_ratio * size
    positive_cnt = size - negative_cnt
    positive_egs = 0
    negative_egs = 0

    train_set = {}
    while len(train_set) < size:
        if len(train_set) % 10000 < 2:
            print("{} train examples added at {}.".format(len(train_set), datetime.now()))

        if positive_egs < positive_cnt:
            #randomly select 2 entities
            a = random.choice(neighbours_data.keys())
            random_b = random.choice(neighbours_data[a])

            if "{}::{}".format(a, random_b) not in test_hops:

                #Calculate score for label
                #TODO: only uses Jaccard; add others
                score = 0.0
                if a in neighbours_data and random_b in neighbours_data:
                    a_neighbours = set(neighbours_data[a])
                    c_neighbours = set(neighbours_data[random_b])
                    score = len(a_neighbours.intersection(c_neighbours)) / float(len(a_neighbours.union(c_neighbours)))

                pos_eg = "{}::{}::{}\n".format(a, random_b, score)
                if pos_eg not in train_set and score > 0.0:
                    train_set[pos_eg] = 1
                    positive_egs += 1
        if negative_egs < negative_cnt:
            #randomly select 2 entities
            a = random.choice(neighbours_data.keys())
            random_b = random.choice(neighbours_data.keys())
            while random_b in neighbours_data[a]:
                random_b = random.choice(neighbours_data.keys())
            if "{}::{}".format(a, random_b) not in test_hops:
                neg_eg = "{}::{}::{}\n".format(a, random_b, 0.0)
                if neg_eg not in train_set:
                    train_set[neg_eg] = 1
                    negative_egs += 1

    print("Train set creation complete at {}. Total train: {}. Positives: {}. Negatives: {}".format(datetime.now(),len(train_set), positive_egs, negative_egs))
    output = 'node1\tnode2\tlabel\n'
    for link in train_set:
        output += link.replace('::', '\t')
    train_file = open(train_filename, 'w')
    train_file.write(output)
    train_file.close()

    return train_set

def create_closed_discovery_train_abc(neighbours_data, test_hops, size, neg_ratio, train_filename):
    #TODO: add some check of size to make sure function doesn't get stuck in infinite loop

    negative_cnt = neg_ratio * size
    positive_cnt = size - negative_cnt
    positive_egs = 0
    negative_egs = 0

    train_set = {}
    while len(train_set) < size:
        if len(train_set) % 10000 < 2:
            print("{} train examples added at {}.".format(len(train_set), datetime.now()))

        if positive_egs < positive_cnt:
            #randomly select 1 entity and get 2 more from 1 and 2 hops away respectively
            a = random.choice(neighbours_data.keys())
            random_b = random.choice(neighbours_data[a])
            b_neighbours = [b_ for b_ in neighbours_data[random_b]]
            b_neighbours.remove(a)
            if len(b_neighbours) > 0:
                random_c = random.choice(b_neighbours)
                if "{}::{}::{}".format(a, random_b, random_c) not in test_hops:
                    #Calculate score for label
                    #TODO: only uses Jaccard; add others
                    score = 0.0
                    if a in neighbours_data and random_c in neighbours_data:
                        a_neighbours = set(neighbours_data[a])
                        c_neighbours = set(neighbours_data[random_c])
                        score = len(a_neighbours.intersection(c_neighbours)) / float(len(a_neighbours.union(c_neighbours)))

                    pos_eg = "{}::{}::{}::{}\n".format(a, random_b, random_c, score)
                    if pos_eg not in train_set and score > 0.0:
                        train_set[pos_eg] = 1
                        positive_egs += 1
        if negative_egs < negative_cnt:
            #randomly select 3 entities
            a = random.choice(neighbours_data.keys())
            random_b = random.choice(neighbours_data.keys())
            random_c = random.choice(neighbours_data.keys())
            while a == random_b or a == random_c or random_b == random_c or random_b in neighbours_data[a] or random_b in neighbours_data[random_c] or random_c in neighbours_data[a]:
                random_b = random.choice(neighbours_data.keys())
                random_c = random.choice(neighbours_data.keys())
            if "{}::{}::{}".format(a, random_b, random_c) not in test_hops:
                neg_eg = "{}::{}::{}::{}\n".format(a, random_b, random_c, 0.0)
                if neg_eg not in train_set:
                    train_set[neg_eg] = 1
                    negative_egs += 1

    print("Train set creation complete at {}. Total train: {}. Positives: {}. Negatives: {}".format(datetime.now(),len(train_set), positive_egs, negative_egs))
    output = 'node1\tnode2\tnode3\tlabel\n'
    for link in train_set:
        output += link.replace('::', '\t')
    train_file = open(train_filename, 'w')
    train_file.write(output)
    train_file.close()

    return train_set

def create_open_discovery_test(neighbours_data, a, b_filename, c_filename, test_filename):
    c_lst = {}
    a_neighbours = set(neighbours_data[a])
    print("A has {} neighbours.".format(len(a_neighbours)))

    test_set = {}
    output = 'node1\tnode2\tlabel\n'
    b_added = []
    for b in a_neighbours:

        #Remove previous AC if it turns out a direct link already exist
        ab = "{}::{}".format(a,b)
        ba = "{}::{}".format(b,a)

        for c in set(neighbours_data[b]):
            if c == a or c in a_neighbours:
                continue

            bc = "{}::{}".format(b,c)
            cb = "{}::{}".format(c,b)

            ac = "{}::{}".format(a,c)
            ca = "{}::{}".format(c,a)
            if bc not in test_set and cb not in test_set:
                if ab not in test_set:
                    test_set[ab] = 1
                    b_added.append(b)
                test_set[bc] = 1
                c_lst[c] = 1

    print("Total test link: {}. Total Cs: {}".format(len(test_set), len(c_lst)))
    b_added = set(b_added)
    print("Total Bs added: {}".format(len(b_added)))

    ##start multiprocessing
    manager = multiprocessing.Manager()
    return_outputs = manager.dict()

    split_sizes = len(test_set.keys())/16 #divisor is amt of CPU cores to use
    test_set_sub = {}
    jobs = []
    for ind, link in enumerate(test_set):
        test_set_sub[link] = 1
        if ind + 1 == len(test_set):
            p = multiprocessing.Process(target=process_scores_func, args=(test_set_sub,neighbours_data, ind + 1, return_outputs))
            p.start()
            jobs.append(p)
        elif len(test_set_sub) % split_sizes == 0:
            p = multiprocessing.Process(target=process_scores_func, args=(test_set_sub,neighbours_data, ind + 1, return_outputs))
            jobs.append(p)
            p.start()
            test_set_sub = {}

    for proc in jobs:
        proc.join()
    final_output = 'node1\tnode2\tlabel\n'
    final_outputs = []
    for output in return_outputs.values():
        final_outputs.append(output)
    outputs = "".join(final_outputs)
    output = final_output + outputs
    ##end multiprocessing

    test_file = open(test_filename, 'w')
    test_file.write(output)
    test_file.close()

    c_output = ''
    for c_ in c_lst:
        c_output += "{}\n".format(c_)
    c_file = open(c_filename, 'w')
    c_file.write(c_output)
    c_file.close()

    b_file = open(b_filename, 'w')
    b_file.write(', '.join([str(b_) for b_ in b_added]))
    b_file.close()

    return test_set

def process_scores_func(test_set_sub, neighbours_data, end_ind, return_output):
    print("Start process with end ind: {} at {}".format(end_ind, datetime.now()))
    output = ""
    for test_link in test_set_sub:
        a_ = test_link.split('::')[0]
        other = test_link.split('::')[1]

        #Calculate score for label
        #TODO: only uses Jaccard; add others
        score = 0.0
        if a_ in neighbours_data and other in neighbours_data:
            a_neighbours = set(neighbours_data[a_])
            other_neighbours = set(neighbours_data[other])
            score = len(a_neighbours.intersection(other_neighbours)) / float(len(a_neighbours.union(other_neighbours)))
        output += "{}\t{}\t{}\n".format(a_,other,score)
    print("End process with end ind: {} at {}".format(end_ind, datetime.now()))
    return_output[end_ind] = output

def create_open_discovery_train(neighbours_data, test_hops, size, neg_ratio, train_filename): #, pos_label, neg_label):
    #TODO: add some check of size to make sure function doesn't get stuck in infinite loop

    negative_cnt = neg_ratio * size
    positive_cnt = size - negative_cnt
    positive_egs = 0
    negative_egs = 0

    train_set = {}
    while len(train_set) < size:
        if len(train_set) % 50000 < 10:
            print("{} train examples added at {}. Positives: {}. Negatives: {}".format(len(train_set), datetime.now(), positive_egs, negative_egs))

        if positive_egs < positive_cnt:
            #randomly select 2 entities
            a = random.choice(neighbours_data.keys())
            random_b = random.choice(neighbours_data[a])
            if "{}::{}".format(a, random_b) not in test_hops:

                #Calculate score for label
                #TODO: only uses Jaccard; add others
                score = 0.0
                if a in neighbours_data and random_b in neighbours_data:
                    a_neighbours = set(neighbours_data[a])
                    random_b_neighbours = set(neighbours_data[random_b])
                    score = len(a_neighbours.intersection(random_b_neighbours)) / float(len(a_neighbours.union(random_b_neighbours)))

                pos_eg = "{}::{}::{}\n".format(a, random_b, score)
                if pos_eg not in train_set and score > 0.0:
                    train_set[pos_eg] = 1
                    positive_egs += 1

        if negative_egs < negative_cnt:
            #randomly select 2 entities
            a = random.choice(neighbours_data.keys())
            random_b = random.choice(neighbours_data.keys())
            while random_b in neighbours_data[a]:
                random_b = random.choice(neighbours_data.keys())
            if "{}::{}".format(a, random_b) not in test_hops:
                neg_eg = "{}::{}::{}\n".format(a, random_b, 0.0)
                if neg_eg not in train_set:
                    train_set[neg_eg] = 1
                    negative_egs += 1

    print("Train set creation completed at {}. Total train: {}. Positives: {}. Negatives: {}".format(datetime.now(), len(train_set), positive_egs, negative_egs))
    output = 'node1\tnode2\tlabel\n'
    for link in train_set:
        output += link.replace('::', '\t')
    train_file = open(train_filename, 'w')
    train_file.write(output)
    train_file.close()

    return train_set


def create_open_discovery_convnet_test(neighbours_data, future_neighbours_data, a, pos_label, test_filename, c_filename, b_filename, all_data):
    c_lst = {}
    a_neighbours = set(neighbours_data[a])
    print("A has {} neighbours.".format(len(a_neighbours)))

    test_set = {}
    test_set_conv = {}
    b_added = []
    real_cs = 0
    a_neighbours_fut = set(neighbours_data[a] + future_neighbours_data[a])
    a_neighbours_fut_len = len(a_neighbours_fut)
    for b in a_neighbours:
        for c in set(neighbours_data[b]):
            if c == a or c in a_neighbours:
                continue

            ac = "{}::{}".format(a,c)
            ca = "{}::{}".format(c,a)

            if ac not in test_set and ca not in test_set:
                #Calculate score for label
                #TODO: only uses Jaccard; add others
                score = 0.0
                if c in future_neighbours_data and c in neighbours_data:
                    c_neighbours = set(neighbours_data[c] + future_neighbours_data[c])
                    score = len(a_neighbours_fut.intersection(c_neighbours)) / float(len(a_neighbours_fut.union(c_neighbours)))
                elif c in future_neighbours_data:
                    #not in both and so in futures only
                    c_neighbours = set(future_neighbours_data[c])
                    score = len(a_neighbours_fut.intersection(c_neighbours)) / float(len(a_neighbours_fut.union(c_neighbours)))
                elif c in neighbours_data:
                    #Not in both so in pre cut-off neighbours only
                    c_neighbours = set(neighbours_data[c])
                    score = len(a_neighbours_fut.intersection(c_neighbours)) / float(len(a_neighbours_fut.union(c_neighbours)))

                test_set[ac] = score
                c_lst[c] = len(c_neighbours)
                test_set_conv[ac] = [b]
                if (a in future_neighbours_data and c in future_neighbours_data) and (c in future_neighbours_data[a] or a in future_neighbours_data[c]):
                    real_cs += 1
            else:
                if ac in test_set and b not in test_set_conv[ac]:
                    test_set_conv[ac].append(b)
                if ca in test_set and b not in test_set_conv[ca]:
                    test_set_conv[ca].append(b)

    print("\nTotal test link: {}. Total possible Cs: {}".format(len(test_set), len(c_lst)))
    print("Total ACs which did form: {}. ACs which did not form: {}".format(real_cs, len(c_lst) - real_cs))

    #Rank Bs in descending order of score and format just as training data
    ordered_test_set_conv = {}
    b_score_lst = {}
    for ind, (ac, b_lst) in enumerate(test_set_conv.iteritems()):
        #Start sorting Bs by score
        sorted_b_lst = []
        for b in b_lst:
            if "{}::{}".format(a,b) in all_data:
                ab = "{}::{}".format(a,b)
            elif "{}::{}".format(b,a) in all_data:
                ab = "{}::{}".format(b,a)
            else:
                print("ERROR: unknown key: {}::{}".format(a,b))

            sorted_b_lst.append((b, float(all_data[ab])))
        sorted_b_lst = sorted(sorted_b_lst, key = lambda x: x[1], reverse=True)
        sorted_b_lst = [tup_[0] for tup_ in sorted_b_lst]
        b_lst = sorted_b_lst
        #End sorting Bs by score
        ordered_test_set_conv[ac] = b_lst
    test_set_conv = ordered_test_set_conv
    #End Ranking Bs

    test_set_final = {}
    for ac, b_lst in test_set_conv.iteritems():

        ent1 = ac.split('::')[0]
        ent2 = ac.split('::')[1]
        if ent1 == a:
            c = ent2
        else:
            c = ent1

        #Calculate score for label
        #TODO: only uses Jaccard; add others
        score = 0.0
        if c in future_neighbours_data and c in neighbours_data:
            c_neighbours = set(neighbours_data[c] + future_neighbours_data[c])
            score = len(a_neighbours_fut.intersection(c_neighbours)) / float(len(a_neighbours_fut.union(c_neighbours)))
        elif c in future_neighbours_data:
            #not in both and so in futures only
            c_neighbours = set(future_neighbours_data[c])
            score = len(a_neighbours_fut.intersection(c_neighbours)) / float(len(a_neighbours_fut.union(c_neighbours)))
        elif c in neighbours_data:
            #Not in both so in pre cut-off neighbours only
            c_neighbours = set(neighbours_data[c])
            score = len(a_neighbours_fut.intersection(c_neighbours)) / float(len(a_neighbours_fut.union(c_neighbours)))

        conv_frame = ""
        a = ac.split('::')[0]
        c = ac.split('::')[1]
        for ind, b in enumerate(b_lst):
            b_added.append(b)
            conv_frame += "{}::{}::{}".format(a, b, c)
            if ind < len(b_lst) - 1:
                conv_frame += "-"
        test_set_final[conv_frame] = score

    output = 'train_nodes\tlabel\n'
    for test_frame, score_ in test_set_final.iteritems():
        output += "{}\t{}\n".format(test_frame, score_)
    test_file = open(test_filename, 'w')
    test_file.write(output)
    test_file.close()

    c_output = ''
    for c_ in c_lst:
        c_output += "{}\n".format(c_)
    c_file = open(c_filename, 'w')
    c_file.write(c_output)
    c_file.close()

    b_file = open(b_filename, 'w')
    b_file.write(', '.join([str(b_) for b_ in b_added]))
    b_file.close()

    return test_set

def create_open_discovery_convnet_train(neighbours_data, future_neighbours_data, test_hops, size, neg_ratio, pos_label, neg_label, train_filename, all_data):
    #TODO: add some check of size to make sure function doesn't get stuck in infinite loop

    max_pos_frame_height = 10 #needs placeholder value to start

    negative_cnt = neg_ratio * size
    positive_cnt = size - negative_cnt
    positive_egs = 0
    negative_egs = 0

    track_train_pos = {}
    track_train_neg = {}

    pos_train_frame_amts_total = 0
    pos_train_frame_amts_min = 10000
    pos_train_frame_amts_max = 0
    neg_train_frame_amts_total = 0
    neg_train_frame_amts_min = 10000
    neg_train_frame_amts_max = 0

    train_set = {}
    b_score_lst = {}
    weak_negatives_cnt = 0
    strong_negatives_cnt = 0
    weak_negatives_length_total = 0
    strong_negatives_length_total = 0

    while len(train_set) < size:
        if len(train_set) % 10000 < 10:
            print("{} train examples added at {}. {} positive and {} negative.".format(len(train_set), datetime.now(), positive_egs, negative_egs))

        if positive_egs < positive_cnt:
            for i in range(10):
                #randomly select entitities
                a = random.choice(neighbours_data.keys())
                random_b = random.choice(neighbours_data[a])
                c = random.choice(neighbours_data[random_b])

                if c not in neighbours_data[a] and "{}::{}".format(a, c) not in test_hops:
                    b_lst = list(set(neighbours_data[a]).intersection(neighbours_data[c]))

                    #Start sorting Bs by score
                    sorted_b_lst = []

                    for b in b_lst:
                        if "{}::{}".format(a,b) in all_data:
                            ab = "{}::{}".format(a,b)
                        elif "{}::{}".format(b,a) in all_data:
                            ab = "{}::{}".format(b,a)
                        else:
                            print("ERROR: unknown key: {}::{}".format(a,b))

                        sorted_b_lst.append((b, float(all_data[ab])))
                    sorted_b_lst = sorted(sorted_b_lst, key = lambda x: x[1], reverse=True)
                    sorted_b_lst = [tup_[0] for tup_ in sorted_b_lst]
                    b_lst = sorted_b_lst
                    #End sorting Bs by score

                    #Truncate b list
                    b_lst = b_lst[:75]
                    lst_len = len(b_lst)

                    if lst_len > max_pos_frame_height and lst_len <= 75:
                        max_pos_frame_height = lst_len

                    if lst_len > 0:
                        #Update train data positive examples stats
                        pos_train_frame_amts_total += lst_len
                        if lst_len < pos_train_frame_amts_min:
                            pos_train_frame_amts_min = lst_len
                        if lst_len > pos_train_frame_amts_max:
                            pos_train_frame_amts_max = lst_len

                        conv_frame = ''
                        for ind, b in enumerate(b_lst):
                            conv_frame += "{}::{}::{}".format(a, b, c)
                            if ind < lst_len - 1:
                                conv_frame += "-"

                        #Calculate score for label
                        #TODO: only uses Jaccard; add others
                        score = 0.0
                        a_neighbours = set(neighbours_data[a])
                        c_neighbours = set(neighbours_data[c])
                        score = len(a_neighbours.intersection(c_neighbours)) / float(len(a_neighbours.union(c_neighbours)))

                        #Add completed positive example to training data
                        pos_eg = "{}\t{}\n".format(conv_frame, score)

                        if pos_eg not in train_set and score > 0.0:
                            train_set[pos_eg] = 1
                            positive_egs += 1
                            if "{}::{}".format(a, c) in track_train_pos:
                                track_train_pos["{}::{}".format(a, c)] += 1
                            elif "{}::{}".format(c, a) in track_train_pos:
                                track_train_pos["{}::{}".format(c, a)] += 1
                            else:
                                track_train_pos["{}::{}".format(a, c)] = 1

        if negative_egs < negative_cnt:
            for i in range(10):
                #randomly select entitities
                a = random.choice(neighbours_data.keys())
                c = random.choice(neighbours_data.keys())
                if c not in neighbours_data[a] and "{}::{}".format(a, c) not in test_hops:
                    b_lst = set(neighbours_data[a]).intersection(neighbours_data[c])
                    conv_frame = ""
                    #randomly get Bs which are not in the B list
                    added_window = 0
                    conv_height = random.randrange(1,max_pos_frame_height)

                    #Update train data negative examples stats
                    neg_train_frame_amts_total += conv_height
                    if conv_height < neg_train_frame_amts_min:
                        neg_train_frame_amts_min = conv_height
                    if conv_height > neg_train_frame_amts_max:
                        neg_train_frame_amts_max = conv_height

                    #If possible, add existing Bs to create instance where Bs exist but still no connection ('weak negatives')
                    if len(b_lst) > 0:
                        weak_negatives_cnt += 1
                        weak_negatives_length_total += len(b_lst)
                        for ind, b in enumerate(list(b_lst)[:conv_height]):
                            conv_frame += "{}::{}::{}-".format(a, b, c)
                        conv_frame = conv_frame[:-1]
                    else:
                        strong_negatives_cnt += 1
                        row_amt = random.randrange(1,conv_height+1)
                        strong_negatives_length_total += row_amt
                        while added_window < row_amt:
                            b = random.choice(neighbours_data.keys())
                            if b not in b_lst:
                                conv_frame += "{}::{}::{}".format(a, b, c)
                                if added_window < row_amt - 1:
                                    conv_frame += "-"
                                added_window += 1

                    #Add completed negative example to training data
                    neg_label = 0.0
                    neg_eg = "{}\t{}\n".format(conv_frame, neg_label)
                    if neg_eg not in train_set:
                        train_set[neg_eg] = 1
                        negative_egs += 1
                        if "{}::{}".format(a, c) in track_train_neg:
                            track_train_neg["{}::{}".format(a, c)] += 1
                        elif "{}::{}".format(c, a) in track_train_neg:
                            track_train_neg["{}::{}".format(c, a)] += 1
                        else:
                            track_train_neg["{}::{}".format(a, c)] = 1

    print("\nCompleted: {}. Total train: {}. Positives: {}. Negatives: {} ({} strong with avg length {:.4f}, {} weak with avg length {:.4f})".format(datetime.now(), len(train_set), positive_egs, negative_egs,
                                                                                                                                        strong_negatives_cnt, float(strong_negatives_length_total)/strong_negatives_cnt,
                                                                                                                                        weak_negatives_cnt, float(weak_negatives_length_total)/weak_negatives_cnt))
    #get various useful stats of the train data
    mean_pos = float(sum(track_train_pos.values())) / max(len(track_train_pos.values()), 1)
    mean_neg = float(sum(track_train_neg.values())) / max(len(track_train_neg.values()), 1)
    overlap = set(track_train_pos.keys()).intersection(set(track_train_neg.keys()))
    print("\nMean amount of examples per AC connection- pos: {}, neg: {}. There were {} ACs s both positive and negative examples.".format(mean_pos, mean_neg, len(overlap)))
    print("\nFrame height stats: Positive: Range from {} - {} with a mean of {}. Negative: Range from {} - {} with a mean of {}.".format(pos_train_frame_amts_min, pos_train_frame_amts_max,
                                                            pos_train_frame_amts_total/max(len(track_train_pos), 1), neg_train_frame_amts_min, neg_train_frame_amts_max,
                                                            neg_train_frame_amts_total/max(len(track_train_neg), 1)))

    output = 'train_nodes\tlabel\n'
    for link in train_set:
        output += link
    train_file = open(train_filename, 'w')
    train_file.write(output)
    train_file.close()

    return train_set

def main(argv):
    args = argparser().parse_args(argv[1:])

    A = args.a_node
    C = args.c_node
    cutoff_year = args.cutoff_year
    neg_label = args.labels.split(':')[0]
    pos_label = args.labels.split(':')[1]
    neg_ratio = args.negative_ratio/(args.negative_ratio + 1.0)

    print("Reading data...")
    all_data, neighbours_data, future_neighbours_data = read_data(args.input_file, args.col_labels, args.col_indices, int(cutoff_year))

    #create adjacency matrix
    print("creating adjacency matrix...")
    vertices = create_adjmat(neighbours_data, args.vertices_filename, args.test_graph_filename, all_data)

    print("Setting up {} LBD.".format(args.lbd_method))
    if args.lbd_method == 'closed_discovery_with_aggregators':
        #AB links
        b_set = get_closed_discovery_Bs(neighbours_data, A, C)
        test_hops = create_closed_discovery_test(b_set, A, C, neighbours_data, args.test_filename)
        train_set = create_closed_discovery_train(neighbours_data, test_hops, int(args.train_set_size), neg_ratio, args.train_filename)
    elif args.lbd_method == 'closed_discovery_without_aggregators':
        #ABC links
        b_set = get_closed_discovery_Bs(neighbours_data, A, C)
        test_hops = create_closed_discovery_test_abc(b_set, A, C, neighbours_data, args.test_filename)
        train_set = create_closed_discovery_train_abc(neighbours_data, test_hops, int(args.train_set_size), neg_ratio, args.train_filename)
    elif args.lbd_method == 'open_discovery_with_aggregators_and_accumulators':
        #create AB and BC hops (test)
        test_hops = create_open_discovery_test(neighbours_data, A, args.B_filename, args.C_filename, args.test_filename)
        #create train set
        train_set = create_open_discovery_train(neighbours_data, test_hops, int(args.train_set_size), neg_ratio, args.train_filename)
    elif args.lbd_method == 'open_discovery_without_aggregators_and_accumulators':
        #format test instances for convnet
        test_hops = create_open_discovery_convnet_test(neighbours_data, future_neighbours_data, A, pos_label, args.test_filename, args.C_filename, args.B_filename, all_data)
        #create train set and format for convnet
        train_set = create_open_discovery_convnet_train(neighbours_data, future_neighbours_data, test_hops, int(args.train_set_size), neg_ratio, pos_label, neg_label, args.train_filename, all_data)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
