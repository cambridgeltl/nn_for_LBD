# @author Simon Baker
#
# This script evaluates time slicing (for future co-occurance) on all LION metric combinations for open discovery.
# the evaluation metric is average rank for all corresponding future direct C node conections, given a sample of A nodes
#
from __future__ import print_function
import sys
import ujson
import numpy as np
import random

from collections import Counter

from sklearn.metrics import average_precision_score
try:
    import lionlbd.graphdb as graphdb
    from lionlbd.proxyinterface import GraphProxy
except ImportError:
    print("""
ERROR: Failed to import neo4jInterface.py. This script is found in the
    lion-lbd repository and should be copied or symlinked to this directory
    along with the file config.py. Try e.g. the following:

    git clone git@github.com:cambridgeltl/lion-lbd.git
    ln -s lion-lbd/neo4jInterface.py
    ln -s lion-lbd/config.py
""",file=sys.stderr)
    raise

DATA_PATH = "/srv/lion/link_prediction_for_lbd/biogrid/2016_test.tsv"

memi = GraphProxy('http://127.0.0.1:8081/graph')
composite_functions= memi.get_aggregation_functions()
accumulation_functions = memi.get_accumulation_functions()
metrics = memi.get_metrics()
from random import shuffle
def read_data(path):
    data_dict = {}
    lines = open(path).readlines()
    for line in lines[:1000]:
        splits = line.strip().split("\t")
        a_id = splits[0]
        c_list = splits[1].split(";")
        data_dict[a_id] = c_list
    return data_dict

data = read_data(DATA_PATH)


def calculate_map(a_id, comp, accu, met, c_list):
    print("calculating for: {}, {}, {}".format(met,comp,accu))
    results =memi.open_discovery(a_id, met, comp, accu, year=2016)
    positives = [results[0].index(c_id) for c_id in c_list]
    y=np.zeros(len(results[0]))
    y[positives] = 1
    scores = np.array(results[1])
    map = average_precision_score(y,scores,average="micro")
    return map, len(positives), len(y), scores, positives, y

#------------ main loop ----------------
final_res = {}
res_mr = {}
res_mrr = {}
res_hr = {}
baseline_map_output = ""
baseline_mr_output = ""
baseline_mrr_output = ""
baseline_har_output = ""
small_scores = ""
for accu in accumulation_functions:
    final_res.setdefault(accu,{})
    res_mr.setdefault(accu,{})
    res_mrr.setdefault(accu,{})
    res_hr.setdefault(accu,{})
    for comp in composite_functions:
        final_res[accu].setdefault(comp,{})
    	res_mr[accu].setdefault(comp,{})
    	res_mrr[accu].setdefault(comp,{})
    	res_hr[accu].setdefault(comp,{})
        for met in metrics:
            values = []
            mr_values = []
            mrr_values = []
            h_at_r_values = []
            for a_ind, a_id in enumerate(data.keys()):
                ap, tp, c_cnt, scores, true_inds, y_true = calculate_map(a_id, comp, accu, met, data[a_id])
                scores_deets = "y score: {}".format(scores)
                values.append(ap)
                average_map = np.average(np.array(values))

                true_scores = [scores[ind] for ind in true_inds]
                sorted_scores = sorted(scores, reverse=True)
                true_ranks = []
                scores_cnter = Counter(scores)
                for ts in true_scores:
                    ts_index = sorted_scores.index(ts)
                    true_ranks.append((( ((ts_index + 1) + (ts_index + scores_cnter[ts]))/2 ), ts))
                mr = np.mean([tr_[0] for tr_ in true_ranks])
                mrr = np.mean([1.0/tr_[0] for tr_ in true_ranks])
                hits_at_R = len([tup_[0] for tup_ in sorted(true_ranks, key= lambda x: x[0]) if tup_[0] <= len(true_ranks)])/float(len(true_ranks))

                mr_values.append(mr)
                mrr_values.append(mrr)
                h_at_r_values.append(hits_at_R)

                ap_out = "{}. A: {}, AP: {}. MR: {}. MRR: {}. Hits at R: {}. TP: {}/{}. \nRanks: {}\n\n".format(a_ind + 1, a_id, ap, mr, mrr, hits_at_R, tp, c_cnt, sorted(true_ranks, key= lambda x: x[0]))
                baseline_map_output += ap_out
                mr_out = "{}. A: {}, MR: {}. TP: {}/{}. \nRanks: {}\n\n".format(a_ind + 1, a_id, mr, tp, c_cnt, sorted(true_ranks, key= lambda x: x[0]))
                baseline_mr_output += mr_out
                mrr_out = "{}. A: {}, MRR: {}. TP: {}/{}. \nRanks: {}\n\n".format(a_ind + 1, a_id, mrr, tp, c_cnt, sorted(true_ranks, key= lambda x: x[0]))
                baseline_mrr_output += mrr_out
                har_out = "{}. A: {}, Hits at R: {}. TP: {}/{}. \nRanks: {}\n\n".format(a_ind + 1, a_id, hits_at_R, tp, c_cnt, sorted(true_ranks, key= lambda x: x[0]))
                baseline_har_output += har_out

                if c_cnt < 150:
                	small_scores += "\n{}\ny_true: {}\ny_score: {}\n".format(ap_out, y_true, scores_deets)

    	    average_mr = np.average(np.array(mr_values))
    	    average_mrr = np.average(np.array(mrr_values))
    	    average_h_at_r = np.average(np.array(h_at_r_values))

            final_res[accu][comp].setdefault(met, {})
            final_res[accu][comp][met].setdefault("values", values)
            final_res[accu][comp][met].setdefault("average", average_map)
            baseline_map_o = "Mean MAP was: {}.".format(average_map)
    	    baseline_map_output = "\n\n{}-{}-{}\n{}\n\n{}".format(accu, comp, met, baseline_map_o, baseline_map_output)

            res_mr[accu][comp].setdefault(met, {})
            res_mr[accu][comp][met].setdefault("values", mr_values)
            res_mr[accu][comp][met].setdefault("average", average_mr)
            baseline_mr_o = "Mean MR was: {}.".format(average_mr)
            baseline_mr_output = "\n\n{}-{}-{}\n{}\n\n{}".format(accu, comp, met, baseline_mr_o, baseline_mr_output)

            res_mrr[accu][comp].setdefault(met, {})
            res_mrr[accu][comp][met].setdefault("values", mrr_values)
            res_mrr[accu][comp][met].setdefault("average", average_mrr)
            baseline_mrr_o = "Mean MRR was: {}.".format(average_mrr)
            baseline_mrr_output = "\n\n{}-{}-{}\n{}\n\n{}".format(accu, comp, met, baseline_mrr_o, baseline_mrr_output)

            res_hr[accu][comp].setdefault(met, {})
            res_hr[accu][comp][met].setdefault("values", h_at_r_values)
            res_hr[accu][comp][met].setdefault("average", average_h_at_r)
            baseline_hr_o = "Mean Hits at R was: {}.".format(average_mr)
            baseline_hr_output = "\n\n{}-{}-{}\n{}\n\n{}".format(accu, comp, met, baseline_hr_o, baseline_har_output)


with open("Baselines-MAP_devel.txt", 'w') as fil:
    fil.write(baseline_map_output)


with open("Baselines-MR_devel.txt", 'w') as fil2:
    fil2.write(baseline_mr_output)

with open("Baselines-MRR_devel.txt", 'w') as fil3:
    fil3.write(baseline_mrr_output)

with open("Baselines-H@R_devel.txt", 'w') as fil4:
    fil4.write(baseline_har_output)

with open("Baselines-MAP_devel-scores.txt", 'w') as fil5:
    fil5.write(small_scores)


open("./biogrid_baseline_eval_1000_MAP_results.json",'w').write(ujson.dumps(final_res,indent=4))
open("./biogrid_baseline_eval_1000_MR_results.json",'w').write(ujson.dumps(res_mr,indent=4))
open("./biogrid_baseline_eval_1000_MRR_results.json",'w').write(ujson.dumps(res_mrr,indent=4))
open("./biogrid_baseline_eval_1000_H@R_results.json",'w').write(ujson.dumps(res_hr,indent=4))
print("done!")
sys.exit(0)
