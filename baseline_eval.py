# @author Simon Baker
#
# This script evaluates time slicing (for future co-occurance) on all LION metric combinations for open discovery.
# the evaluation metric is average rank for all corresponding future direct C node conections, given a sample of A nodes
#
from __future__ import print_function
import sys
import ujson
import numpy as np

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

DATA_PATH = "/srv/lion/link_prediction_for_lbd/time_slicing/test.tsv"

memi = GraphProxy('http://127.0.0.1:8081/graph')
composite_functions=memi.get_aggregation_functions()
accumulation_functions = memi.get_accumulation_functions()
metrics = memi.get_metrics()
from random import shuffle
def read_data(path):
    data_dict = {}
    lines = open(path).readlines()
    for line in lines[0:1000]:
        splits = line.strip().split("\t")
        a_id = splits[0]
        c_list = splits[1].split(";")
        data_dict[a_id] = c_list
    return data_dict

data = read_data(DATA_PATH)

def calculate_map(a_id, comp, accu, met, c_list):
    print("calculating for: {}, {}, {}".format(met,comp,accu))
    results =memi.open_discovery(a_id, met, comp, accu, year=2009)
    shuffle(results)
    positives = [results[0].index(c_id) for c_id in c_list]
    y=np.zeros(len(results[0]))
    y[positives] = 1
    scores = np.array(results[1])
    map = average_precision_score(y,scores,average="micro")
    return map

#------------ main loop ----------------
final_res = {}
for accu in accumulation_functions:
    final_res.setdefault(accu,{})
    for comp in composite_functions:
        final_res[accu].setdefault(comp,{})
        for met in metrics:
            values = []
            for a_id in data.keys():
                values.append(calculate_map (a_id, comp, accu, met, data[a_id]))
                average_map = np.average(np.array(values))
            final_res[accu][comp].setdefault(met, {})
            final_res[accu][comp][met].setdefault("values", values)
            final_res[accu][comp][met].setdefault("average", average_map)
open("./baseline_eval_results.json",'w').write(ujson.dumps(final_res,indent=4))
print("done!")
sys.exit(0)
