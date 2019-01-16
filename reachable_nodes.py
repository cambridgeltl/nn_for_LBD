# @author Simon Baker
#
# This script returns all "reachable nodes" for a given a node. this is to be used as part of time slicing.
# the returned results will be all reachable A-B-C chains.  This input is a list of A-nodes.
#
from __future__ import print_function
import sys
import ujson

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

def read_data(path):
    data_dict = {}
    lines = open(path).readlines()
    for line in lines[0:100]:
        splits = line.strip().split("\t")
        a_id = splits[0]
        c_list = splits[1].split(";")
        data_dict[a_id] = c_list
    return data_dict

data = read_data(DATA_PATH)

def get_c_nodes(a_id,year):
    return memi.open_discovery(a_id, "count", "max", "max", year=year, limit=None,offset=0)[0]

def get_b_nodes(a_id, c_id,year):
    return memi.closed_discovery(a_id, c_id, metric="count",agg_func='avg', year=year,limit=None,offset=0)[0]

def get_reachable(A_list, year=2009):
    results = {} # dictionary (a_id,c_id) -> [b_ids]
    for a_id in A_list:
        c_nodes = get_c_nodes(a_id,year)
        for c_id in c_nodes:
            results[(a_id, c_id)] = get_b_nodes(a_id,c_id,year)
    return results

results = get_reachable(data.keys())
open("./reachable_top_100_save.json",'w').write(ujson.dumps(results,indent=4))
print("done!")
sys.exit(0)
