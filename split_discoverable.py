import ujson
import numpy as np
from random import randint


path = "/home/sb895/discoverable_edges_complete_2009.json"

data = ujson.loads(open(path).read())

a2c = {}

for d in data:
    a2c.setdefault(d["A"],set([])).add(d["C"])

lengths = np.array([len(v) for v in a2c.values()])
print str(len(a2c))
print "max: {}".format(np.max(lengths))
print "min: {}".format(np.min(lengths))
print "mean: {}".format(np.mean(lengths))
print "meadian: {}".format(np.median(lengths))


def writeToFile(dict,fpath):
    f = open(fpath,'w')
    for k in dict:
        v =  ";".join(dict[k])
        f.write(k + "\t" + v +"\n")

#a_list = a2c.keys()
dev = {}
test = {}

for k in a2c:
    r =randint(0, 2)
    if r == 0:
        dev[k] = a2c[k]
    else:
        test[k] = a2c[k]

print str(len(dev))
print str(len(test))
