import networkx as nx
import json
from datetime import datetime
import operator
import argparse
import scipy.sparse
from scipy.sparse import csr_matrix

parser = argparse.ArgumentParser(description='Detecting groups')
parser.add_argument('--metadata', help='path to metadata')
parser.add_argument('--rc', help='path to reviewContent')
parser.add_argument('--dg', help='path to detected groups')

args = parser.parse_args()

CC_mapper = {}


class Review:
    def __init__(self, userid, useridmapped, prodid, prodidmapped, rating, label, date):
        self.userid = userid
        self.useridmapped = useridmapped
        self.prodid = prodid
        self.prodidmapped = prodidmapped
        self.rating = rating
        self.label = label
        self.date = date

    def __repr__(self):
        return '({})'.format(self.userid)

    # return '({},{})'.format(self.userid)

    def __hash__(self):
        return hash((self.userid))

    def __eq__(self, other):
        return self.userid == other.userid

    def __ne__(self, other):
        return not self.__eq__(other)


def isAdjacent(e1, e2):
    if e1[0] == e2[0] or e1[1] == e2[0] or e1[0] == e2[1] or e1[1] == e2[1]:
        return True
    return False


def degree(G, edge):
    return G.degree[edge[0]] + G.degree[edge[1]] - 1


def canbeincluded(userset):
    if len(userset) == 0:
        return 0
    union = set()
    intersection = allusers[list(userset)[0]]
    for u in userset:
        union = union.union(allusers[u])
        intersection = intersection.intersection(allusers[u])
        jaccard = len(intersection) / len(union) * 1.0
        if jaccard > 0.5:
            return 1
        return 0


minn = {}
d = {}
fake = set()
# count=0
filee = open('meta_chi.txt')
# while count < 67394:
for f in filee:
    fsplit = f.strip().split(' ')
    # print count
    if len(fsplit) == 9:

        userid = fsplit[2]
        prodid = fsplit[3]
        # rating=int(round(float(fsplit[8])))
        label = fsplit[4]

        if label == 'Y':
            fake.add(userid)

        date = fsplit[0].strip()
        if prodid not in d:
            minn[prodid] = 0
            d[prodid] = datetime.strptime(date, "%m/%d/%Y").date()

        minn[prodid] = datetime.strptime(date, "%m/%d/%Y").date()
        if minn[prodid] < d[prodid]:
            d[prodid] = minn[prodid]

filee.close()

G = nx.Graph()
reviewsperproddata = {}
nodedetails = {}
prodlist = {}
dictprod = {}
dictprodr = {}
mainnodelist = set()
count = 0
filee = open('meta_chi.txt', 'r')
for f in filee:

    fsplit = f.strip().split(" ")
    if len(fsplit) == 1: break
    userid = fsplit[2]
    prodid = fsplit[3]
    rating = str(int(round(float(fsplit[8]))))
    label = fsplit[4]
    date = fsplit[0].strip()
    newdate = datetime.strptime(date, "%m/%d/%Y").date()
    datetodays = (newdate - d[prodid]).days
    if label == 'Y':
        label = -1
    else:
        label = 1
    review = Review(userid, '', prodid, '', rating, label, datetodays)

    if prodid + "_" + rating not in reviewsperproddata:
        count = count + 1
        reviewsperproddata[prodid + "_" + rating] = set()
        dictprod[count] = prodid + "_" + rating
        dictprodr[prodid + "_" + rating] = count
        prodlist[prodid + "_" + rating] = []
        G.add_node(count)

    prodlist[prodid + "_" + rating].append(review)

    reviewsperproddata[prodid + "_" + rating].add(review)
filee.close()

edgedetails = {}
cmnrevrsedges = {}
cmnrevrslist = {}
cmnrevrslistr = {}
cmnrevrsedgeslen = {}
countt = 0
visited = {}
mark = {}
graphlist = list(G.nodes())
cr = {}
for node1i in range(len(graphlist)):
    node1 = graphlist[node1i]
    if node1 not in cr:
        cr[node1] = []
    for u1i in range(len(prodlist[dictprod[node1]])):
        u1 = prodlist[dictprod[node1]][u1i]
        cr11 = set()
        cr11.add(u1)
        for u2i in range(u1i + 1, len(prodlist[dictprod[node1]])):
            u2 = prodlist[dictprod[node1]][u2i]
            if abs(u1.date - u2.date) < 10:
                cr11.add(u2)
        cr[node1].append(cr11)
    cr[node1].sort(key=len, reverse=True)

edgecount = {}
for node1i in range(len(graphlist)):

    node1 = graphlist[node1i]
    for node2i in range(node1i + 1, len(graphlist)):
        node2 = graphlist[node2i]

        maxx = 0
        maxxcr = set()

        cr1 = cr[node1]
        cr2 = cr[node2]
        crlist = set()
        f = 0
        for cri1 in cr1:
            if len(cri1) < 2:
                break
            for cri2 in cr2:
                if len(cri2) < 2:
                    f = 1
                    break
                crr = cri1.intersection(cri2)
                crr = frozenset(crr)
                if len(crr) > 1:
                    crlist.add(crr)

            if f == 1:
                break

        crlist = list(crlist)
        crlist.sort(key=len, reverse=True)

        for commonreviewers in crlist:
            if len(commonreviewers) > 1:

                if commonreviewers not in cmnrevrslistr:
                    countt = countt + 1
                    cmnrevrslist[countt] = commonreviewers
                    cmnrevrslistr[commonreviewers] = countt
                    maincount = countt
                else:
                    maincount = cmnrevrslistr[commonreviewers]
                if node1 < node2:
                    n1 = node1
                    n2 = node2
                else:
                    n1 = node2
                    n2 = node1

                if maincount not in cmnrevrsedges:
                    cmnrevrsedges[maincount] = []

                if (n1, n2) not in edgecount:
                    edgecount[(n1, n2)] = 0
                    G.add_edge(n1, n2)
                    edgedetails[(n1, n2)] = crlist

                if (n1, n2) not in cmnrevrsedges[maincount]:
                    cmnrevrsedges[maincount].append((n1, n2))
                    edgecount[(n1, n2)] = edgecount[(n1, n2)] + 1

A = nx.adjacency_matrix(G)
scipy.sparse.save_npz('adjacency_nyc.npz',A)
#nx.adjacency: return scipy sparse

#convert feature matrix to scipy sparse
features = csr_matrix(features)
scipy.sparse.save_npz('features_nyc.npz', features)

print('end')