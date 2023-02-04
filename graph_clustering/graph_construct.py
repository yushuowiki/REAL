import networkx as nx
import json
import sys
import time
from datetime import datetime
import pickle
import operator
import numpy as np
import simplejson as json
import scipy.sparse
from scipy.sparse import csr_matrix

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
        return '({},{})'.format(self.userid)

    def __hash__(self):
        return hash((self.userid))

    def __eq__(self, other):
        return self.userid == other.userid

    def __ne__(self, other):
        return not self.__eq__(other)


def load_graph():

    allusers = {}
    minn = {}
    d = {}
    G = nx.Graph()
    reviewsperproddata = {}
    prodlist = {}
    dictprod = {}
    dictprodr = {}
    count = 0
    filee = open('../data/metadata_zip', 'r')
    for f in filee:

        fsplit = f.split("\t")

        userid = int(fsplit[0])
        prodid = int(fsplit[1])
        rating = int(round(float(fsplit[2])))
        label = int(fsplit[3])

        if userid not in allusers:
            allusers[userid] = []
        if prodid not in allusers[userid]:
            allusers[userid].append(prodid)
        #
        # if int(label) == -1:
        #     fake.add(userid)

        date = fsplit[4].strip()

        if prodid not in d:
            minn[prodid] = 0
            d[prodid] = datetime.strptime(date, "%Y-%m-%d").date()

        minn[prodid] = datetime.strptime(date, "%Y-%m-%d").date()
        if minn[prodid] < d[prodid]:
            d[prodid] = minn[prodid]

        newdate = datetime.strptime(date, "%Y-%m-%d").date()
        datetodays = (newdate - d[prodid]).days
        review = Review(userid, '', prodid, '', rating, label, datetodays)

        nod = str(prodid) + "_" + str(rating)

        if nod not in reviewsperproddata:
            reviewsperproddata[nod] = set()
            dictprod[count] = nod
            dictprodr[nod] = count
            prodlist[nod] = []
            if rating != 3 and rating != 2:
                G.add_node(count)
                count = count + 1

        prodlist[nod].append(review)

    filee.close()


    feature = np.zeros((count, len(allusers)), dtype=float, order='C')
    for i in range(count):
        detail_pr = dictprod[i] #prod_rating
        reviewList = prodlist[detail_pr]
        for review in reviewList:
            feature[i][int(review.userid)-len(minn)] = 1

    print(feature.max())

    with open('../refine_groups/dictprod_zip.pickle', 'wb') as handle:
        pickle.dump(dictprod, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../refine_groups/prodlist_zip.pickle', 'wb') as handle:
        pickle.dump(prodlist, handle, protocol=pickle.HIGHEST_PROTOCOL)

    edgedetails = {}
    cmnrevrsedges = {}
    cmnrevrslist = {}
    cmnrevrslistr = {}
    countt = 0
    graphlist = list(G.nodes())
    cr = {}
    for node1i in range(len(graphlist)):
        node1 = graphlist[node1i]
        if node1 not in cr:
            cr[node1] = []
        for u1i in range(len(prodlist[dictprod[node1]])):
            u1 = prodlist[dictprod[node1]][u1i]#review
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

                        cmnrevrslist[countt] = commonreviewers
                        cmnrevrslistr[commonreviewers] = countt
                        maincount = countt
                        countt = countt + 1
                    else:
                        maincount = cmnrevrslistr[commonreviewers]
                    if node1 < node2:
                        n1 = node1
                        n2 = node2
                    else:
                        n1 = node2
                        n2 = node1
                    # n1<n2
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
    scipy.sparse.save_npz('adjacency_zip.npz',A)
    features = csr_matrix(feature)
    scipy.sparse.save_npz('features_zip.npz', features)

    return A, features

def main():
    load_graph()

if __name__ == "__main__":
    main()