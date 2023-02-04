import networkx as nx
import json
import sys
import time
import ast
import pickle
import numpy as np
from datetime import datetime
import math
import re
from scipy import sparse
from collections import Counter
from scipy.sparse import csr_matrix
import argparse
from numpy import *


class Groups:
    def __init__(self, users, prods,id):
        self.users = users
        self.prods = prods
        self.id = id
        #self.ratings = ratings

    def __lt__(self, other):
        return len(self.users) < len(other.users)


class Review:
    def __init__(self, userid, useridmapped, prodid, prodidmapped, rating, label, date):#concontent):
        self.userid = userid
        self.useridmapped = useridmapped
        self.prodid = prodid
        self.prodidmapped = prodidmapped
        self.rating = rating
        self.label = label
        self.date = date

    def __repr__(self):
        return '({})'.format(self.prodid)

    def __hash__(self):
        return hash(self.prodid)

    def __eq__(self, other):
        return self.prodid == other.prodid

    def __ne__(self, other):
        return not self.__eq__(other)




allprods = {}
allusers = {}
reviewtime = {}
reviewrating = {}
reviewcontent = {}
wholerev = {}
minn = {}
d = {}
fake = set()
rvdate = {}
maxrvdate = {}
maxrvcon = {}
c = 0


# filee = open('../data/meta_chi', 'r')
# for f in filee:
#
#     fsplit = f.split(" ")
#     if len(fsplit) != 9:
#         break
#     userid = fsplit[2]
#     prodid = fsplit[3]
#     rating = int(round(float(fsplit[8])))
#     label = fsplit[5]
#
#     if label == 'Y':
#         fake.add(userid)
#         label = -1
#     else:
#         label = 1
#
#     date = fsplit[0].strip()
#     date = datetime.strptime(date, "%m/%d/%Y").date()
filee = open("../data/metadata_zip", 'r')
for f in filee:

    fsplit = f.split("\t")

    userid = int(fsplit[0])
    prodid = int(fsplit[1])
    rating = int(round(float(fsplit[2])))
    label = fsplit[3]

    if int(label) == -1:
        fake.add(userid)

    date = fsplit[4].strip()
    date = datetime.strptime(date, "%Y-%m-%d").date()

    if prodid not in d:
        minn[prodid] = 0
        d[prodid] = date

    minn[prodid] = date
    if minn[prodid] < d[prodid]:
        d[prodid] = minn[prodid]#d records the earliest time

    if userid not in rvdate:
        rvdate[userid] = {}
        maxrvdate[userid] = {}
        maxrvcon[userid] = {}
    if prodid not in rvdate[userid]:
        rvdate[userid][prodid] = date
        maxrvdate[userid][prodid] = date
        #maxrvcon[userid][prodid] = text[c]

    rvdate[userid][prodid] = date
    if rvdate[userid][prodid] > maxrvdate[userid][prodid]:
        maxrvdate[userid][prodid] = rvdate[userid][prodid]
        #maxrvcon[userid][prodid] = text[c]
    c = c + 1
    #maxrvdate:最近的评价时间
filee.close()

c = 0
userId, prodId, dates, ratings = [], [], [], []
# filee = open('../data/meta_chi', 'r')
# for f in filee:
#
#     fsplit = f.split(" ")
#     if len(fsplit) != 9: break
#     userid = fsplit[2]
#     prodid = fsplit[3]
#     rating = int(round(float(fsplit[8])))
#     label = fsplit[4]
#     date = fsplit[0].strip()
#
#     if label == 'Y':
#         label = -1
#     else:
#         label = 1
#
#     newdate = datetime.strptime(date, "%m/%d/%Y").date()
filee = open("../data/metadata_zip", 'r')
for f in filee:

    fsplit = f.split("\t")

    userid = int(fsplit[0])
    prodid = int(fsplit[1])
    rating = int(round(float(fsplit[2])))
    label = fsplit[3]

    userId.append(userid)
    prodId.append(prodid)
    #dates.append()
    ratings.append(rating)

    if int(label) == -1:
        fake.add(userid)

    date = fsplit[4].strip()
    dates.append(date)
    newdate = datetime.strptime(date, "%Y-%m-%d").date()
    if newdate == maxrvdate[userid][prodid]:

        datetodays = (newdate - d[prodid]).days

        r = Review(userid, '', prodid, '', rating, label, datetodays)#, maxrvcon[userid][prodid])

        if userid not in reviewtime:
            reviewtime[userid] = {}
        if prodid not in reviewtime[userid]:
            reviewtime[userid][prodid] = datetodays#reviewtime
        if userid not in reviewrating:
            reviewrating[userid] = {}
        if prodid not in reviewrating[userid]:
            reviewrating[userid][prodid] = rating
        if userid not in allusers:
            allusers[userid] = []
        if prodid not in allprods:
            allprods[prodid] = []
        if userid not in wholerev:
            wholerev[userid] = {}
        if prodid not in wholerev[userid]:
            wholerev[userid][prodid] = r

        #np.save('reviewrating.npy', reviewrating)

        allprods[prodid].append(userid)
        allusers[userid].append(prodid)

        c = c + 1
filee.close()


def reviewtightness(group, L):
    v = 0
    for user in group.users:
        for prod in group.prods:
            # prod=prod.split("_")[0]
           # print(prod,' ',user)
            if prod in reviewtime[user]:
                v = v + 1
    if len(list(group.prods)) == 0:
        return 0
    return (v * L) / (1.0 * len(list(group.users)) * len(list(group.prods)))


def neighbortightness(group, L):
    userlist = list(group.users)
    denom = 0
    num = 0
    for user1i in range(len(userlist)):
        user1 = userlist[user1i]
        for user2i in range(user1i + 1, len(userlist)):
            user2 = userlist[user2i]
            union = set(allusers[user1]).union(set(allusers[user2]))
            intersection = set(allusers[user1]).intersection(set(allusers[user2]))
            num = num + len(intersection) / (len(union) * 1.0)
            denom = denom + 1
    print("neighbor tightness:", (num * L) / (1.0 * denom))
    return (num * L) / (1.0 * denom)


def producttightness(group):
    c = 0
    userlist = list(group.users)
    for user in userlist:
        if c == 0:
            intersection = set(allusers[user])
            union = set(allusers[user])
        else:
            intersection = intersection.intersection(set(allusers[user]))
            union = union.union(set(allusers[user]))
        c = c + 1
    print("prod tight:",len(intersection) / (len(union) * 1.0))
    return len(intersection) / (len(union) * 1.0)

'''
def productreviewerratio(group):
    maxx = 0

    for prod in group.prods:

        num = 0
        denom = 0
        for user in group.users:
            if prod in reviewtime[user]:
                num = num + 1  #

        for r in allprods[prod]:
            # if int(r.rating)==int(prod.split("_")[1]):
            denom = denom + 1

        ans = num / (1.0 * denom)
        if ans > maxx:
            maxx = ans
    print("prratio:",maxx)
    return maxx
'''

def groupsize(group):
    return 1 / (1 + math.exp(3 - len(list(group.users))))


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    word = re.compile(r'\w+')
    words = word.findall(text)
    return Counter(words)





#
def averagetimewindow_ratingvariance(group, L):
    if len(group.prods) == 0:
        return 0, 0
    avg = 0
    var = 0
    for prod in group.prods:
        prodlist = []
        prodtym = []
        # prod=prod.split("_")[0]
        minn = float('inf')
        maxx = 0
        for user in group.users:
            if prod in reviewtime[user]:
                prodlist.append(reviewrating[user][prod])
                prodtym.append(reviewtime[user][prod])

        var = var + np.var(prodlist)#方差
        ans = np.std(prodtym)
        if ans < 30:
            avg = avg + (1 - ans / 30.0)

    var = var / (-1.0 * len(group.prods))
    rating_variance = 2 * (1 - (1.0 / (1 + math.exp(var))))
    print("windows")
    return (avg * L) / (1.0 * len(group.prods))


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def RD_prod(prod,rat):
    """Rating deviation of products.
    Parameters
    ----------
    prodId : (R,) array of int
            product Ids.
    ratings : (R,) array of float
            ratings of reviews.
    Returns
    -------
    RD : (P,) array of float
            Rating deviation of products.
    """
    uprod, ind_prods = np.unique(prod, return_inverse=True)
    avgRating = np.zeros((len(rat),))
    for i in range(len(uprod)):
        ind = ind_prods == i
        if any(ind):
            r = np.array(rat)[ind]
            avgRating[ind_prods == i] = np.sum(r) / len(r)
    print("rd")
    rd = np.fabs(rat - avgRating)
    return normalization(rd)

def avgRD(group):
    """Avg. rating deviation of users/products.
    """
    # uuser, ind_users = np.unique(userId, return_inverse=True)
    # #uprod, ind_prods = np.unique(prodId, return_inverse=True)
    # #RD = RD_prod(prodId, ratings)
    # avgRD_user = np.zeros((len(uuser),))
    # for i in range(len((uuser))):
    #     ind = ind_users==i
    #     if any(ind):
    #         r = np.array(RD)[ind]
    #         avgRD_user[i] = np.sum(r)/len(r)

    indx = []
    for guser in group.users:
        indx += list(i for i in range(len(uuser)) if uuser[i] == guser)
    ard = np.array(avgRD_user)[indx]
    result = mean(ard)
    print("ard:", result)
    return result


def BST_user(user, dat):
    """Burstiness of users.
    Parameters
    ----------
    userId : (R,) array of int
            user Ids.
    dates : (R,) list of datetime.datetime
            date of reviews.
    Returns
    -------
    BST_user : (U,) array of float
            burstiness of users.
    """
    uuser, ind_users = np.unique(user, return_inverse=True)
    int_dates = np.array([datetime.strptime(date, '%Y-%m-%d').toordinal() for date in dat])
    tau = 30
    BST_user = np.ones((len(uuser),))
    for i in range(len(uuser)):
        ind = ind_users==i
        if np.sum(ind) > 1:
            dt = int_dates[ind]
            ndays = dt.max() - dt.min()
            if ndays > tau:
                BST_user[i] = 0
            else:
                BST_user[i] = 1 - ndays/tau
    print("bst")
    return BST_user

def BST(group):
    #uuser, ind_users = np.unique(userId, return_inverse=True)
    indx = []
    for guser in group.users:
        indx += list(i for i in range(len(uuser)) if uuser[i] == guser)
    #user = np.array(userId)[indx]

    gbst = np.array(bst)[indx]
    result = mean(gbst)
    print("gbst:", result)
    return result


def calc_score(g, Lsub):
    #score = []
    ans = averagetimewindow_ratingvariance(g, Lsub)
    rt = reviewtightness(g, Lsub)
    nt = neighbortightness(g, Lsub)
    pt = producttightness(g)
    tw0 = ans
    gbst = BST(g)
    ard = avgRD(g)#MNR_group(g)]#productreviewerratio(g),
    #gmnr = (MNR_group(g) - 0.068) / 0.02
    #anslist, rtlist, ntlist, ptlist,tw0list,tw1list,gbstlist,ardlist = [],[],[],[],[],[],[]
    scored['rtd'][g.id],scored['ntd'][g.id],scored['ptd'][g.id],scored['tw0d'][g.id],\
        scored['gbstd'][g.id],scored['ardd'][g.id]=\
        rt,nt,pt,tw0,gbst,ard
    score = [(rt-0.95)/0.17, (nt-0.3)/0.16, (pt*6-0.06)/0.225, (tw0-0.015)/0.077,  (gbst*6-2.6)/1.07, (ard*4-0.85)/0.56]
    total = np.array([pow(i, 2) for i in score])
    collu = sqrt(np.sum(total)*1.0/len(score))
    return score, collu


def create_groups():

    clusters = np.load("../graph_clustering/clusters_zip01.npy")
    x = 0
    v = 5
# grps[len(grps)]={'id':len(grps),'users':list(userset),'prods':list(prodset),'scorepred':0, 'scoregt':0, 'fakegt':0,'fakepred':0}
    fgrps = {}
    for i in range(np.max(clusters)):
        fgrps[i] = {}
        fgrps[i]['id'] = i#count
        fgrps[i]['users'] = set()
        fgrps[i]['prods'] = set()
        fgrps[i]['ratings'] = set()
        fgrps[i]['scorepred'] = 0
        fgrps[i]['scoregt'] = 0
        fgrps[i]['fakegt'] = 0

    with open('dictprod_zip.pickle', 'rb') as handle:
        dictprod = pickle.load(handle)#count ---prodid_rating

    with open('prodlist_zip.pickle', 'rb') as handle:
        prodlist = pickle.load(handle)#pro_rating---review

    for i in range(np.max(clusters)):#for every cluster
        userset = set()
        for j in range(len(clusters)):# for every prod_rating

            if clusters[j] == i:#i--组号 j--prod_rating号
                    prod_str = dictprod[j]
                    strlist = prod_str.strip().split('_')
                    fgrps[i]['prods'].add(int(strlist[0]))
                    fgrps[i]['ratings'].add(int(strlist[1]))
                    revlist = prodlist[prod_str]
                    cusers = set(rev.userid for rev in revlist)
                    if len(userset) == 0:
                        userset = cusers
                    else:
                        userset = userset & cusers

        fgrps[i]['users'] = userset
        fgrps[i]['prods'] = list(fgrps[i]['prods'])
        fgrps[i]['users'] = list(fgrps[i]['users'])

    # with open("./fgrps", 'w') as fp:
    #     json.dump(fgrps, fp)
    counter = 0
    for grp in fgrps:
        if len(fgrps[grp]['users'])>4:

            group = Groups(fgrps[grp]['users'], fgrps[grp]['prods'],fgrps[grp]['id'])

            Lsub = 1.0 / (1 + (math.exp(3 - len(list(group.users)) - len(list(group.prods)))))#punction
            #smnr = (MNR_group(group)-0.068)/0.02
            # out[str(grp)]['scorepred'].append(smnr)
            # scorepred = out[str(grp)]['scorepred']
            # spamicity = np.mean(np.array(scorepred))

            ans = calc_score(group, Lsub)
            scorepred = ans[0]
            spamicity = ans[1]

            c = 0#fakers in the group
            denom = 1
            for u in group.users:
                 if u in fake:
                     c = c + 1
                 denom = denom + 1#users数量
            store = (c * 1.0) / denom

            c = 0#虚假评论数
            denom = 1#总评论数
            for u in group.users:
                 for p in group.prods:
                     if p in wholerev[u]:
                         if int(wholerev[u][p].label) == -1:
                             c = c + 1
                         denom = denom + 1
            freview = (c * 1.0) / denom

            grp = int(grp)
            if grp not in grps:

                grps[x] = {'id': grp, 'users': list(group.users), 'prods': list(group.prods),'scorepred': scorepred,
                           'scoregt': store,  'fakegt': freview,
                           'fakepred': spamicity}
                x = x + 1


    for grp in grps:

        summ = grps[grp]['fakepred']
        if summ > 0.2:
            grps2[len(grps2)] = grps[grp]
            grps2[len(grps2) - 1]['id'] = len(grps2) - 1
            #grps2[len(grps2) - 1]['summ'] = summ


grps = {}
grps2 = {}

RD = RD_prod(prodId, ratings)

uuser, ind_users = np.unique(userId, return_inverse=True)
avgRD_user = np.zeros((len(uuser),))
for i in range(len((uuser))):
    ind = ind_users == i
    if any(ind):
        r = np.array(RD)[ind]
        avgRD_user[i] = np.sum(r) / len(r)

bst = BST_user(userId, dates)

# mnr = MNR(userId, dates)

scored = {}
scored['rtd'], scored['ntd'], scored['ptd'], scored['tw0d'], scored['gbstd'], scored['ardd']= {},{},{},{},{},{}
create_groups()
with open("./outgrps_zip", 'w') as fp:
    json.dump(grps, fp)

with open("./grpscore_zip", 'w') as fp:
    json.dump(grps2, fp)

with open("./scoredict_zip", 'w') as fp:
    json.dump(scored, fp)


print('end')
