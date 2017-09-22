#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: model.py
Author: dmjvictory(dmjvictory@163.com)
Date: 2017/09/18 17:06:15
"""

import os
import re
import sys
import thulac
import collections
import numpy as np
from scipy import sparse
from gensim import models
from gensim import corpora
from sklearn import cluster
from sklearn import metrics
from gensim import matutils


def unclassify():
    thu = thulac.thulac(filt=True, T2S=True)
    site = set([l.strip() for l in open('classified.site').readlines()])
    for line in open('domain.descr2').readlines():
        item = line.strip('\n').split('\t')
        if re.search(u'游戏', item[2].decode('utf8')) and item[0] not in site:
            print line.strip()


def preprocess(load=False):
    if load:
        #dictionary = corpora.Dictionary.load('word_id.dict')
        corpus = corpora.MmCorpus('ps_domain_keyword.mm')
        docs = models.TfidfModel(corpus)[corpus]
        info = open('ps_domain_keyword.detail').readlines()
        return corpus2csr(docs), np.array(info)

    X = []
    info = []
    thu = thulac.thulac(filt=True, T2S=True)
    counter = collections.Counter()

    for line in open('domain.descr').readlines():
        data = []
        item = line.strip('\n').split('\t')
        
        if item[0].startswith('ximalaya'):
            print len(info)

        if 1:#re.search(u'游戏', item[2].decode('utf8')):
            for word in re.split(',|', item[2]) + re.split(',|', item[3]):
                if word:
                    word = word.lower()
                    if len(word) < 13:
                        data.append(word.decode('utf8'))
                    else:
                        child_list = thu.cut(word, text=True).split()
                        for child in set(child_list):
                            try:
                                child_query, child_tag = child.split('_')
                                if child_tag in ['n', 'v', 'ns', 'ni', 'nz', 'np']:
                                    data.append(child_query.decode('utf8'))
                            except Exception as e:
                                print e
                                continue
        if data:
            X.append(data)
            info.append(line.strip())

    dictionary = corpora.Dictionary(X)
    dictionary.filter_extremes(no_below=3)
    corpus = [dictionary.doc2bow(text) for text in X]
    
    #dictionary.save('word_id.dict')
    corpora.MmCorpus.serialize('ps_domain_keyword.mm', corpus)
    f = open('ps_domain_keyword.detail', 'w')
    print >> f, '\n'.join(info)
    docs = models.TfidfModel(corpus)[corpus]
    return corpus2csr(docs), np.array(info)


def corpus2csr(corpus):
    data = []
    rows = []
    cols = []
    line_count = 0
    for _ in corpus:
        for elem in _:
            rows.append(line_count)
            cols.append(elem[0])
            data.append(elem[1])
        line_count += 1
    csr = sparse.csr_matrix((data, (rows,cols)))
    return csr


def db_cluster():
    docs, info = preprocess()
    clf = cluster.DBSCAN(eps=0.5, min_samples=5, metric='cosine')
    clf.fit(docs)
    labels = clf.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print len(labels[labels!=-1]), n_clusters_
    print metrics.silhouette_score(docs, labels)
    for l in set(labels):
        if l > -1:
            for i in np.argwhere(labels==l):
                print l, info[i[0]]


def get_neighbours():
    docs, info = preprocess(load=True)
    sample = docs[152, :]
    sim = []
    for i in range(docs.shape[0]):
        sim.append((i, metrics.pairwise.cosine_similarity(sample, docs[i, :])))
    scores = sorted(sim, key=lambda x:x[1], reverse=True)[: 30]
    competitors = info[map(lambda x:x[0], scores)]
    competitors = map(lambda x: '{}###{}\n'.format(x[0].strip(), x[1]), zip(competitors, scores))
    print '\n'.join(competitors)

if __name__ == '__main__':
    #db_cluster()
    get_neighbours()

