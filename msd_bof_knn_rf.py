import os
import glob
import hdf5_getters
import pickle
import numpy as np
from sklearn.cluster import KMeans
import time
import resource
import random
import sqlite3
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier

random.seed(3222)
np.random.seed(3222)

def build_set(data):
    num_of_items = 0
    for i in range(len(data)):
        num_of_items += data[i].shape[0]
    print num_of_items
    results = np.ndarray((num_of_items, data[0].shape[1]))
    count = 0
    for i in range(len(data)):
        for j in range(data[i].shape[0]):
            results[count] = data[i][j]
            count += 1
    return results

print "loading data"
segments_pitches_1000 = pickle.load(open("/scratch/ms8599/MillionSongSubset/pitches1000", "rb"))
segments_pitches = pickle.load(open("/scratch/ms8599/MillionSongSubset/pitches", "rb"))
artist_terms = pickle.load(open("/scratch/ms8599/MillionSongSubset/artist_terms", "rb"))
print "done"

# fill vocabulary set using first 1000 songs
print "building vocab set"
t1 = time.time()
vocab_pitches = build_set(segments_pitches_1000)
print "took", time.time()-t1

# get top 300 terms
conn_artist_term = sqlite3.connect("/scratch/ms8599/MillionSongSubset/AdditionalFiles/subset_artist_term.db")
q = "SELECT term,Count(artist_id) FROM artist_term GROUP BY term"
res = conn_artist_term.execute(q)
term_freq_list = res.fetchall()
term_freq = {}
for k in term_freq_list:
    term_freq[k[0]] = int(k[1])
ordered_terms = sorted(term_freq, key=term_freq.__getitem__, reverse=True)
top300 = ordered_terms[:300]
print 'Top 300 hundred terms are:',top300[:3],'...',top300[298:]

# extract terms from artist_terms that are in top 300 and not in vocabulary set of songs
data_terms = np.zeros((9000,300))
count = 0
for i in range(1000, 10000):
    for j in range(artist_terms[i].shape[0]):
        if artist_terms[i][j] in top300:
            data_terms[count, top300.index(artist_terms[i][j])] = 1
    count += 1

print "fitting kmeans to first 1000 songs"
t1 = time.time()
kmeans = KMeans(n_clusters=10, n_jobs=-1)
kmeans.fit(vocab_pitches)
print "took ", time.time()-t1

print "kmeans vector quantization"
t1 = time.time()
pitches_tf = np.zeros((9000, 10))
count = 0
for i in range(1000, 10000):
    labels = kmeans.predict(segments_pitches[i])
    for j in range(len(labels)):
        pitches_tf[count][labels[j]] += 1
    pitches_tf[count] /= sum(pitches_tf[count])
    count += 1
print "took ", time.time()-t1

pickle.dump(pitches_tf, open("/scratch/ms8599/MillionSongSubset/pitches_tf", "wb"))
pickle.dump(data_terms, open("/scratch/ms8599/MillionSongSubset/data_terms", "wb"))

# All preprocessing done, time to fit classifier and predict
train_size = 4500
print "Running knn"
'''audio_clf = Pipeline([('clf',  KNeighborsClassifier())])
parameters = {'clf__n_neighbors': (1,2,5,10)}
gs_clf = GridSearchCV(audio_clf, parameters, n_jobs=-1)
gs_clf.fit(pitches_tf[:train_size], data_terms[:train_size])
prediction =  gs_clf.predict(pitches_tf[train_size:])
precision, recall, fscore, _ =  precision_recall_fscore_support(data_terms[train_size:], prediction, average="micro")
print "precision: ", precision
print "recall: ", recall
print "fscore: ", fscore
print "knn done"'''

for i in range(1,11):
    audio_clf = Pipeline([('clf',  KNeighborsClassifier(n_neighbors=i))])
    audio_clf.fit(pitches_tf[:train_size], data_terms[:train_size])
    prediction =  audio_clf.predict(pitches_tf[train_size:])
    print precision_recall_fscore_support(data_terms[train_size:], prediction, average="macro")
print "knn done"

print "Running random forest"
'''clf = RandomForestClassifier(n_estimators=10)
clf.fit(pitches_tf[:train_size], data_terms[:train_size])
prediction =  clf.predict(pitches_tf[train_size:])
precision, recall, fscore, _ =  precision_recall_fscore_support(data_terms[train_size:], prediction, average="micro")
print "precision: ", precision
print "recall: ", recall
print "fscore: ", fscore
print "random forest done"'''

for i in range(1,11):
    clf = RandomForestClassifier(n_estimators=i)
    clf.fit(pitches_tf[:train_size], data_terms[:train_size])
    prediction =  clf.predict(pitches_tf[train_size:])
    print precision_recall_fscore_support(data_terms[train_size:], prediction, average="macro")
print "random forest done"
