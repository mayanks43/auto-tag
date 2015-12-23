import os
import glob
import hdf5_getters
import pickle
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import time
import resource
import random
import sqlite3
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import preprocessing

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

def getbof(centroids, signal):
    bof_tf = np.zeros(len(centroids))
    labels = assign_points_to_labels(centroids, signal)
    for i in range(labels.shape[0]):
        bof_tf[labels[i]] += 1
    return bof_tf/signal.shape[0]

def assign_points_to_labels(centroids, points):
    # 1 list for each centroid (will contain indices of points)
    k = len(centroids)
    labels = np.zeros(points.shape[0], dtype=int)
    dists = euclidean_distances(points, centroids)
    for i in range(points.shape[0]):
        # find nearest centroid to this point
        best_centroid = 0
        best_distance = dists[i, best_centroid]
        for c in range(1, k):
            distance = dists[i, c]
            if distance < best_distance:
                best_distance = distance
                best_centroid = c
        labels[i] = best_centroid
    return labels

def cache_nearest_centroids(centroids, points, k):
    points_to_clusters = [-1 for i in range(len(points))]
    dists = euclidean_distances(centroids, points)
    for i in range(len(points)):
        # find nearest centroid to this point
        best_centroid = 0
        best_distance = dists[best_centroid, i]
        for c in range(1, k):
            distance = dists[c, i]
            if distance < best_distance:
                best_distance = distance
                best_centroid = c
        points_to_clusters[i] = best_centroid
    return points_to_clusters

def minibatch_kmeans(X, k, b, max_iter=300, C=None):
    n_samples, n_features = X.shape
    if not C:
        C = [X[c] for c in random.sample(xrange(n_samples), k)]

    for i in xrange(max_iter):
        v = [0 for i in xrange(k)]
        M = [X[j] for j in random.sample(xrange(n_samples), b)]
        d = cache_nearest_centroids(C, M, k)
        for j in range(len(M)):
            c = C[d[j]]
            v[d[j]] += 1
            eta = 1./v[d[j]]
            C[d[j]] = (1-eta)*c + eta*M[j]
    return C

def vlad(centroids, signal):
    # signal = variable, d
    # centroids = k, d
    k = len(centroids)
    d = signal.shape[1]
    v = np.zeros([k, d])
    labels = assign_points_to_labels(centroids, signal)
    # predicted_clusters contains indices of descriptors in signal belonging to different centroids.
    for i in range(labels.shape[0]):
        v[labels[i]] += signal[i] - centroids[labels[i]]
    # Global L2 normalization
    n_v = v.reshape(k*d)
    n_v = n_v/np.sqrt(n_v.dot(n_v))
    return n_v

print "loading data"
segments_pitches_1000 = pickle.load(open("/scratch/ms8599/MillionSongSubset/pitches1000", "rb"))
segments_pitches = pickle.load(open("/scratch/ms8599/MillionSongSubset/pitches", "rb"))
segments_timbre_1000 = pickle.load(open("/scratch/ms8599/MillionSongSubset/timbre1000", "rb"))
segments_timbre = pickle.load(open("/scratch/ms8599/MillionSongSubset/timbre", "rb"))
artist_terms = pickle.load(open("/scratch/ms8599/MillionSongSubset/artist_terms", "rb"))
print "done"

# fill vocabulary set using first 1000 songs
print "building vocab set"
t1 = time.time()
vocab_pitches = build_set(segments_pitches_1000)
vocab_timbre = build_set(segments_timbre_1000)
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
        #print i
        if artist_terms[i][j] in top300:
            #print artist_terms[i][j],
            data_terms[count, top300.index(artist_terms[i][j])] = 1
    count += 1
#print

print "fitting kmeans to first 1000 songs"
t1 = time.time()
k = 200
pitches_centroids = minibatch_kmeans(vocab_pitches, k, 1000)
timbre_centroids = minibatch_kmeans(vocab_timbre, k, 1000)
print "took ", time.time()-t1

'''
print "vlad vector quantization"
t1 = time.time()
pitches_tf = np.zeros((9000, k*12))
timbre_tf = np.zeros((9000, k*12))
count = 0
for i in range(1000, 10000):
    pitches_tf[count] = vlad(pitches_centroids, segments_pitches[i])
    timbre_tf[count] = vlad(timbre_centroids, segments_pitches[i])
    count += 1
print "took ", time.time()-t1

print "Fitting PCA"
t1 = time.time()
pca = PCA(n_components=300)
pitches_tf = pca.fit_transform(pitches_tf)
timbre_tf = pca.fit_transform(timbre_tf)
print "pitches new shape", pitches_tf.shape
print "timbre new shape", timbre_tf.shape
print "took ", time.time()-t1
'''


print "Bag of features"
t1 = time.time()
pitches_tf = np.zeros((9000, k))
timbre_tf = np.zeros((9000,k))
count = 0
for i in range(1000, 10000):
    pitches_tf[count] = getbof(pitches_centroids, segments_pitches[i])
    timbre_tf[count] = getbof(timbre_centroids, segments_pitches[i])
    count += 1
print "took ", time.time()-t1

pickle.dump(pitches_tf, open("/scratch/ms8599/MillionSongSubset/pitches_tf", "wb"))
pickle.dump(timbre_tf, open("/scratch/ms8599/MillionSongSubset/timbre_tf", "wb"))
pickle.dump(data_terms, open("/scratch/ms8599/MillionSongSubset/data_terms", "wb"))

#pitches_tf = pickle.load(open("/scratch/ms8599/MillionSongSubset/pitches_tf", "rb"))
#timbre_tf = pickle.load(open("/scratch/ms8599/MillionSongSubset/timbre_tf", "rb"))
#data_terms = pickle.load(open("/scratch/ms8599/MillionSongSubset/data_terms", "rb"))

# All preprocessing done, time to fit classifier and predict
train_size = 4500
print "Baseline"
avgnterm = 19
prediction = np.zeros([9000-train_size, 300], dtype=int)
for i in range(prediction.shape[0]):
    # set top n terms to 1
    for j in range(avgnterm):
        prediction[i][j] = 1
print precision_recall_fscore_support(data_terms[train_size:], prediction, average="micro")
print "Baseline done"

tf = np.hstack([pitches_tf, timbre_tf])
tf = preprocessing.scale(tf)
print "New feature size: ", tf.shape
print "Running svm"
classif = OneVsRestClassifier(SVC(kernel='linear'))
classif.fit(tf[:train_size], data_terms[:train_size])
prediction = classif.predict(tf[train_size:])
print precision_recall_fscore_support(data_terms[train_size:], prediction, average="micro")
print "svm done"

print "Running knn"

for i in [1,10,20]:
    audio_clf = Pipeline([('clf',  KNeighborsClassifier(n_neighbors=i))])
    audio_clf.fit(tf[:train_size], data_terms[:train_size])
    prediction =  audio_clf.predict(tf[train_size:])
    print precision_recall_fscore_support(data_terms[train_size:], prediction, average="micro")
print "knn done"

print "Running random forest"

for i in [1,10,20]:
    clf = RandomForestClassifier(n_estimators=i)
    clf.fit(tf[:train_size], data_terms[:train_size])
    prediction =  clf.predict(tf[train_size:])
    print precision_recall_fscore_support(data_terms[train_size:], prediction, average="micro")
print "random forest done"
