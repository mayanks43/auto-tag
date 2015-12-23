import pickle
import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier

random.seed(3222)
np.random.seed(3222)

pitches_tf = pickle.load(open("/scratch/ms8599/MillionSongSubset/pitches_tf", "rb"))
data_terms = pickle.load(open("/scratch/ms8599/MillionSongSubset/data_terms", "rb"))

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
