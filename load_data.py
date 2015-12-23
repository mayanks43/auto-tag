import os
import glob
import hdf5_getters
import pickle

def get_data(basedir, function, upto=10000, ext='.h5'):
    data = []
    count = 0
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        for f in files:
            if count == upto:
                return data
            if count%100 == 0:
                print count
            h5 = hdf5_getters.open_h5_file_read(f)
            data.append(function(h5))
            h5.close()
            count += 1
    return data

# save as pickle to save time in loading
segments_pitches_1000 = get_data('/scratch/ms8599/MillionSongSubset/data', hdf5_getters.get_segments_pitches, upto=1000)
pickle.dump(segments_pitches_1000, open("/scratch/ms8599/MillionSongSubset/pitches1000", "wb"))
segments_pitches = get_data('/scratch/ms8599/MillionSongSubset/data', hdf5_getters.get_segments_pitches)
pickle.dump(segments_pitches, open("/scratch/ms8599/MillionSongSubset/pitches", "wb"))
artist_terms = get_data('/scratch/ms8599/MillionSongSubset/data', hdf5_getters.get_artist_terms)
pickle.dump(artist_terms, open("/scratch/ms8599/MillionSongSubset/artist_terms", "wb"))
