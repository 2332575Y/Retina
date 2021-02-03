import pickle

def loadPickle(fname):
    with open(fname, "rb") as f:
        temp = pickle.load(f)
    return temp

def savePickle(fname, data):
    with open(fname, "wb") as f:
            pickle.dump(data, f)