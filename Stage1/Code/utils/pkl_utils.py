import pickle

def save(obj, fname, protocol=3):
    with open(fname, "wb") as f:
        pickle.dump(obj, f, protocol)

def load(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)
