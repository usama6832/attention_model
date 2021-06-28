import cPickle
#import _pickle as cPickle
import numpy as np

vector_size = 300

def get_idx_from_sent(sent, word_idx_map, max_l=51, k=vector_size, filter_h=5, pad_left=True):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    if pad_left:
        #for i in xrange(pad):
        for i in range(pad):
            x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x


def make_idx_data_cv(revs, word_idx_map, cv, max_l=51, k=vector_size, filter_h=5, pad_left=True):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h, pad_left=pad_left)
        sent.append(rev["y"])
        if rev["split"]==cv:
            test.append(sent)
        else:
            train.append(sent)
    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")
    return [train, test]


def make_idx_data_cv_org_text(revs, word_idx_map, cv, max_l=51, k=vector_size, filter_h=5, pad_left=True):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for rev in revs:
        if rev["split"]==cv:
            test.append(rev["text"])
        else:
            train.append(rev["text"])
    return [train, test]


x = None
def load_data(fold, pad_left=True):
    global x
    if x is None:
        x = cPickle.load(open("mr.p","rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    #revs, W, word_idx_map, vocab = x[0], x[1], x[2], x[3]
    datasets = make_idx_data_cv(revs, word_idx_map, fold, max_l=56, k=vector_size, filter_h=5, pad_left=pad_left)
    img_h = len(datasets[0][0])-1
    return datasets[0][:,:img_h], datasets[0][:, -1], datasets[1][:,: img_h], datasets[1][: , -1], W, W2
    #return datasets[0][:,:img_h], datasets[0][:, -1], datasets[1][:,: img_h], datasets[1][: , -1], W

def load_data_org(fold, pad_left=True):
    global x
    if x is None:
        x = cPickle.load(open("mr.p","rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    #revs, W, word_idx_map, vocab = x[0], x[1], x[2], x[3]
    datasets = make_idx_data_cv(revs, word_idx_map, fold, max_l=56, k=vector_size, filter_h=5, pad_left=pad_left)
    train_text, test_text = make_idx_data_cv_org_text(revs, word_idx_map, fold, max_l=51, k=vector_size, filter_h=5, pad_left=pad_left)
    img_h = len(datasets[0][0])-1
    return datasets[0][:,:img_h], datasets[0][:, -1], datasets[1][:,: img_h], datasets[1][: , -1], W, W2, train_text, test_text
    #return datasets[0][:,:img_h], datasets[0][:, -1], datasets[1][:,: img_h], datasets[1][: , -1], W, train_text, test_text
