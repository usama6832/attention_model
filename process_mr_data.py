import numpy as np
#import cPickle
import _pickle as cPickle
from collections import defaultdict
import sys, re
import pandas as pd
import csv

vector_size = 300 #50
#nnum = 130
#sent_len = np.zeros((2,nnum))
#for i in range(nnum):
#    sent_len[0,i] = (i+1)*10

'''
def build_data_cv(data_folder, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    vocab = defaultdict(float)
    with open(pos_file, "rb") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join('rev'))
            else:
                orig_rev = " ".join('rev').lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":1,
                      "text": orig_rev,
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
    with open(neg_file, "rb") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join('rev'))
            else:
                orig_rev = " ".join('rev').lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":0,
                      "text": orig_rev,
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
    return revs, vocab
'''

def build_data_cv_multi(data_folder, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    vocab = defaultdict(float)
    ll = ['糖尿病','高血压', '慢性阻塞性肺病','心律失常','哮喘','胃炎']
    pos_file = open(pos_file,'r',encoding = "ISO-8859-1").readlines()
    neg_file = open(neg_file,'r',encoding = "ISO-8859-1").readlines()
    
    for i in range(len(pos_file)):
        rev = []
        rr = pos_file[i].strip()
        rev.append(rr)
        words = set(rr.split())
        for word in words:
            vocab[word] += 1
        num_words = len(rr.split())
        #kkk = num_words / 10
        #sent_len[1,kkk] += 1
        datum  = {"y":1,
                  "text": rr,                             
                  "num_words": num_words,
                  "split": np.random.randint(0,cv)}
        revs.append(datum)
    
    for i in range(len(neg_file)):
        rev = []
        rr = neg_file[i].strip()
        rev.append(rr)
        words = set(rr.split())
        for word in words:
            vocab[word] += 1
        num_words = len(rr.split())
        #kkk = num_words / 10
        #sent_len[1,kkk] += 1
        datum  = {"y":0,
                  "text": rr,                             
                  "num_words": num_words,
                  "split": np.random.randint(0,cv)}
        revs.append(datum)


    return revs, vocab



def get_W(word_vecs, k=vector_size):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map 

'''
def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float64').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float64')
            else:
                f.read(binary_len)
    
    return word_vecs
'''

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, 'r') as f:
        for line in f:
            lines = f.readline()
            #dictionary = dict()
            #word_counter = 0    
    #for line in lines:
            words = lines.split()
            word_vecs[words[0]] = np.asarray(words[1:], dtype='float64')
            print('Found %s word vectors matching enron data set.' % len(word_vecs))           
    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=1, k=vector_size):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

if __name__=="__main__":
    #w2v_file = ""
    w2v_file = sys.argv[1] 
    #data_folder = ["./data/rt-polarity.pos", "./data/rt-polarity.neg"]
    data_folder = ["./data/rt-polarity.pos",
                   "./data/rt-polarity.neg"]
    #pos_file_path = "rt-polarity.pos"
    #neg_file_path = "rt-polarity.neg"
    print ("loading data...",)
    revs, vocab = build_data_cv_multi(data_folder, cv=10, clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    mean_l = np.mean(pd.DataFrame(revs)["num_words"])
#    writer = csv.writer(open('sent_len.csv','wb'))
#    writer.writerows(sent_len)
    print ("data loaded!")
    w2v = load_bin_vec(w2v_file, vocab)
    print ("number of sentences: " + str(len(revs)))
    print ("vocab size: " + str(len(vocab)))
    print ("max sentence length: " + str(max_l))
    print ("mean sentence length: " + str(mean_l))
    print ("loading word2vec vectors...")
    print ("word2vec loaded!")
    print ("num words already in word2vec: " + str(len(w2v)))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)

    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W, word_idx_map = get_W(rand_vecs)
    W2, _ = get_W(rand_vecs)

    cPickle.dump([revs, W, W2, word_idx_map, vocab], open("mr.p", "wb"))
    print ("dataset created!")
