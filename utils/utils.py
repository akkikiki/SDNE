import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
import pdb

class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def getSimilarity(result):
    print("getting similarity...")
    return np.dot(result, result.T)
    
def check_link_reconstruction(embedding, graph_data, check_index):
    def get_precisionK(embedding, data, max_index):
        print("get precisionK...")
        #similarity = getSimilarity(embedding).reshape(-1)
        similarity = getSimilarity(embedding).reshape(-1).astype(np.float16)
        print("got similarity")
        #print(type(similarity))
        #sortedInd = np.argsort(similarity)
        #topKInd = np.argpartition(-similarity, check_index)[:]
        #sortedInd = np.argpartition(similarity, range(max_index))[:max_index]
        #topK = 10
        topK = max_index
        sortedInd = []
        for i in range(topK):
            Ind = np.argmax(similarity)
            similarity[Ind] = 0
            sortedInd.insert(0, Ind)
        cur = 0
        count = 0
        precisionK = []
        sortedInd = sortedInd[::-1] # reverse
        for ind in sortedInd:
            x = ind / data.N
            y = ind % data.N
            count += 1
            print("converting adjacent matrix...")
            if (data.adj_matrix[x].toarray()[0][y] == 1 or x == y):
                cur += 1 
            precisionK.append(1.0 * cur / count)
            if count > max_index:
                break
            print("prec@K computation done...")
        return precisionK
        
    precisionK = get_precisionK(embedding, graph_data, np.max(check_index))
    ret = []
    for index in check_index:
        print("precisonK[%d] %.2f" % (index, precisionK[index - 1]))
        ret.append(precisionK[index - 1])
    return ret


def check_multi_label_classification(X, Y, test_ratio = 0.9):
    def small_trick(y_test, y_pred):
        y_pred_new = np.zeros(y_pred.shape,np.bool)
        sort_index = np.flip(np.argsort(y_pred, axis = 1), 1, 10)
        for i in range(y_test.shape[0]):
            num = sum(y_test[i])
            for j in range(num):
                y_pred_new[i][sort_index[i][j]] = True
        return y_pred_new
        
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = test_ratio)
    clf = OneVsRestClassifier(LogisticRegression())
    clf.fit(x_train, y_train)
    y_pred = clf.predict_proba(x_test)
    
    ## small trick : we assume that we know how many label to predict
    y_pred = small_trick(y_test, y_pred)
    
    micro = f1_score(y_test, y_pred, average = "micro")
    macro = f1_score(y_test, y_pred, average = "macro")
    print("micro_f1: %.4f" % (micro))
    print("macro_f1: %.4f" % (macro))
    #############################################


    
