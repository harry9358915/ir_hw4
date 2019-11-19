import numpy as np
import numba
import collections
import sklearn as sk
from scipy.sparse import issparse, csr_matrix, coo_matrix
from sklearn.utils import check_array, check_random_state
import fuc
import math

@numba.njit(fastmath=True,nogil=True)#nogil:當進入這類編譯好的函數時，Numba將會釋放全局線程鎖，fastmath:減少運算時間
def SMM_e_step(a,p_smm,p_smm_w,p_w_BG,probability_threshold=1e-32):

    '''
    X:dtf matrix NxV N:top doc數量 V:top doc內的word
    p_smm:V
    p_smm_w:V
    p_w_BG:BG model，array index為word_id。
    probability_threshold:閥值10^-32次方
    '''
    norm=0.0
    for w in range(p_smm.shape[0]):        
        fraction=(1-a)*p_smm[w]
        denominator=(1-a)*p_smm[w]+a*p_w_BG[w]
        p_smm_w[w]=fraction/denominator
        norm+=p_smm_w[w]
    for i in range(len(p_smm_w)):
        if norm>0:
            p_smm_w[i]/=norm
    
    return p_smm_w

@numba.njit(fastmath=True,nogil=True)
def SMM_m_step(X_rows,X_cols,X_vals,p_smm,p_smm_w):
    norm_psmm=0.0
    p_smm[:]=0.0
    for nz in range(X_vals.shape[0]):
        w=X_cols[nz]
        x=X_vals[nz]
        
        sum_temp=x*p_smm_w[w]

        p_smm[w]+=sum_temp
        norm_psmm+=sum_temp
    for i in range(len(p_smm)):
        if norm_psmm>0:
            p_smm[i]/=norm_psmm
    
    return p_smm

@numba.njit(fastmath=True, nogil=True)
def log_likelihood(a,X_rows, X_cols, X_vals, p_smm, p_w_BG):
    result=1.0
    for nz in range(X_vals.shape[0]):
        w=X_cols[nz]
        x=X_vals[nz]
        result*=math.pow((1-a)*p_smm[w]+a*p_w_BG[w],x)      
    return result

def SMM_fit(X,a,n_iter,p_w_BG):
    p_smm=np.zeros(X.shape[1],dtype=np.float64)
    p_smm_w=np.empty(X.shape[1],dtype=np.float64)
    p_smm=np.random.random(size=(X.shape[1]))
    fuc.normalize(p_smm)
    X = check_array(term_doc_matrix, accept_sparse="csr")
    if not issparse(X):
        X = csr_matrix(X)
    X = X.tocoo()
    for _ in range(n_iter):
        p_smm_w=SMM_e_step(a,
                        p_smm,
                        p_smm_w,
                        p_w_BG)
        p_smm=SMM_m_step(X.row,
                        X.col,
                        X.data,
                        p_smm,
                        p_smm_w)
        #print(log_likelihood(a,X.row,X.col,X.data,p_smm,p_w_BG))
    p_smm_w=p_smm_w.argsort()
    p_smm_w=p_smm_w[::-1]
    return p_smm_w




class SMM():
    def __init__(self,n_iter=100,m=50):
        self.n_iter=n_iter
        self.m=m
    def fit(self,X,a,p_w_BG):
        p_smm=SMM_fit(X,a,self.n_iter,p_w_BG)
        return p_smm

if __name__ == '__main__':
    doc_list=[]
    top_list=[]
    query_list=[]
    querys_word=[]

    f=open("prefit.txt")    
    for line in f.readlines():
        line=line.strip('\n')
        top_list.append(line.split())
    f.close()
    f=open("query_list.txt")
    for line in f.readlines():
        line=line.strip('\n')
        query_list.append(line)
    f.close()
    Q=len(query_list)
    for q in range(Q):
        path_f="Query/"+query_list[q]
        W_list,Q_dict=fuc.getword_tf(path_f)
        querys_word.append(W_list)
    f=open("doc_list.txt")
    for line in f.readlines():
        line=line.strip('\n')
        doc_list.append(line)
    f.close()
    f=open("BGLM.txt")
    p_w_BG=[]
    for line in f.readlines():
        (key, val) = line.split()
        val=math.exp(float(val))
        p_w_BG.append(val)
    p_w_BG=np.array(p_w_BG,dtype=np.float64)
    for t in range(len(top_list)):
        if t<9:
            f=open("Query/4000"+str(t+1)+".query","a")
        elif (t>=9 and t<99):
            f=open("Query/400"+str(t+1)+".query","a")
        else:
            f=open("Query/40"+str(t+1)+".query","a")
        print("q:"+str(t))
        docs_word=[]
        doc_w_index_dict={}
        doc = lambda: collections.defaultdict(doc)
        doc_dict=doc()
        for nz in top_list[t]:
            path_f="Document/"+doc_list[int(nz)]
            W_list,D_dict,doc_w_index_dict=fuc.getword_tf_index(path_f,doc_w_index_dict,True)
            docs_word.append(W_list)
            doc_dict[nz]=D_dict
        N=len(docs_word)
        V=len(doc_w_index_dict)
        term_doc_matrix=np.zeros((N,V),dtype=np.float64)
        for n in range(len(docs_word)):
            doc_norepeat_word=list({}.fromkeys(docs_word[n]).keys())
            for w in doc_norepeat_word:
                w_index = doc_w_index_dict.get(w,None)
                if(w_index==None):
                    print("index=0 error"+w+" w_index:"+str(n))
                count = doc_dict[top_list[t][n]].get(w,0)
                term_doc_matrix[n][w_index]=count
        
        p_smm=SMM().fit(term_doc_matrix,0.4,p_w_BG)
        p_smm=p_smm[:30]
        p_smm.tolist()
        doc_w_index_dict = {v: k for k, v in doc_w_index_dict.iteritems()}
        for item in p_smm:
            f.write(doc_w_index_dict.get(item,None)+" ")
        f.write("\n")