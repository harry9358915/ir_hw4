import numpy as np
import collections
import datetime
import fuc
import numba
from scipy.sparse import issparse, csr_matrix, coo_matrix
from sklearn.utils import check_array
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
@numba.njit(fastmath=True,nogil=True)
def vector(X_rows,X_cols,X_vals,IDF_matrix,temp_vector):
    N=temp_vector.shape[0]
    for i in range(X_vals.shape[0]):
        d=X_rows[i]
        w=X_cols[i]
        x=X_vals[i]
        if(IDF_matrix[w]!=0):
            temp_vector[d][w]=x*np.log(N*1.0/IDF_matrix[w])
    return temp_vector
'''
@numba.njit(fastmath=True,nogil=True)
def similarity(doc_vector,X_rows,X_cols,X_vals,Q): 
    N=doc_vector.shape[0]
    V=doc_vector.shape[1]
    Sum_weight=np.zeros((Q,N),dtype=np.float64)
    Sum_dw=np.zeros((Q,N),dtype=np.float64)
    Sum_qw=np.zeros((Q,N),dtype=np.float64)
    sim=np.zeros((Q,N),dtype=np.float64)
    
    for d in range(N):
        Sum_dw[:,d]=np.sum(np.square(doc_vector[d]))
        for nz in range(X_vals.shape[0]):
            q=X_rows[nz]
            w=X_cols[nz]
            x=X_vals[nz]
            Sum_weight[q][d]+= x*doc_vector[d][w]     
            Sum_qw[q][d]+=np.square(x)
    Sum_dw=np.sqrt(Sum_dw)
    Sum_qw=np.sqrt(Sum_qw)
    t1=Sum_dw*Sum_qw
    sim=np.true_divide(Sum_weight,t1) 
    return sim 
'''
@numba.njit(fastmath=True,nogil=True)
def similarity(doc_vector,query_vector):
    Q=query_vector.shape[0]
    D=doc_vector.shape[0]
    V=query_vector.shape[0]
    cos=np.zeros((Q,N),dtype=np.float64)
    normb = np.zeros(N,dtype=np.float64)
    for d in range(D):
        normb[d] = np.linalg.norm(doc_vector[d])
    for q in range(Q): 
        dot = np.dot(doc_vector, query_vector[q])
        norma = np.linalg.norm(query_vector[q])
        cos[q] = dot / (norma * normb)
    return cos

class VectorSpace():
    def __init__(self,querylist_path,doclist_path):
        self.querylist_path=querylist_path
        self.doclist_path=doclist_path

    def getdata(self,path_query):
        query_list=[]
        querys_word=[]
        doc_list=[]
        docs_word=[]
        query = lambda: collections.defaultdict(query)
        query_dict=query()
        doc = lambda: collections.defaultdict(doc)
        doc_dict=doc()
        D_IDF_dict={}
        doc_w_index_dict={}
        Q_IDF_dict={}
        f=open(self.querylist_path)

        for line in f.readlines():
            line=line.strip('\n')
            query_list.append(line)
        f.close()
        Q=len(query_list)

        f=open(self.doclist_path)
        for line in f.readlines():
            line=line.strip('\n')
            doc_list.append(line)
        f.close()
        N=len(doc_list)


        for q in range(Q):
            path_f=path_query+query_list[q]
            W_list,Q_dict,Q_IDF_dict=fuc.getword_idf(path_f,Q_IDF_dict)
            querys_word.append(W_list)
            query_dict[q]=Q_dict

        doc_len=np.zeros(N,dtype=np.int16)

        for n in range(N):
            path_f="Document/"+doc_list[n]
            W_list,D_dict,doc_w_index_dict,D_IDF_dict=fuc.getword_idf_index(path_f,doc_w_index_dict,D_IDF_dict,True)
            doc_len[n]=len(W_list)
            docs_word.append(W_list)
            doc_dict[n]=D_dict

        term_doc_matrix=np.zeros((N,len(doc_w_index_dict)),dtype=np.int)
        D_IDF_matrix=np.zeros(len(doc_w_index_dict),dtype=np.int)
        Q_IDF_matrix=np.zeros(len(doc_w_index_dict),dtype=np.int)
        term_query_matrix=np.zeros((Q,len(doc_w_index_dict)),dtype=np.int)

        for n in range(len(querys_word)):
            query_norepeat_word=list({}.fromkeys(querys_word[n]).keys())
            for w in query_norepeat_word:
                w_index = doc_w_index_dict.get(w,None)
                if(w_index!=None):
                    count = query_dict[n].get(w,0)
                    term_query_matrix[n][w_index]=count
                    count = Q_IDF_dict.get(w,0)
                    Q_IDF_matrix[w_index]=count
        del query_dict

        for n in range(len(docs_word)):
            doc_norepeat_word=list({}.fromkeys(docs_word[n]).keys())
            for w in doc_norepeat_word:
                w_index = doc_w_index_dict.get(w,None)
               
                if(w_index==None):
                    print("doc index=0 error"+w+" w_index:"+str(n))
                count = doc_dict[n].get(w,0)
                term_doc_matrix[n][w_index]=count
                count = D_IDF_dict.get(w,0)
                D_IDF_matrix[w_index]=count
        del doc_dict,D_IDF_dict
        
        return query_list,doc_list,term_doc_matrix,term_query_matrix,D_IDF_matrix,Q_IDF_matrix

    def fit(self):
        query_list,doc_list,term_doc_matrix,term_query_matrix,D_IDF_matrix,Q_IDF_matrix=self.getdata("Query/")
        N=term_doc_matrix.shape[0]
        V=term_doc_matrix.shape[1]
        Q=term_query_matrix.shape[0]
        doc_vector=np.zeros((N,V),dtype=np.float64)
        query_vector=np.zeros((Q,V),dtype=np.float64)
        X = check_array(term_doc_matrix, accept_sparse="csr")
        if not issparse(X):
            X = csr_matrix(X)
        X = X.tocoo()        
        doc_vector=vector(X.row,X.col,X.data,D_IDF_matrix,doc_vector)
        X = check_array(term_query_matrix, accept_sparse="csr")
        if not issparse(X):
            X = csr_matrix(X)
        X = X.tocoo() 
        query_vector=vector(X.row,X.col,X.data,Q_IDF_matrix,query_vector)
        return doc_vector,query_vector,query_list,doc_list

if __name__ == '__main__':
       
    doc_vector,query_vector,query_list,doc_list=VectorSpace("query_list.txt","doc_list.txt").fit()
    Q=query_vector.shape[0]
    V=query_vector.shape[1]
    N=len(doc_list)
    top=5
    iters=10


    X = check_array(query_vector, accept_sparse="csr")
    if not issparse(X):
        X = csr_matrix(X)
    X = X.tocoo() 
    #sim=cosine_similarity(query_vector,doc_vector)
    sum_qw=0.0
    sim=similarity(doc_vector,query_vector)
    #sim=similarity(doc_vector,X.row,X.col,X.data,Q)
    sim=np.argsort(sim,axis=1)
    sim=sim[:,::-1] 


    for _ in range(iters):
        retrieval_vecs=doc_vector[sim[:,:top]].mean(axis=1)
        query_vector = 1 * query_vector + 0.8 * retrieval_vecs
        sim=similarity(doc_vector,query_vector)
        sim=np.argsort(sim,axis=1)
        sim=sim[:,::-1] 

    with open('submission.txt', mode='w') as file:
        file.write('Query,RetrievedDocuments\n')
        for query_name, ranking in zip(query_list, sim):
            ranked_docs = ' '.join([doc_list[idx] for idx in ranking])
            file.write('%s,%s\n' % (query_name, ranked_docs))  



