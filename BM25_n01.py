import numpy as np
import numba
import collections
import fuc

@numba.njit(fastmath=True,nogil=True)
def bm25_step(K1,K3,b,term_doc_matrix,term_query_matrix,IDF_matrix,doc_len,querys_repeat_word):
    N=term_doc_matrix.shape[0]
    doc_avglen=np.sum(doc_len)*1.0/N
    sim=np.zeros(N,dtype=np.float64)

    for n in range(N):
        for i in querys_repeat_word:
            dtf_temp=term_doc_matrix[n][i]
            qtf_temp=term_query_matrix[i]
            IDF_temp=IDF_matrix[i]
            if(dtf_temp!=0 and qtf_temp!=0):
                doc_weight=((K1+1)*dtf_temp)/(K1*((1-b)+b*(doc_len[n]/doc_avglen))+dtf_temp)
                query_weight=((K3+1)*qtf_temp)/(K3+qtf_temp)
                IDF=np.log(1+(N-IDF_temp+0.5)/(IDF_temp+0.5))
                sim[n]=sim[n]+(doc_weight*query_weight*IDF)
    sim=sim.argsort()
    sim=sim[::-1]
    return sim

class BM25():
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
        IDF_dict={}
        doc_w_index_dict={}

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
            W_list,Q_dict=fuc.getword_tf(path_f)
            querys_word.append(W_list)
            query_dict[q]=Q_dict

        doc_len=np.zeros(N,dtype=np.int16)

        for n in range(N):
            path_f="Document/"+doc_list[n]
            W_list,D_dict,doc_w_index_dict,IDF_dict=fuc.getword_idf_index(path_f,doc_w_index_dict,IDF_dict,True)
            doc_len[n]=len(W_list)
            docs_word.append(W_list)
            doc_dict[n]=D_dict

        term_doc_matrix=np.zeros((N,len(doc_w_index_dict)),dtype=np.int16)
        IDF_matrix=np.zeros(len(doc_w_index_dict),dtype=np.int16)
        term_query_matrix=np.zeros((Q,len(doc_w_index_dict)),dtype=np.int16)
        querys_repeat_word=[]
        for n in range(len(querys_word)):
            query_norepeat_word=list({}.fromkeys(querys_word[n]).keys())
            t=[]
            for w in querys_word[n]:
                w_index = doc_w_index_dict.get(w,None)
                if(w_index!=None):
                    t.append(w_index)
            querys_repeat_word.append(t)
            for w in query_norepeat_word:
                w_index = doc_w_index_dict.get(w,None)
                if(w_index==None):
                    print("query index=0 error"+w+" w_index:"+str(n))
                else:
                    count = query_dict[n].get(w,0)
                    term_query_matrix[n][w_index]=count
        del query_dict,querys_word

        for n in range(len(docs_word)):
            doc_norepeat_word=list({}.fromkeys(docs_word[n]).keys())
            for w in doc_norepeat_word:
                w_index = doc_w_index_dict.get(w,None)
                if(w_index==None):
                    print("doc index=0 error"+w+" w_index:"+str(n))
                count = doc_dict[n].get(w,0)
                term_doc_matrix[n][w_index]=count
                count = IDF_dict.get(w,0)
                IDF_matrix[w_index]=count
        del doc_dict,IDF_dict
        
        return query_list,doc_list,term_doc_matrix,term_query_matrix,IDF_matrix,doc_len,querys_repeat_word

    def fit(self,K1,K3,b,top=None):
        query_list,doc_list,term_doc_matrix,term_query_matrix,IDF_matrix,doc_len,querys_repeat_word=self.getdata("Query/")
        if top==None:
            top=len(doc_list)
        fp = open("submission.txt", "w")
        fp.write("Query,RetrievedDocuments")
        Q=len(query_list)
        for q in range(Q):
            print(q)
            sim=bm25_step(
                K1,
                K3,
                b,
                term_doc_matrix,
                term_query_matrix[q],
                IDF_matrix,
                doc_len,
                np.array(querys_repeat_word[q])
                )
            fp.write("\n"+query_list[q]+",")
            for n in range(top):
                fp.write(doc_list[sim[n]]+" ")
        fp.close()

    def prefit(self,K1,K3,b,top=None):
        query_list,doc_list,term_doc_matrix,term_query_matrix,IDF_matrix,doc_len,querys_repeat_word=self.getdata("Query/")
        if top==None:
            top=len(doc_list)
        fp = open("prefit.txt", "w")
        Q=len(query_list)
        for q in range(Q):
            print(q)
            sim=bm25_step(
                K1,
                K3,
                b,
                term_doc_matrix,
                term_query_matrix[q],
                IDF_matrix,
                doc_len,
                np.array(querys_repeat_word[q])
                )
            for n in range(top):
                fp.write(str(sim[n])+" ")
            fp.write("\n")
        fp.close()
        


if __name__ == "__main__":
    K3=5
    K1=1.6
    b=0.788
    sim_list=[]
    sim_list=BM25("query_list.txt","doc_list.txt").fit(K1,K3,b)