import numpy as np
import numba
import collections

def getword_tf(path_f,bool_doc=False):
    word_temp=[]
    dict_temp={}
    f = open(path_f)
    for line in f.readlines():
        line=line.strip('\n')
        word_temp.extend(line.split())
    f.close()
    word_temp=[z for z in word_temp if z!='-1']
    if bool_doc:
        word_temp=word_temp[5:]
    distinctword=list({}.fromkeys(word_temp).keys())
    for z in distinctword:
        dict_temp[z]=word_temp.count(z)
    return word_temp,dict_temp

def getword_idf(path_f,IDF_dict,bool_doc=False):
    word_temp=[]
    dict_temp={}
    f = open(path_f)
    for line in f.readlines():
        line=line.strip('\n')
        word_temp.extend(line.split())
    f.close()
    word_temp=[z for z in word_temp if z!='-1']
    if bool_doc:
        word_temp=word_temp[5:]
    distinctword=list({}.fromkeys(word_temp).keys())
    for z in distinctword:
        dict_temp[z]=word_temp.count(z)
        if (IDF_dict.get(z,0)==0):
            IDF_dict[z]=1
        else:
            IDF_dict[z]=IDF_dict.get(z)+1
    return word_temp,dict_temp,IDF_dict

#@numba.njit(fastmath=True,nogil=True)
def bm25_step(K1,K3,b,doc_dict,query_dict_1,IDF_dict,querys_word,doc_len):
    N=len(doc_dict)
    V=len(query_dict_1)
    doc_avglen=np.sum(doc_len)*1.0/N
    sim=np.empty(N,dtype=np.float64)
    for n in range(N):
        for w in range(V):
            dtf_temp=doc_dict[n].get(querys_word[w],0)
            qtf_temp=query_dict_1.get(querys_word[w],0)
            IDF_temp=IDF_dict.get(querys_word[w],0)
            if(dtf_temp!=0 and qtf_temp!=0):
                doc_weight=((K1+1)*dtf_temp)/(K1*((1-b)+b*(doc_len[n]/doc_avglen))+dtf_temp)
                query_weight=((K3+1)*qtf_temp)/(K3+qtf_temp)
                IDF=np.log(1+(N-IDF_temp+0.5)/(IDF_temp+0.5))
                sim[n]=[n]+(doc_weight*query_weight*IDF)
    sim=sim.argsort()
    sim=sim[::-1]
    return sim

class BM25():
    def __init__(self):
        self.query_list=[]
        self.querys_word=[]
        self.doc_list=[]
        self.docs_word=[]
        query = lambda: collections.defaultdict(query)
        self.query_dict=query()
        doc = lambda: collections.defaultdict(doc)
        self.doc_dict=doc()
        self.IDF_dict={}
        self.doc_len=[]
    def getdata(self,querylist_path,doclist_path):
        
        f=open(querylist_path)
        for line in f.readlines():
            line=line.strip('\n')
            self.query_list.append(line)
        f.close()
        Q=len(self.query_list)

        f=open(doclist_path)
        for line in f.readlines():
            line=line.strip('\n')
            self.doc_list.append(line)
        f.close()
        N=len(self.doc_list)
    

        for q in range(Q):
            path_f="Query/"+self.query_list[q]
            W_list,Q_dict=getword_tf(path_f)
            self.querys_word.append(W_list)
            self.query_dict[q]=Q_dict
        
        for n in range(N):
            path_f="Document/"+self.doc_list[n]
            W_list,D_dict,self.IDF_dict=getword_idf(path_f,self.IDF_dict,True)
            self.doc_len.append(len(W_list))
            self.docs_word.append(W_list)
            self.doc_dict[n]=D_dict

        return self

    def fit(self,K1,K3,b,top=None):
        if top==None:
            top=len(self.doc_list)
        Q=len(self.query_list)
        fp = open("submission.txt", "w")
        fp.write("Query,RetrievedDocuments")
        for q in range(Q):
            sim=bm25_step(
                K1,
                K3,
                b,
                self.doc_dict,
                self.query_dict[q],
                self.IDF_dict,
                self.querys_word[q],
                self.doc_len
                )
            fp.write("\n"+self.query_list[q]+",")
            for n in range(top):
                fp.write(self.doc_list[sim[n]]+" ")
        fp.close()

    def prefit(self,K1,K3,b,top=None):
        sim_list=[]
        if top==None:
            top=len(self.doc_list)
        Q=len(self.query_list)
        fp = open("prefit.txt", "w")
        for q in range(Q):
            sim=bm25_step(
                K1,
                K3,
                b,
                self.doc_dict,
                self.query_dict[q],
                self.IDF_dict,
                self.querys_word[q],
                self.doc_len
                )
            for n in range(top):
                fp.write(str(sim[n])+" ")
            fp.write("\n")
            sim_list.append(sim)
        fp.close()
        return sim_list
        


if __name__ == "__main__":
    K3=5
    K1=1.6
    b=0.788
    B=BM25().getdata("query_list.txt","doc_list.txt")
    B.prefit(K1,K3,b,50)