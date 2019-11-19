import numpy as np

def normalize(vec):
    #vec = vec*1.0 / np.linalg.norm(vec) 
    s=np.sum(vec)
    #assert(abs(s) != 0.0) # the sum must not be 0
    for i in range(len(vec)):
        vec[i]=vec[i]*1.0/s

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

def getword_tf_index(path_f,doc_w_index_dict,bool_doc=False):
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
        if(doc_w_index_dict.get(z,None)==None):
            doc_w_index_dict[z]=len(doc_w_index_dict)
    return word_temp,dict_temp,doc_w_index_dict

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

def getword_idf_index(path_f,doc_w_index_dict,IDF_dict,bool_doc=False):
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
        if(doc_w_index_dict.get(z,None)==None):
            doc_w_index_dict[z]=len(doc_w_index_dict)
    return word_temp,dict_temp,doc_w_index_dict,IDF_dict

