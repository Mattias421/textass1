import numpy as np


class Retrieve:
    
    # Create new Retrieve object storing index and term weighting 
    # scheme. (You can extend this method, as required.)
    def __init__(self,index, term_weighting,pseudoRelevanceFeedback, n, t):
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        self.pseudoRelevanceFeedback = pseudoRelevanceFeedback
        self.n = n
        self.t = t

        #stop list
        self.stops = set()
        with open('stop_list.txt', 'r') as stop_fs:
            for line in stop_fs :
                self.stops.add(line.strip())

        
    def compute_number_of_documents(self):
        self.doc_ids = set()
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids) 
    
    #term frequency
    tf = lambda self, t, d : self.index[t][d]

    #inverse document frequency
    idf = lambda self, t : np.log(self.num_docs / (1 + len(self.index[t])))

    #weight each term, store in matrix W (W[0] is the query weights)
    def weight_terms(self, query):

        #matrix for query/documents and each term weight, document 0 is the query
        W = np.zeros(((self.num_docs + 1),len(self.index)))

        i = 0 #term number
        
        for t in self.index:
            if t in query:
                if self.term_weighting == 'binary':
                    W[0,i] = 1
                elif self.term_weighting == 'tf':
                    W[0,i] = query.count(t)
                else:
                    W[0,i] = query.count(t)*self.idf(t)
                
            for doc in self.index[t]:
                d = int(doc)
                #add term weight for each doc
                if self.term_weighting == 'binary':
                    W[d,i] = 1
                elif self.term_weighting == 'tf':
                    W[d,i] = self.tf(t,d)
                else:
                    #tfidf
                    W[d,i] = self.tf(t,d) * self.idf(t)

            i += 1
            
        return W
   
    def get_top_docs(self, query, N=10):

        #calculate term weighting matrix
        W = self.weight_terms(query) #same for every query?

        W_n = np.matmul(W[0],W.T) #numerators of cosine equation

        W_d = np.zeros(self.num_docs + 1) #document vector size
        
        for i in range(1, self.num_docs + 1):
            W_d[i] = np.sqrt(np.dot(W[i],W[i]))

        sims = np.zeros(self.num_docs + 1) #cosine similarity
        
        #divide by size of doc vector, could this be optimized?
        for i in range(1, self.num_docs + 1):
            sims[i] = W_n[i] / W_d[i] 

        return sims.argsort()[-N:][::-1] #return top N docs

    #
    # PSEUDO RELEVENCE FEEDBACK -------------------------------------------------------------------
    #

    def doc_terms(self, docs):
        #return list of terms which appear in a collection of docs

        terms = []

        for t in self.index:
            for d in docs:
                if d in self.index[t]:
                    terms.append(t)
        return terms

    #Pseudo Relevence Feedback, T is number of terms to add
    def prf(self, query, top_docs, T=1):

        terms = self.doc_terms(top_docs) #all terms from top docs

        expanded_terms_weights = {}

        for t in terms:
            if t in self.index:
                raw_count = 0
                for d in self.index[t]:
                    raw_count += self.index[t][d]

                weight = (raw_count * self.idf(t)) #document collection tfidf

                expanded_terms_weights[t] = weight
            else:
                #remove term which doesn't exist in index
                terms.remove(t)
       
        for i in range(T):
            #since T=1 is best, for loop is not needed
            try:
                top = max(expanded_terms_weights, key=expanded_terms_weights.get)

                query.append(top)

                expanded_terms_weights.pop(top) #zero out current top term 
            
            except:
                #incase all terms from expanded_terms_weights are popped
                i=T
  

        return self.get_top_docs(query)   


    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms). Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):

        if self.pseudoRelevanceFeedback:
            #go through PRFS
            n = self.n
            t = self.t

            top_docs = self.get_top_docs(query,n) #get n top docs   
            return self.prf(query, top_docs, t)
        else:
            return self.get_top_docs(query)


            
