import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle as pkl
import torch.autograd as autograd
import torch.nn as nn
import os
import gensim
import nltk
import random
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import enchant
import matplotlib.pyplot as plt
from nltk import WordNetLemmatizer, word_tokenize, bigrams, ngrams, RegexpTokenizer
lemmatize = WordNetLemmatizer()



def clean_string(text):
    tokens = word_tokenize(text)
    clean_tokens = [lemmatize.lemmatize(token.lower().strip()) for token in tokens]
    return ' '.join(clean_tokens)



def clean_corpus(documents):
    return [clean_string(doc) for doc in documents]








class CRAYBOT(nn.Module):


    def __init__(self,encoder,sent_length=3,n_samples=70,hidden_size=50,max_iter=10,n_layers=1,temperature=1,losses=None):

        super(CRAYBOT,self).__init__()

        self.sent_length=sent_length
        self.n_samples=n_samples
        self.n_layers=n_layers
        self.hidden_size=hidden_size
        self.max_iter=max_iter
        self.temperature=temperature
        size=encoder.wv['the'].shape[0]
        self.losses=None
        self.hidden=self.init_hidden()
        self.lstm = nn.LSTM(size, hidden_size,n_layers,batch_first=True)
        self.layer=nn.Linear(hidden_size,len(encoder.wv.vocab))
        self.encoder=encoder
        self.dropout=nn.Dropout(p=0.1)
        self.doc=None

    def forward(self,x,hidden):
        lstm_out, self.hidden=self.lstm(x,hidden)
        return self.layer(lstm_out), self.hidden


    def _clean(self,doc):
        doc1=' . '.join(doc)
        return [w.lower() for w in doc1.split()]


    def word_to_idx(self,word):
        return list(self.encoder.wv.vocab).index(word)


    def index_to_word(self,n):
        return list(self.encoder.wv.vocab)[n]

    def _make_training_set(self,doc):

        #self.n_samples=n_samples

        self.doc=doc

        clean_doc=self._clean(doc)
        if self.encoder is None:
            self.encoder=gensim.models.Word2Vec([clean_doc], size=self.size, window=5, min_count=1, workers=4,sg=1)
        sent_length=self.sent_length
        n_samples=self.n_samples

        doc_words=' . '.join(doc).split()


        doc_length=len(doc_words)
        training_examples=[]
        X_train=[]
        #y_train=[]
        y_labels=[]
        for i in range(n_samples):
            index=random.choice(range(doc_length-self.sent_length))
            x=doc_words[index:index+self.sent_length]
            y=doc_words[index+1:index+self.sent_length+1]
            next_word=self.word_to_idx(y[-1])
            x_tensor=torch.tensor([self.encoder.wv[w.lower()] for w in x])
            X_train.append(x_tensor)
            y_labels.append(next_word)

        Xt=torch.cat(X_train).view(self.n_samples,self.sent_length,-1)
        tags=torch.tensor(y_labels)

        return Xt, tags

    #Note that prep_string takes in 'doc' as a string.

    def clean_string(self,text):
        tokens = word_tokenize(text)
        clean_tokens = [lemmatize.lemmatize(token.lower().strip()) for token in tokens]
        return ' '.join(clean_tokens)

    def prep_string(self,doc):
        doc1=self.clean_string(doc).split()
        length=len(doc1)
        x_tensor=torch.tensor([self.encoder.wv[w.lower()] for w in doc1])
        return x_tensor.view(1,length,-1)


    def _sent_to_vec(self,sent):
        return torch.tensor([self.encoder.wv[w.lower()] for w in sent.split()])


    def init_hidden(self,batch_size=None):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if batch_size is None:
            return (torch.zeros(self.n_layers, self.n_samples, self.hidden_size),
                    torch.zeros(self.n_layers, self.n_samples, self.hidden_size))
        else:
            return (torch.zeros(self.n_layers, batch_size, self.hidden_size),
                    torch.zeros(self.n_layers, batch_size, self.hidden_size))


    # 'string' should appear as a simple string.

    def fit(self,string):
        loss_function=nn.CrossEntropyLoss()

        optimizer = optim.Adam(self.parameters(),lr=.001)


        # Note that when we call '_make_training_set' we must repackage the string into a list.
        # In general we would pass in a list of documents, here we just have one document.

        Xt,tags=self._make_training_set([string])
        losses=[]


        for epoch in range(self.max_iter):
            self.hidden = self.init_hidden()
            hidden=self.hidden
            optimizer.zero_grad()
            length=self.sent_length

            out, hidden = self.forward(Xt,hidden)
            out2=out[:,length-1,:]
            loss=loss_function(out2,tags)
            #print(float(loss.detach().numpy()))
            losses.append(float(loss.detach().numpy()))
            loss.backward(retain_graph=True)
            optimizer.step()
        self.losses=losses


    def predict(self,sent,n):
        sent_string=sent[0]
        with torch.no_grad():
            hidden=self.init_hidden(batch_size=1)
            length=len(sent[0].split())
            for i in range(n):
                x=self.prep_string(sent[0])
                #x=clean_string(sent).split()
                out,hidden=self.forward(x,hidden)
                out2=out[:,length-1,:]
                probs=F.softmax(out2.div(self.temperature)).detach().numpy()[0]
                word=self.index_to_word(np.random.choice(len(probs),p=probs))
                sent_string=sent_string+' '+ word
        return sent_string
