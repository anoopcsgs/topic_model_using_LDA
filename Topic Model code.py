##Libraries required: nltk, gensim, pyLDAvis (even if lyLDAvis doesn't install, it's okay - only the last part will not run


#import libraries
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import pandas as pd
from gensim.models import ldamodel
from gensim import corpora
import pyLDAvis.gensim as gensimvis
import pyLDAvis
import re
from gensim.utils import ClippedCorpus


##read news headlines (you can use a cursor to read all news headlines directly into data below. I am passing all news headlines to a list news below


data = pd.read_csv('News.csv')
news = list(data['News'])


##stop words and pre-processing

stop = set(stopwords.words('english')+ ['ba','bthe','b',"r", "n", "amp", 
           "girl","woman","world",'u',"year", "u", ]) 
lemma = WordNetLemmatizer()
def clean(doc):
    punc_free = re.sub("[^a-zA-Z]"," ", doc)
    stop_free = " ".join([i for i in punc_free.lower().split() if i not in stop])
    normalized = " ".join(lemma.lemmatize(word) for word in stop_free.split())
    return normalized
data_clean = [clean(doc).split() for doc in news]


##dictionary to create corporate and fit bag of words matrix


dictionary = corpora.Dictionary(data_clean)
dictionary.filter_extremes(no_below = 100, no_above = 0.05)
d_mat = [dictionary.doc2bow(doc) for doc in data_clean]


##creating corpus for LDA model

clipped_corpus = ClippedCorpus(d_mat, 5000)
Lda = ldamodel.LdaModel
ldamod = Lda(clipped_corpus, num_topics = 20, id2word = dictionary, passes = 15, random_state = 20)
ldamod.print_topics(20)

##seeing in notebook:: you can ignore this step

pyLDAvis.enable_notebook()

##interactive visual for LDA
plot = gensimvis.prepare(ldamod,clipped_corpus, dictionary)
pyLDAvis.display(plot)

