from rake_nltk import Rake
from string import punctuation
from nltk.corpus import stopwords
from allennlp.predictors.predictor import Predictor
import spacy
import wikipedia
import textblob
import re
import requests
from requests_html import HTMLSession
from bs4 import BeautifulSoup
import traceback
from math import log
from nltk.stem import SnowballStemmer
from flask import Flask, request, jsonify, make_response, Response
from gevent.pywsgi import WSGIServer
import time

nlp=spacy.load('en')
stop=stopwords.words('english')
key=Rake(min_length=1, stopwords=stop, punctuations=punctuation, max_length=6)
wh_words="who|what|how|where|when|why|which|whom|whose|explain".split('|')
stemmer=SnowballStemmer('english')
blob=textblob.Blobber()
stop.extend(wh_words)
session = HTMLSession()

predictor = Predictor.from_path("bidaf-model-2017.09.15-charpad.tar.gz")

def termFrequency(term, doc): 
    # Splitting the document into individual terms 
    normalizeTermFreq = doc.lower().split()
    normalizeTermFreq=[stemmer.stem(i) for i in normalizeTermFreq]
    # Number of times the term occurs in the document 
    term_in_document = normalizeTermFreq.count(term)

    # Total number of terms in the document 
    len_of_document = float(len(normalizeTermFreq )) 

    # Normalized Term Frequency 
    normalized_tf = term_in_document / len_of_document 

    return normalized_tf, normalizeTermFreq
def inverseDocumentFrequency(term, allDocs): 
    num_docs_with_given_term = 0
    # Iterate through all the documents 
    #term=stemmer.stem(term.lower())
    for doc in allDocs: 
        if term in doc: 
            num_docs_with_given_term += 1
            #print(num_docs_with_given_term)

    if num_docs_with_given_term > 0: 
        # Total number of documents 
        total_num_docs = len(allDocs) 

        # Calculating the IDF 
        idf_val = log(float(total_num_docs) / num_docs_with_given_term+1) 
        return idf_val 
    else: 
        return 0
wikipedia.set_rate_limiting(True)

class extractAnswer:
    def __init__(self):
        self.wiki_error=(wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.HTTPTimeoutError, wikipedia.exceptions.WikipediaException)
        self.pass_len=None
        self.article_title=None
    def extractAnswer_model(self, passage, question, s=0.4, e=0.3, wiki=False):
        if type(passage)==list:
            passage=" ".join(passage)
        if not question[-1]=='?':
            question=question+'?'
        pre=predictor.predict(passage=passage, question=question)
        if wiki:
            if max(pre['span_start_probs'])>0.3:
                e=0.15
            if max(pre['span_end_probs'])>0.35:
                s=0.17
            if max(pre['span_end_probs'])>0.4:
                s=0.15
                if max(pre['span_end_probs'])>0.5:
                    s=0.12   
        if max(pre['span_start_probs'])>s and max(pre['span_end_probs'])>e:
            key.extract_keywords_from_text(question)
            ques_key=[stemmer.stem(i) for i in ' '.join(key.get_ranked_phrases())]
            key.extract_keywords_from_text(passage)
            pass_key=[stemmer.stem(i) for i in ' '.join(key.get_ranked_phrases())]
            l=len(ques_key)
            c=0
            for i in ques_key:
                if i in pass_key:
                    c+=1
            if c>=l/2:
                print(max(pre['span_start_probs']), max(pre['span_end_probs']))
                return pre['best_span_str']
            print(ques_key, c, l)
            print(max(pre['span_start_probs']), max(pre['span_end_probs']))
            return 0
        else:
            print(max(pre['span_start_probs']), max(pre['span_end_probs']), pre['best_span_str'])
            return 0
        
    def wiki_search_api(self, query):
        article_list=[]
        try:
            article_list.extend(wikipedia.search(query, results=5))
            print(article_list)
            return article_list
        except self.wiki_error:
            params={'search':query, 'profile':'engine_autoselect', 'format':'json', 'limit':5}
            article_list.extend(requests.get('https://en.wikipedia.org/w/api.php?action=opensearch', params=params).json()[1])
            return article_list
        except:
            print('Wikipedia search error!')
            print(traceback.format_exc())
            return 0
        
    def wiki_passage_api(self,article_list, topic):
        passage_list=[]
        normalize_passage_list=[]
        keywords=" ".join(self.noun+self.ques_key+[topic.lower()]).split()
        #print(passage_list)
        keywords=list(set([stemmer.stem(i.lower()) for i in keywords if i.lower() not in stop]))
        #if len(keywords)<=2:
            #article_list=article_list[:3]
        #print(article_list)
        for i,article_title  in enumerate(article_list):
            print(i)
            try:
                passage=wikipedia.summary(article_title)
                #print(passage)
                passage_list.append(self.passage_pre(passage))
                #print(1)
                #return passage, article_title
            except wikipedia.exceptions.DisambiguationError as e:
                print(e.options[0], e.options)
                for p in range(2):
                    params={'search':e.options[p], 'profile':'engine_autoselect', 'format':'json'}
                    article_url=requests.get('https://en.wikipedia.org/w/api.php?action=opensearch', params=params).json()
                    if not article_url[3]:
                        continue
                    article_url=article_url[3][0]
                    r = session.get(article_url)
                    soup = BeautifulSoup(r.html.raw_html)
                    print(soup.title.string)
                    article_title=soup.title.string.rsplit('-')[0]
                    try:
                        url="https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&explaintext&redirects=1&titles={}".format(article_title)
                        passage=requests.get(url).json()['query']['pages']
                        for i in passage.keys():
                            if 'extract' in passage[i]:
                                passage_list.append(self.passage_pre(passage[i]['extract']))
                                #return passage[i]['extract'], article_title

                    except wikipedia.exceptions.HTTPTimeoutError:
                        passage=wikipedia.summary(article_title)
                        passage_list.append(self.passage_pre(passage))
                        #return passage, article_title
            except:
                continue
                print(traceback.format_exc())
                #return 0, 0
        tf=[]
        if not passage_list:
            return 0
        passage_list=[i for i in passage_list if i]
        for i in passage_list:
            temp_tf={}
            #print('passage',i,len(i), end='\n\n\n')
            if not i:
                continue
            c=0   
            for j in keywords:
                temp_tf[j], temp_pass=termFrequency(j, i)
                if temp_tf[j]:
                    c+=1
            normalize_passage_list.append(temp_pass)
                
            temp_tf['key_match']=c
            tf.append(temp_tf)
        #print(tf)
        print(keywords)
        idf={}
        for i in keywords:
            idf[i]=inverseDocumentFrequency(i, normalize_passage_list)
        tf_idf=0
        print(tf, idf)
        tfidf=[]
        for i in tf:
            key_match_ratio=i['key_match']/len(keywords)
            for j in keywords:
                tf_idf+=i[j]*idf[j]
            tfidf.append(tf_idf*key_match_ratio)
            tf_idf=0
        print(tfidf)
        
        max_tfidf=max(tfidf)
        idx=tfidf.index(max_tfidf)
        #print(len(passage_list), idx, len(normalize_passage_list))
        return passage_list, article_title
        
        
    def passage_pre(self, passage):
        self.pass_len=len(passage)
        #passage=re.findall("[\da-zA-z\.\,\'\-\/\–\(\)]*", passage)
        #passage=" ".join(i for i in passage if i)
        passage=re.sub('\[[^\]]+\]', '', passage)
        passage=re.sub('\\\\.+\\\\', '', passage)
        passage=re.sub('{.+}', '', passage)
        #passage=re.sub('\(.+\)','',passage)
        passage=re.sub('\n', ' ', passage)
        passage=re.sub(' +', ' ', passage)
        
        return passage
    def wiki(self, question, topic):
        if not question:
            return 0
        question=question.title()
        key.extract_keywords_from_text(question)
        self.ques_key=key.get_ranked_phrases()
        doc=nlp(question)
        self.noun=[str(i).lower() for i in doc.noun_chunks if str(i).lower() not in wh_words]
        print(self.ques_key, self.noun)
        if not self.noun + self.ques_key:
            return 0
        article_list=None
        if self.noun:
            article_list=self.wiki_search_api(' '.join(self.noun))
        if self.ques_key and not article_list:
            article_list=self.wiki_search_api(self.ques_key[0])
        if not article_list:
            article_list=self.wiki_search_api(' '.join(self.ques_key))
        if not article_list:
            print('Article not found on wikipedia.')
            return 0
        #print()
        article_list=list(set(article_list))
        passage, article_title=self.wiki_passage_api(article_list, topic)
        if not passage:
            print('Problem article extraction.')
            return 0
        #passage=self.passage_pre(passage)
        #print(passage)
        ans=self.extractAnswer_model(passage, question, s=0.22,e=0.22, wiki=True)
#        if not ans:
 #           passage=wikipedia.page(article_title).content[self.pass_len:]
  #          passage=self.passage_pre(self, passage)
  #          ans=self.extractAnswer_model(passage, question, s=0.2)
        print(ans, '\n', '\n',article_list)
        return ans
        

extractor=extractAnswer()
        
app = Flask(__name__)

@app.route("/", methods=["POST", "get"])
def ans():
    start=time.time()
    question=request.args.get('question')
    topic=request.args.get('topic')
    passage=request.args.get('passage')
    if passage:
        answer=extractor.extractAnswer_model(passage, question)
    else:
        answer=extractor.wiki(question, topic)
    end=time.time()
    if answer:
        return jsonify(Status='S', Answer=answer, Time=end-start)
    else:
        return jsonify(Status='E', Answer=answer, Time=end-start)
    
if __name__ == "__main__":
    http_server = WSGIServer(('0.0.0.0', 7080), app)
    #    http_server = WSGIServer(('0.0.0.0', 8083), app,keyfile='../ssl/private.key', certfile='../ssl/certificate.crt')
    http_server.serve_forever()
        