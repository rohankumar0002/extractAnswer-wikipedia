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
from nltk.util import ngrams
from math import log
from nltk.stem import SnowballStemmer
from flask import Flask, request, jsonify, make_response, Response
from gevent.pywsgi import WSGIServer
import time


nlp=spacy.load('en')
stop=stopwords.words('english')
key=Rake(min_length=1, stopwords=stop, punctuations=punctuation, max_length=6)
wh_words="who|what|how|where|when|why|which|whom|whose".split('|')
stemmer=SnowballStemmer('english')
blob=textblob.Blobber()
wikipedia.set_rate_limiting(True)
session = HTMLSession()

predictor = Predictor.from_path("bidaf-model-2017.09.15-charpad.tar.gz")

nlp=spacy.load('en')
stop=stopwords.words('english')
key=Rake(min_length=1, stopwords=stop, punctuations=punctuation, max_length=6)
wh_words="who|what|how|where|when|why|which|whom|whose|explain".split('|')
stemmer=SnowballStemmer('english')
blob=textblob.Blobber()
stop.extend(wh_words)
session = HTMLSession()

def termFrequency(term, doc):
    normalizeTermFreq = doc.lower().split()
    normalizeTermFreq=[stemmer.stem(i) for i in normalizeTermFreq]
    dl=len(normalizeTermFreq)
    normalizeTermFreq=' '.join(normalizeTermFreq)
    term_in_document = normalizeTermFreq.count(term)
    #len_of_document = len(normalizeTermFreq )
    #normalized_tf = term_in_document / len_of_document 
    normalized_tf = term_in_document / dl
    return normalized_tf, normalizeTermFreq, dl

def inverseDocumentFrequency(term, allDocs):
    num_docs_with_given_term = 0
    for doc in allDocs: 
        if term in doc: 
            num_docs_with_given_term += 1
    if num_docs_with_given_term > 0: 
        total_num_docs = len(allDocs)
        #if num_docs_with_given_term==total_num_docs:
            #return 0.1
        
        idf_val = log(1+(total_num_docs / num_docs_with_given_term))
        term_split=term.split()
        if len(term_split)==3:
            if len([term_split[i] for i in [0,2] if term_split[i] not in stop])==2:
                return idf_val*1.5
            return idf_val
        return idf_val 
    else: 
        return 0

class extractAnswer:
    def __init__(self):
        self.wiki_error=(wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.HTTPTimeoutError, wikipedia.exceptions.WikipediaException)
        self.article_title=None
        self.symbol = """!#$%^&*();:\n\t\\\"!\{\}\[\]<>-"""
    def extractAnswer_model(self, passage, question, s=0.4, e=0.3, wiki=False):
        if type(passage)==list:
            passage=" ".join(passage)
        if not question[-1]=='?':
            question=question+'?'
        pre=predictor.predict(passage=passage, question=question)
        if wiki:
            
            if max(pre['span_end_probs'])>0.5:
                s=0.12   
            elif max(pre['span_end_probs'])>0.4:
                s=0.13
            elif max(pre['span_end_probs'])>0.35:
                s=0.14
            if max(pre['span_start_probs'])>0.5:
                e=0.12
            elif max(pre['span_start_probs'])>0.4:
                e=0.14
            elif max(pre['span_start_probs'])>0.3:
                e=0.15
            
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
        
    def wiki_passage_api(self,article_list, topic, question):
        passage_list=[]
        normalize_passage_list=[]
        keywords=" ".join(self.noun+self.ques_key+[topic.lower()])
        keywords=re.sub('[{0}]'.format(self.symbol), ' ', keywords).split()
        question=question+' '+topic
        ques_tokens=[stemmer.stem(i.lower()) for i in question.split() if i.lower() not in wh_words]
        print(ques_tokens)
        keywords_bigram = [' '.join(i) for i in list(ngrams(ques_tokens, 2)) if i[0] not in stop and i[1] not in stop]
        if len(ques_tokens)>3:
            keywords_trigram = [' '.join(i) for i in list(ngrams(ques_tokens, 3)) if (i[0] in stop) + (i[2] in stop) +(i[1] in stop)<3]
        else:
            keywords_trigram=[]
        #if len(ques_tokens)>5:
            #keywords_4gram = [' '.join(i) for i in list(ngrams(ques_tokens, 4)) if (i[0] in stop) + (i[2] in stop) +(i[1] in stop)+(i[3] in stop)<4]
        #else:
            #keywords_4gram=[]
        #print(passage_list)
        keywords_unigram=list(set([stemmer.stem(i.lower()) for i in keywords if i.lower() not in stop]))
        keywords=keywords_unigram+list(set(keywords_bigram))+keywords_trigram#+keywords_4gram
        Disambiguation_title={}
        for i,article_title  in enumerate(article_list):
            print(i)
            try:
                passage=wikipedia.summary(article_title)
                #print(passage)
                passage_list.append(self.passage_pre(passage))
                #print(1)
            except wikipedia.exceptions.DisambiguationError as e:
                print(e.options[0], e.options)
                
                Disambiguation_title[article_title]=[]
                for p in range(2):
                    params={'search':e.options[p], 'profile':'engine_autoselect', 'format':'json'}
                    article_url=requests.get('https://en.wikipedia.org/w/api.php?action=opensearch', params=params).json()
                    if not article_url[3]:
                        continue
                    article_url=article_url[3][0]
                    r = session.get(article_url)
                    soup = BeautifulSoup(r.html.raw_html)
                    print(soup.title.string)
                    article_title_dis=soup.title.string.rsplit('-')[0]
                    try:
                        url="https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&explaintext&redirects=1&titles={}".format(article_title_dis)
                        passage=requests.get(url).json()['query']['pages']
                        for i in passage.keys():
                            if 'extract' in passage[i]:
                                passage_list.append(self.passage_pre(passage[i]['extract']))
                    except wikipedia.exceptions.HTTPTimeoutError:
                        passage=wikipedia.summary(article_title_dis)
                        passage_list.append(self.passage_pre(passage))
                    except:
                        continue
                    Disambiguation_title[article_title].append(article_title_dis)
            except:
                passage_list.append('')
                continue
                print(traceback.format_exc())
        if Disambiguation_title:
            for i, j in Disambiguation_title.items():
                idx=article_list.index(i)
                article_list.insert(idx, j[0])
                article_list.insert(idx+1, j[1])
                article_list.remove(i)
        tf=[]
        if not passage_list:
            return 0
        pass_len=[]
        print('Extraction complete')
        for n,i in enumerate(passage_list):
            temp_tf={}
            #print('passage',i,len(i), end='\n\n\n')
            if not i:
                print(n, article_list.pop(n))
                continue
            c=0   
            for j in keywords:
                temp_tf[j], temp_pass, temp_len=termFrequency(j, i)
                if temp_tf[j]:
                    c+=1
            normalize_passage_list.append(temp_pass)
            pass_len.append(temp_len)   
            temp_tf['key_match']=c
            tf.append(temp_tf)
        #print(tf)
        passage_list=[i for i in passage_list if i]
        print(keywords)
        idf={}
        for i in keywords:
            idf[i]=inverseDocumentFrequency(i, normalize_passage_list)
        print(tf, idf)
        tfidf=[]
        for n, i in enumerate(tf):
            tf_idf=0
            key_match_ratio=i['key_match']/len(keywords)
            for j in keywords:
                tf_idf+=(i[j]*idf[j])/(0.3361+(1-0.3361)*((1+log(pass_len[n]))/(1+log(sum(pass_len)/len(pass_len)))))
            tfidf.append(tf_idf*key_match_ratio)
            #tfidf.append(tf_idf)
            #tf_idf=0
        print(tfidf)
        print(article_list)
        if len(passage_list)>1:
            sorted_tfidf=sorted(tfidf,reverse=1)
            idx1=tfidf.index(sorted_tfidf[0])
            passage1=passage_list[idx1]
            article_title=article_list[idx1]
        #print(article_title)
        #tfidf.pop(idx)
            idx2=tfidf.index(sorted_tfidf[1])
            passage2=passage_list[idx2]
            article_title=(article_title, article_list[idx2])
        else:
            if passage_list:
                passage1=passage_list[0]
                passage2=0
            else:
                passage1=0
                passage2=0
        return passage1,passage2, article_title
        
        
    def passage_pre(self, passage):
        #passage=re.findall("[\da-zA-z\.\,\'\-\/\–\(\)]*", passage)
        passage=re.sub('\[[^\]]+\]', '', passage)
        passage=re.sub('\\\\.+\\\\', '', passage)
        passage=re.sub('{.+}', '', passage)
        passage=re.sub('\n', ' ', passage)
        passage=re.sub(' +', ' ', passage)
        
        return passage
    def wiki(self, question, topic):
        if not question:
            return 0
        question=re.sub(' +', ' ', question)
        question=question.title()
        key.extract_keywords_from_text(question)
        self.ques_key=key.get_ranked_phrases()
        doc=nlp(question)
        self.noun=[str(i).lower() for i in doc.noun_chunks if str(i).lower() not in wh_words]
        print(self.ques_key, self.noun)
        question=re.sub('[{0}]'.format(self.symbol), ' ', question)
        if not self.noun + self.ques_key:
            return 0
        article_list=None
        question=question.lower()
        if self.noun:
            if len(self.noun)==2:
                self.noun=question[question.find(self.noun[0]):question.find(self.noun[1])+len(self.noun[1])+1].split()
                if self.noun[0] in stop:
                    self.noun.pop(0)
                print(self.noun)
            article_list=self.wiki_search_api(' '.join(self.noun))
        if self.ques_key and not article_list:
            article_list=self.wiki_search_api(self.ques_key[0])
        if not article_list:
            article_list=self.wiki_search_api(' '.join(self.ques_key))
        if not article_list:
            print('Article not found on wikipedia.')
            return 0
        article_list=list(set(article_list))
        passage1, passage2, article_title=self.wiki_passage_api(article_list, topic, question)
        #print(passage)
        #if not (passage1 and passage2):
            #print('passage1:',passage1)
            #print('\npassage2:',passage2)
            #print('Problem article extraction.')
            #return 0
        if passage1:
            ans=self.extractAnswer_model(passage1, question, s=0.20,e=0.20, wiki=True)
        else:
            ans=0
        if ans:
            if len(ans)>600:
                print(ans)
                print('Repeat')
                ans=self.extractAnswer_model(ans, question, s=0.20,e=0.20, wiki=True)
        else:
            ans=self.extractAnswer_model(passage2, question, s=0.20,e=0.20, wiki=True) if passage2 else 0
        print(ans, '\n', '\n',article_title)
        return ans

extractor=extractAnswer()
        
app = Flask(__name__)

@app.route("/", methods=["POST", "get"])
def ans():
    start=time.time()
    question=request.args.get('question')
    topic=request.args.get('topic')
    passage=request.args.get('passage')
    if not question:
        return jsonify(Status='E', Answer="No question given") 
    if not topic:
        topic=''
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
    http_server = WSGIServer(('0.0.0.0', 7091), app)
    http_server.serve_forever()
        

