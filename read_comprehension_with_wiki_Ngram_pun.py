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
from math import log, log10
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
results=5
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
    normalizeTermFreq = re.sub('[\[\]\{\}\(\)]','',doc.lower()).split()
    normalizeTermFreq=[stemmer.stem(i) for i in normalizeTermFreq]
    dl=len(normalizeTermFreq)
    normalizeTermFreq=' '.join(normalizeTermFreq)
    term_in_document = normalizeTermFreq.count(term)
    #len_of_document = len(normalizeTermFreq )
    #normalized_tf = term_in_document / len_of_document 
    normalized_tf = term_in_document
    return normalized_tf, normalizeTermFreq, dl#, n_unique_term

def inverseDocumentFrequency(term, allDocs):
    num_docs_with_given_term = 0
    for doc in allDocs: 
        if term in doc: 
            num_docs_with_given_term += 1
    if num_docs_with_given_term > 0: 
        total_num_docs = len(allDocs)
        idf_val = log10(((total_num_docs+1) / num_docs_with_given_term))
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
                if wiki:
                    return pre['best_span_str'], max(pre['span_start_probs'])+ max(pre['span_end_probs'])
                return pre['best_span_str']
            
            print(ques_key, c, l)
            print(max(pre['span_start_probs']), max(pre['span_end_probs']))
            return 0,0
        else:
            print(max(pre['span_start_probs']), max(pre['span_end_probs']), pre['best_span_str'])
            return 0,0
        
    def wiki_search_api(self, query):
        article_list=[]
        try:
            article_list.extend(wikipedia.search(query, results=results))
            print(article_list)
            return article_list
        except self.wiki_error:
            params={'search':query, 'profile':'engine_autoselect', 'format':'json', 'limit':results}
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
        if len(ques_tokens)>5:
            keywords_4gram = [' '.join(i) for i in list(ngrams(ques_tokens, 4)) if (i[0] in stop) + (i[2] in stop) +(i[1] in stop)+(i[3] in stop)<4]
        else:
            keywords_4gram=[]
        keywords_unigram=list(set([stemmer.stem(i.lower()) for i in keywords if i.lower() not in stop]))
        keywords=keywords_unigram+list(set(keywords_bigram))+keywords_trigram+keywords_4gram
        Disambiguation_title={}
        for i,article_title  in enumerate(article_list):
            print(i)
            try:
                passage=wikipedia.summary(article_title)
                passage_list.append(self.passage_pre(passage))
            except wikipedia.exceptions.DisambiguationError as e:
                print(e.options[0], e.options)
                
                Disambiguation_title[article_title]=[]
                for p in range(2 if len(e.options)>1 else len(e.options)):
                    params={'search':e.options[p], 'profile':'engine_autoselect', 'format':'json'}
                    article_url=requests.get('https://en.wikipedia.org/w/api.php?action=opensearch', params=params).json()
                    if not article_url[3]:
                        continue
                    article_url=article_url[3][0]
                    r = session.get(article_url)
                    soup = BeautifulSoup(r.html.raw_html)
                    print(soup.title.string)
                    article_title_dis=soup.title.string.rsplit('-')[0].strip()
                    if article_title_dis in article_list:
                        print('continue')
                        continue
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
                        passage_list.append('')
                        continue
                    Disambiguation_title[article_title].append(article_title_dis)
            except:
                passage_list.append('')
                continue
                print(traceback.format_exc())
        if Disambiguation_title:
            for i, j in Disambiguation_title.items():
                idx=article_list.index(i)
                for k in j:
                    article_list.insert(idx, k)
                    idx=idx+1
                article_list.remove(i)
        tf=[]
        if not passage_list:
            return 0
        pass_len=[]
        #n_u_t=[]
        key_dict={i: keywords.count(i) for i in keywords}
        print('Extraction complete')
        remove_pass={}
        for n,i in enumerate(passage_list):
            if len(i)<200 or not i:
                remove_pass[article_list[n]]=i
                print(n, article_list[n])
        passage_list=[i for i in passage_list if i not in remove_pass.values()]
        article_list=[i for i in article_list if i not in remove_pass.keys()]
        passage_list_copy=passage_list.copy()
        article_list_copy=article_list.copy()
        for i in range(len(passage_list_copy)):
            if passage_list.count(passage_list_copy[i])>1:
                passage_list.remove(passage_list_copy[i])
                article_list.remove(article_list_copy[i])
                print('Copy:', article_list_copy[i])
        del passage_list_copy
        del article_list_copy
        for n,i in enumerate(passage_list):
            temp_tf={}
            c=0   
            for j in keywords:
                temp_tf[j], temp_pass, temp_len=termFrequency(j, i)
                if temp_tf[j]:
                    c+=1
            normalize_passage_list.append(temp_pass)
            pass_len.append(temp_len)   
            temp_tf['key_match']=c
            tf.append(temp_tf)
        print(pass_len)
        
        print(keywords)
        idf={}
        for i in keywords:
            idf[i]=inverseDocumentFrequency(i, normalize_passage_list)
        print(tf, idf)
        tfidf=[]
        b=0.333
        avg_pass_len=sum(pass_len)/len(pass_len)
        #pivot=sum(n_u_t)/len(n_u_t)
        for n, i in enumerate(tf):
            tf_idf=0
            avg_tf=sum(i.values())/len(i)
            key_match_ratio=i['key_match']/len(keywords)
            for j in keywords:
   #             tf_idf+=(i[j]*idf[j])/(0.3361+(1-0.3361)*((1+log(pass_len[n]))/(1+log(sum(pass_len)/len(pass_len)))))
                #pun=(((1+log(1+i[j] if i[j] else 1))/(1+log(1+avg_tf if avg_tf else 1)))/((1-slope)*pivot+slope*n_u_t[n]))
                #tf_idf+=(i[j]*idf[j])/pun
                tf_idf+=(key_dict[j]*idf[j])*((log(1+log(1+i[j])))/(1-b+(b*pass_len[n]/avg_pass_len)))
            tfidf.append(tf_idf*key_match_ratio)
        tfidf=[i/sum(tfidf)*100 for i in tfidf if any(tfidf)]
        if not tfidf:
            return 0,0,0,0,0
        print(tfidf)
        print(article_list, len(passage_list))  
        if len(passage_list)>1:
            sorted_tfidf=sorted(tfidf,reverse=1)
            idx1=tfidf.index(sorted_tfidf[0])
            passage1=passage_list[idx1]
            #article_title=
            tfidf1=sorted_tfidf[0]
            idx2=tfidf.index(sorted_tfidf[1])
            passage2=passage_list[idx2]
            article_title=(article_list[idx1], article_list[idx2])
            tfidf2=sorted_tfidf[1]
        else:
            article_title=0
            tfidf2=0
            if passage_list:
                passage1=passage_list[0]
                tfidf1=tfidf[0]
                passage2=0
            else:
                passage1=0
                passage2=0
                tfidf1, tfidf2=0,0
        return passage1,passage2, article_title, tfidf1, tfidf2
        
        
    def passage_pre(self, passage):
        #passage=re.findall("[\da-zA-z\.\,\'\-\/\–\(\)]*", passage)
        passage=re.sub('\n', ' ', passage)
        passage=re.sub('\[[^\]]+\]', '', passage)
        passage=re.sub('pronunciation','', passage)
        passage=re.sub('\\\\.+\\\\', '', passage)
        passage=re.sub('{.+}', '', passage)
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
            if len(self.noun)==2 and len(" ".join(self.noun).split())<6:
                question1=question.split()
                self.noun=" ".join(self.noun).split()
                self.noun=question1[question1.index(self.noun[0]):question1.index(self.noun[-1])+1]
                del question1
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
            return 0, 0
        article_list=list(set(article_list))
        passage1, passage2, article_title, tfidf1, tfidf2=self.wiki_passage_api(article_list, topic, question)
        if passage1:
            ans1, conf1=self.extractAnswer_model(passage1, question, s=0.20,e=0.20, wiki=True)
        else:
            ans1,conf1=0,0
        if ans1:
            conf2=0
            if len(ans1)>600:
                print(ans1)
                print('Repeat')
                ans1, conf1=self.extractAnswer_model(ans1, question, s=0.20,e=0.20, wiki=True)
        threshhold=0.3 if not ((tfidf1- tfidf2)<0.2 or(-tfidf1+ tfidf2)<0.2) else 0.2
        if (passage2 and conf1<1.5) or (tfidf1- tfidf2)<10:
            ans2, conf2=self.extractAnswer_model(passage2, question, s=0.20,e=0.20, wiki=True) if passage2 else (0,0)
        title=0
        if round(conf1>conf2, 2)-threshhold:
            print('ans1')
            ans=ans1
            title=article_title[0] if article_title else 0
        else:
            print('ans2')
            title=article_title[1] if article_title else 0
            ans=ans2
        print(ans, '\n', '\n',article_title)
        return ans, title

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
        answer, title=extractor.wiki(question, topic)
    end=time.time()
    if answer:
        return jsonify(Status='S', Answer=answer, Time=end-start, Article_from_wikipedia=title)
    else:
        return jsonify(Status='E', Answer=answer, Time=end-start)
    
if __name__ == "__main__":
    http_server = WSGIServer(('0.0.0.0', 7091), app)
    http_server.serve_forever()
        

