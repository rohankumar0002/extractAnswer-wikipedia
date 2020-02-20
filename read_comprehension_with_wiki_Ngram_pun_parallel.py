from rake_nltk import Rake
from string import punctuation
from nltk.corpus import stopwords
from allennlp.predictors.predictor import Predictor
import spacy
import wikipedia
import re
import requests
from requests_html import HTMLSession
from bs4 import BeautifulSoup
import traceback
from nltk.stem import SnowballStemmer
from nltk.util import ngrams
from math import log10
from flask import Flask, request, jsonify
from gevent.pywsgi import WSGIServer
import time
import multiprocessing as mp

NLP = spacy.load('en_core_web_md')
stop = stopwords.words('english')
symbol = r"""!#$%^&*();:\n\t\\\"!\{\}\[\]<>-\?"""
stemmer = SnowballStemmer('english')
wikipedia.set_rate_limiting(True)
session = HTMLSession()
results = 5
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2018.11.30-charpad.tar.gz")
srl = Predictor.from_path('https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz')
key = Rake(min_length=1, stopwords=stop, punctuations=punctuation, max_length=6)
wh_words = "who|what|how|where|when|why|which|whom|whose|explain".split('|')
stop.extend(wh_words)
session = HTMLSession()
output = mp.Queue()

def termFrequency(term, doc):
    normalizeTermFreq = re.sub('[\[\]\{\}\(\)]', '', doc.lower()).split()
    normalizeTermFreq = [stemmer.stem(i) for i in normalizeTermFreq]
    dl = len(normalizeTermFreq)
    normalizeTermFreq = ' '.join(normalizeTermFreq)
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
        term_split = term.split()
        if len(term_split) == 3:
            if len([term_split[i] for i in [0, 2] if term_split[i] not in stop]) == 2:
                return idf_val*1.5
            return idf_val
        return idf_val
    else:
        return 0
def sent_formation(question, answer):
    tags_doc = NLP(question)
    tags_doc_cased = NLP(question.title())
    tags_dict_cased = {i.lower_:i.pos_ for i in tags_doc_cased}
    tags_dict = {i.lower_:i.pos_ for i in tags_doc}
    question_cased = []
    for i in question[:-1].split():
        if tags_dict[i] == 'PROPN' or tags_dict[i] == 'NOUN':
            question_cased.append(i.title())
        else:
            question_cased.append(i.lower())
    question_cased.append('?')
    question_cased = ' '.join(question_cased)
    #del tags_dict,tags_doc, tags_doc_cased
    pre = srl.predict(question_cased)
    verbs = []
    arg1 = []
    for i in pre['verbs']:
        verbs.append(i['verb'])
        if 'B-ARG1' in i['tags']:
            arg1.append((i['tags'].index('B-ARG1'), i['tags'].count('I-ARG1'))\
                if not pre['words'][i['tags'].index('B-ARG1')].lower() in wh_words else \
                    (i['tags'].index('B-ARG2'), i['tags'].count('I-ARG2')))
    arg1 = arg1[0] if arg1 else []
    if not arg1:
        verb_idx = pre['verbs'][0]['tags'].index('B-V')
        verb = pre['words'][verb_idx] if pre['words'][verb_idx] != answer.split()[0].lower() else ''
        subj_uncased = pre['words'][verb_idx+1:] if pre['words'][-1]  not in symbol else \
                        pre['words'][verb_idx+1:-1]
    else:
        verb = ' '.join(verbs)
        subj_uncased = pre['words'][arg1[0]:arg1[0]+arg1[1]+1]
    conj = ''
    if question.split()[0].lower() == 'when':
        conj = ' on' if len(answer.split()) > 1 else ' in'
    subj = []
    for n, i in enumerate(subj_uncased):
        if tags_dict_cased[i.lower()] == 'PROPN' and tags_dict[i.lower()] != 'VERB' or n == 0:
            subj.append(i.title())
        else:
            subj.append(i.lower())
    subj[0] = subj[0].title()
    print(subj)
    print(pre)
    subj = ' '.join(subj)
    sent = "{} {}{} {}.".format(subj, verb, conj, answer if answer[-1] != '.' else answer[:-1])
    return sent

class extractAnswer:
    def __init__(self):
        self.wiki_error = (wikipedia.exceptions.DisambiguationError,
                           wikipedia.exceptions.HTTPTimeoutError,
                           wikipedia.exceptions.WikipediaException)
        self.article_title = None
#        symbol = """!#$%^&*();:\n\t\\\"!\{\}\[\]<>-\?"""
    def extractAnswer_model(self, passage, question, s=0.4, e=0.3, wiki=False):
        if type(passage) == list:
            passage = " ".join(passage)
        if not question[-1] == '?':
            question = question+'?'
        pre = predictor.predict(passage=passage, question=question)
        if wiki:
            if max(pre['span_end_probs']) > 0.5:
                s = 0.12
            elif max(pre['span_end_probs']) > 0.4:
                s = 0.13
            elif max(pre['span_end_probs']) > 0.35:
                s = 0.14
            if max(pre['span_start_probs']) > 0.5:
                e = 0.12
            elif max(pre['span_start_probs']) > 0.4:
                e = 0.14
            elif max(pre['span_start_probs']) > 0.3:
                e = 0.15
        if max(pre['span_start_probs']) > s and max(pre['span_end_probs']) > e:
            key.extract_keywords_from_text(question)
            ques_key = [stemmer.stem(i) for i in ' '.join(key.get_ranked_phrases())]
            key.extract_keywords_from_text(passage)
            pass_key = [stemmer.stem(i) for i in ' '.join(key.get_ranked_phrases())]
            l = len(ques_key)
            c = 0
            for i in ques_key:
                if i in pass_key:
                    c += 1
            if c >= l/2:
                print(max(pre['span_start_probs']),
                      max(pre['span_end_probs']))
                if wiki:
                    return pre['best_span_str'], max(pre['span_start_probs']) + max(pre['span_end_probs'])
                try:
                    ans = sent_formation(question, pre['best_span_str'])
                except:
                    ans = pre['best_span_str']
                    print(traceback.format_exc())
                return ans
            print(ques_key, c, l)
            print(max(pre['span_start_probs']), max(pre['span_end_probs']))
            return 0, 0
        else:
            print(max(pre['span_start_probs']), max(pre['span_end_probs']), pre['best_span_str'])
            return 0, 0

    def wiki_search_api(self, query):
        article_list = []
        try:
            article_list.extend(wikipedia.search(query, results=results))
            print(article_list)
            return article_list
        except self.wiki_error:
            params = {'search': query, 'profile': 'engine_autoselect',
                      'format': 'json', 'limit': results}
            article_list.extend(requests.get('https://en.wikipedia.org/w/api.php?action=opensearch',
                                             params=params).json()[1])
            return article_list
        except:
            print('Wikipedia search error!')
            print(traceback.format_exc())
            return 0
    def wiki_passage_api(self, article_title, article_list, output):
#        Disambiguation_title = {}
        try:
            passage = wikipedia.summary(article_title)
            output.put((article_title, self.passage_pre(passage)))
        except wikipedia.exceptions.DisambiguationError as e:
            print(e.options[0], e.options)
            Disambiguation_pass = {}
            for p in range(2 if len(e.options) > 1 else len(e.options)):
                params = {'search':e.options[p], 'profile':'engine_autoselect', 'format':'json'}
                article_url = requests.get('https://en.wikipedia.org/w/api.php?action=opensearch',
                                           params=params).json()
                if not article_url[3]:
                    continue
                article_url = article_url[3][0]
                r = session.get(article_url)
                soup = BeautifulSoup(r.html.raw_html)
                print(soup.title.string)
                article_title_dis = soup.title.string.rsplit('-')[0].strip()
                if article_title_dis in article_list:
                    print('continue')
                    continue
                try:
                    url = "https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&explaintext&redirects=1&titles={}".format(article_title_dis)
                    passage = requests.get(url).json()['query']['pages']
                    for i in passage.keys():
                        if 'extract' in passage[i]:
                            Disambiguation_pass[article_title_dis] = self.passage_pre(passage[i]['extract'])
                except wikipedia.exceptions.HTTPTimeoutError:
                    passage = wikipedia.summary(article_title_dis)
                    Disambiguation_pass[article_title_dis] = self.passage_pre(passage)
                except:
                    Disambiguation_pass[article_title_dis] = ''
                    continue
            output.put((article_title, Disambiguation_pass))
        except:
            output.put((article_title, ''))
            print(traceback.format_exc())
    def sorting(self, article, question, topic):
        processes = [mp.Process(target=self.wiki_passage_api, args=(article[x], article, output))\
                     for x in range(len(article))]
        for p in processes:
            p.start()
        for p in processes:
            p.join(timeout=3)
        results_p = [output.get() for p in processes]
        article_list = []
        passage_list = []
        for i, j in results_p:
            if type(j) != dict and j:
                article_list.append(i)
                passage_list.append(j)
            elif type(j) == dict and j:
                for k, l in j.items():
                    if l:
                        article_list.append(k)
                        passage_list.append(l)
        normalize_passage_list = []
        start = time.time()
        keywords = " ".join(self.noun+self.ques_key+[topic.lower()])
        keywords = re.sub('[{0}]'.format(symbol), ' ', keywords).split()
        question = question+' '+topic
        ques_tokens = [stemmer.stem(i.lower()) for i in question.split() \
                       if i.lower() not in wh_words]
        print(ques_tokens)
        keywords_bigram = [' '.join(i) for i in list(ngrams(ques_tokens, 2)) \
                           if i[0] not in stop and i[1] not in stop]
        if len(ques_tokens) > 3:
            keywords_trigram = [' '.join(i) for i in list(ngrams(ques_tokens, 3)) \
                                if (i[0] in stop) + (i[2] in stop) + (i[1] in stop) < 3]
        else:
            keywords_trigram = []
        if len(ques_tokens) > 5:
            keywords_4gram = [' '.join(i) for i in list(ngrams(ques_tokens, 4)) \
                              if (i[0] in stop) + (i[2] in stop) +(i[1] in stop)+(i[3] in stop) < 4]
        else:
            keywords_4gram = []
        keywords_unigram = list(set([stemmer.stem(i.lower()) for i in keywords \
                                     if i.lower() not in stop]))
        keywords = keywords_unigram+list(set(keywords_bigram))+keywords_trigram+keywords_4gram
        tf = []
        if not passage_list:
            return 0
        pass_len = []
        #n_u_t=[]
        #key_dict = {i: keywords.count(i) for i in keywords}
        print('Extraction complete')
        #remove_pass={}
        #for n,i in enumerate(passage_list):
            #if len(i)<200 or not i:
                #remove_pass[article_list[n]]=i
                #print(n, article_list[n])
        #passage_list=[i for i in passage_list if i not in remove_pass.values()]
        #article_list=[i for i in article_list if i not in remove_pass.keys()]
        passage_list_copy = passage_list.copy()
        article_list_copy = article_list.copy()
        for i in range(len(passage_list_copy)):
            if passage_list.count(passage_list_copy[i]) > 1:
                passage_list.remove(passage_list_copy[i])
                article_list.remove(article_list_copy[i])
                print('Copy:', article_list_copy[i])
        del passage_list_copy
        del article_list_copy
        for n, i in enumerate(passage_list):
            temp_tf = {}
            c = 0
            for j in keywords:
                temp_tf[j], temp_pass, temp_len = termFrequency(j, i + ' ' + article_list[n])
                if temp_tf[j]:
                    c += 1
            normalize_passage_list.append(temp_pass)
            pass_len.append(temp_len)
            temp_tf['key_match'] = c
            tf.append(temp_tf)
        print(pass_len)
        print(keywords)
        idf = {}
        for i in keywords:
            idf[i] = inverseDocumentFrequency(i, normalize_passage_list)
        #print(tf, idf)
        tfidf = []
        #b=0.333 #for PLN
        b, k = 0.75, 1.2 #for BM25
        avg_pass_len = sum(pass_len)/len(pass_len)
        #pivot=sum(n_u_t)/len(n_u_t)
        for n, i in enumerate(tf):
            tf_idf = 0
            #avg_tf=sum(i.values())/len(i)
            key_match_ratio = i['key_match']/len(keywords)
            for j in keywords:
                #tf_idf+=idf[j]*((log(1+log(1+i[j])))/(1-b+(b*pass_len[n]/avg_pass_len))) #PLN
                tf_idf += idf[j]*(((k+1)*i[j])/(i[j]+k*(1-b+(b*pass_len[n]/avg_pass_len)))) #BM25
            tfidf.append(tf_idf*key_match_ratio)
        tfidf = [i/sum(tfidf)*100 for i in tfidf if any(tfidf)]
        if not tfidf:
            return 0, 0, 0, 0, 0
        print(tfidf)
        print(article_list, len(passage_list))
        if len(passage_list) > 1:
            sorted_tfidf = sorted(tfidf, reverse=1)
            idx1 = tfidf.index(sorted_tfidf[0])
            passage1 = passage_list[idx1]
            #article_title=
            tfidf1 = sorted_tfidf[0]
            idx2 = tfidf.index(sorted_tfidf[1])
            passage2 = passage_list[idx2]
            article_title = (article_list[idx1], article_list[idx2])
            tfidf2 = sorted_tfidf[1]
        else:
            article_title = 0
            tfidf2 = 0
            if passage_list:
                passage1 = passage_list[0]
                tfidf1 = tfidf[0]
                passage2 = 0
            else:
                passage1 = 0
                passage2 = 0
                tfidf1, tfidf2 = 0, 0
        end = time.time()
        print('TFIDF time:', end-start)
        return passage1, passage2, article_title, tfidf1, tfidf2

    def passage_pre(self, passage):
        #passage=re.findall("[\da-zA-z\.\,\'\-\/\–\(\)]*", passage)
        passage = re.sub('\n', ' ', passage)
        passage = re.sub('\[[^\]]+\]', '', passage)
        passage = re.sub('pronunciation', '', passage)
        passage = re.sub('\\\\.+\\\\', '', passage)
        passage = re.sub('{.+}', '', passage)
        passage = re.sub(' +', ' ', passage)
        return passage
    def wiki(self, question, topic=''):
        if not question:
            return 0
        question = re.sub(' +', ' ', question)
        question = question.title()
        key.extract_keywords_from_text(question)
        self.ques_key = key.get_ranked_phrases()
        doc = NLP(question)
        self.noun = [str(i).lower() for i in doc.noun_chunks if str(i).lower() not in wh_words]
        print(self.ques_key, self.noun)
        question = re.sub('[{0}]'.format(symbol), ' ', question)
        if not self.noun + self.ques_key:
            return 0
        article_list = None
        question = question.lower()
        if self.noun:
            if len(self.noun) == 2 and len(" ".join(self.noun).split()) < 6:
                #question1=question
                self.noun = " ".join(self.noun).split()
                if self.noun[0] in stop:
                    self.noun.pop(0)
                self.noun = question[question.index(self.noun[0]):question.index(self.noun[-1]) \
                                     +len(self.noun[-1])+1].split()
                #del question1
                print(self.noun)
            article_list = self.wiki_search_api(' '.join(self.noun))
        if self.ques_key and not article_list:
            article_list = self.wiki_search_api(self.ques_key[0])
        if not article_list:
            article_list = self.wiki_search_api(' '.join(self.ques_key))
        if not article_list:
            print('Article not found on wikipedia.')
            return 0, 0
        article_list = list(set(article_list))
        passage1, passage2, article_title, tfidf1, tfidf2 = self.sorting(article_list,
                                                                         question, topic)
        if passage1:
            ans1, conf1 = self.extractAnswer_model(passage1, question, s=0.20, e=0.20, wiki=True)
        else:
            ans1, conf1 = 0, 0
        if ans1:
            conf2 = 0
            if len(ans1) > 600:
                print(ans1)
                print('Repeat')
                ans1, conf1 = self.extractAnswer_model(ans1, question, s=0.20, e=0.20, wiki=True)
        threshhold = 0.3 if not ((tfidf1- tfidf2) <= 10) else 0.2
        if round(tfidf1- tfidf2) < 5:
            threshhold = 0
        if (tfidf1- tfidf2) > 20:
            threshhold = 0.35
        if (tfidf1- tfidf2) > 50:
            threshhold = 1
        if (passage2 and conf1 < 1.5) or (tfidf1 - tfidf2) < 10:
            ans2, conf2 = self.extractAnswer_model(passage2, question, s=0.20, e=0.20,
                                                   wiki=True) if passage2 else (0, 0)
        title = 0
        if round(conf1, 2) > round(conf2, 2) - threshhold:
            print('ans1')
            ans = ans1
            title = article_title[0] if article_title else 0
        else:
            print('ans2')
            title = article_title[1] if article_title else 0
            ans = ans2
        if not question[-1] == '?':
            question = question+'?'
        try:
            ans = sent_formation(question, ans)
        except:
            print(traceback.format_exc())
        print(ans, '\n', '\n', article_title)
        return ans, title

extractor = extractAnswer()
app = Flask(__name__)
@app.route("/", methods=["POST", "get"])
def ans():
    start = time.time()
    question = request.args.get('question')
    topic = request.args.get('topic')
    passage = request.args.get('passage')
    if not question:
        return jsonify(Status='E', Answer="No question given")
    if not topic:
        topic = ''
    if passage:
        answer = extractor.extractAnswer_model(passage, question)
    else:
        answer, title = extractor.wiki(question, topic)
    end = time.time()
    if answer:
        return jsonify(Status='S', Answer=answer, Time=end-start,
                       Article_from_wikipedia=title if not passage else '')
    else:
        return jsonify(Status='E', Answer=answer, Time=end-start)
if __name__ == "__main__":
    PORT = 7091
    HTTP_SERVER = WSGIServer(('0.0.0.0', PORT), app)
    print('Running on',PORT, '...')
    HTTP_SERVER.serve_forever()
