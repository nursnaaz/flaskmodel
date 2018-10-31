from flask import Flask, render_template, request
from general_utils import create_nmslib_search_index
import nmslib
from lang_model_utils import Query2Emb
from pathlib import Path
import numpy as np
from lang_model_utils import load_lm_vocab
import torch,cv2
from lang_model_utils import lm_vocab, load_lm_vocab, train_lang_model
from general_utils import save_file_pickle, load_file_pickle
import logging
from pathlib import Path
from fastai.text import *
from langdetect import detect
import re
from email_reply_parser import EmailReplyParser
from bs4 import BeautifulSoup

import logging
logging.getLogger().setLevel(logging.ERROR)

app = Flask(__name__)

class search_engine:
    def __init__(self, 
                 nmslib_index, 
                 ref_data, 
                 query2emb_func):
        
        self.search_index = nmslib_index
        self.data = ref_data
        self.query2emb_func = query2emb_func
    
    def search(self, str_search, k=5):
        query = self.query2emb_func(str_search)
        idxs, dists = self.search_index.knnQuery(query, k=k)
        res = []
        for idx, dist in zip(idxs, dists):
            print(idx)  
            a = 'cosine dist: '+str(dist)+" "+ self.data[idx]

            res.append(a)
            #print(f'cosine dist:{dist:.4f}\n---------------\n', self.data[idx])
        return res 


def detectlang(x):
    try:
        lang = detect(x)
        row = lang
    except:
        row = "error"
    return row


def preprocessing_english(x):
    x = BeautifulSoup(x)
    x = EmailReplyParser.parse_reply(x.get_text())
    x = re.sub(r'<.*?>', '', x)
    x = x.replace("\n", " ").strip()
    x = re.sub(pattern=r'[\!"#$%&\*+,-./:;<=>?@^_`()|~=]', repl='', string=x)
    x = x.replace("\n", " ").strip()
    x = x.strip()
    x = re.sub(r"(^|\W)\d+", "", x)
    x = x.lower()
    x = re.sub(r'[^a-zA-Z]', ' ',x)
    x = re.sub("\s\s+", " ", x)

    
    stopwords = {'forwarded','message','lz','logitech','dear', 'my', 'date', 'i', 'recently', 
                 'hi', 'hello', 'product', 'serial', 'number', '1', '2', '3', '4', '5', '6', 
                 '7', '8', '9', '0', 'purchased', 'purchase', 'support', 'http', 'com', 
                 'logitech', 'www', 'https', 'logi', 'customercare','contact', 'terms', 'blvd',
                 'gateway', 'newark', 'usa', 'logo' ,'care', 'ca', 'footer', 'use', 
                 'customer', 'owned', 'us', 'survey', 'americas', 'copyright', 'headquarters', 
                 'owners', 'respective', 'the','rights', 'trademarks', 'reserved', 'property','dear','regards','thanks', 
                 'mail', 'email','lz','g','x','k','date','like','get','one','set','thank',
                 'also','two','see','able','n','could','since','last','know','still','got','pm','p','n','s'
              'operating','system','platform','ce','s','hs','y','mr', 'de','lfcm','sy','m','kh','w','ks','hs','afternoon','morning','regards','thx'
              'thanks','fri', 'mon', 'tue', 'wed', 'thu', 'sat', 'sun', 'jan', 'feb',
               'mar', 'apr', 'may', 'jun', 'jul', 'sep', 'oct', 'nov', 'dec'}

    x = x.split()
    x = [word for word in x if word.lower() not in stopwords]
    x = ' '.join(x)
    return x


def preprocessing_french(x):
    x = BeautifulSoup(x)
    x = EmailReplyParser.parse_reply(x.get_text())
    x = re.sub(r'<.*?>', '', x)
    x = x.replace("\n", " ").strip()
    x = re.sub(pattern=r'[\!"#$%&\*+,-./:;<=>?@^_`()|~=]', repl='', string=x)
    x = x.replace("\n", " ").strip()
    x = x.strip()
    x = re.sub(r"(^|\W)\d+", "", x)
    x = x.lower()
    
    stopwords = {'merci','de', 'nous', 'aider', 'au', 'plus', 'vite', 'bonjour', 'la','le','en',
               'message','cordialement','logitech','cher', 'mon', 'date', 'je', 'récemment', 'salut', 
               'produit', 'en série', 'nombre','achat', 'soutien', 'http', 'com', 'vous',
               'logitech', 'www', 'https','logi', 'service à la clientèle', 'contact', 'termes', 
               'passerelle', 'newark', 'usa', 'logo' ,'care', 'ca', 'footer', 'use', 'customer', 
               'owned', 'us', 'survey', 'americas', 'copyright', 'headquarters', 'owners', 'number',
               'respective','the','rights', 'trademarks', 'reserved', 'property','dear','regards','thanks', 
               'mail', 'email','date','like','get','one','set','thank','also',
               'two', 'see','able','could', 'since','last','know','still','got','pm','p',
               'puisque','operating','system','platform','ce', 'mr', 'de','lfcm',
               'sy','m','kh','w','ks','hs','afternoon','morning','regards','thx'
               'thanks', 'fri', 'mon', 'tue', 'wed', 'thu', 'sat', 'sun', 'jan', 'feb',
               'mar', 'apr', 'may', 'jun', 'jul', 'sep', 'oct', 'nov', 'dec'}

    x = x.split()
    x = [word for word in x if word.lower() not in stopwords]
    x = ' '.join(x)
    return x


@app.route('/')
def student():
   result = []   
   return render_template('message.html',result = result)

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':

      maildata = request.form['Name']
      top = int(request.form['top'])
      language = detectlang(maildata)
      print(language)
      print(maildata)
      if(language == "fr"):
            maildata_preprocess = preprocessing_french(maildata)
            result = se_french.search(maildata_preprocess,top)
      elif(language == "en"):
             maildata_preprocess = preprocessing_english(maildata)
             result = se_english.search(maildata_preprocess,top) 
      else: 
            res = str(language)+" Language is not supported. English and French is supported for now."
            result = [res]    
      print(result)
      name = str(maildata)
      print(name)
      #if(language == "fr"):
      #      lang = "French"
      #elif(language == "en"):
      #      lang = "English"
      #else:
      #      lang = "Not Supported"
      #result = [1,2,3,4,5,6,7]
      return render_template("result.html",name = name, result = result,language = language)

def load(source,data):
      print("load")

      # Load matrix of vectors
      #loadpath = Path('../../data_french_sound/lang_model_emb/')
      #avg_emb_dim500 = np.load(loadpath/'avg_emb_dim500_test_v2.npy')

      # Build search index (takes about an hour on a p3.8xlarge)
      #dim500_avg_searchindex = create_nmslib_search_index(avg_emb_dim500)

      # save search index
      #dim500_avg_searchindex.saveIndex('../../data_french_sound/lang_model_emb/dim500_avg_searchindex.nmslib')


      # Note that if you did not train your own language model and are downloading the pre-trained model artifacts instead, you can similarly download the pre-computed search index here: 
      # 
      # https://storage.googleapis.com/kubeflow-examples/code_search/data/lang_model_emb/dim500_avg_searchindex.nmslib

      # After you have built this search index with nmslib, you can do fast nearest-neighbor lookups.  We use the `Query2Emb` object to help convert strings to the embeddings: 

      print(source)
      dim500_avg_searchindex = nmslib.init(method='hnsw', space='cosinesimil')
      dim500_avg_searchindex.loadIndex(source+'/lang_model_emb/dim500_avg_searchindex.nmslib')

      lang_model = torch.load(source+'/lang_model/lang_model_cpu_v2.torch')
      vocab = load_lm_vocab(source+'/lang_model/vocab_v2.cls')

      q2emb = Query2Emb(lang_model = lang_model.cpu(),
                        vocab = vocab)

      # The method `Query2Emb.emb_mean` will allow us to use the langauge model we trained earlier to generate a sentence embedding given a string.   Here is an example, `emb_mean` will return a numpy array of size (1, 500).
      query = q2emb.emb_mean('Read data into pandas dataframe')
      query.shape

      # **Make search engine to inspect semantic similarity of phrases**.  This will take 3 inputs:
      # 
      # 1. `nmslib_index` - this is the search index we built above.  This object takes a vector and will return the index of the closest vector(s) according to cosine distance.  
      # 2. `ref_data` - this is the data for which the index refer to, in this case will be the docstrings. 
      # 3. `query2emb_func` - this is a function that will convert a string into an embedding.

      source_path = Path(source+'/processed_data/')

      with open(source_path/data, 'r') as f:
            trn_raw = f.readlines()

      with open(source_path/data, 'r') as f:
            val_raw = f.readlines()
      
      with open(source_path/data, 'r') as f:
            test_raw = f.readlines()


      se = search_engine(nmslib_index=dim500_avg_searchindex, ref_data = test_raw,
                        query2emb_func = q2emb.emb_mean)
      #se.search("sound not working")
      print("done")
      return se
      #a = se.search('Audio is not working')



if __name__ == '__main__':
   from pathlib import Path
   english = "../../data_gaming_english/"
   french = "../../data_french_sound/"
   french_data = "data_inp_sound.txt"
   english_data = "data_inp.txt"
   se_english = load(english,english_data)
   se_french = load(french,french_data)
   
   print("loaded") 
   app.run(debug = True,host='0.0.0.0', port=5000 )