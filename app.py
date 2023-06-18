import streamlit as st
import numpy as np
import pandas as pd
import snscrape.modules.twitter as sntwitter #Scrapper
import re #regex tools buat manipulasi text
import string #string population
import nltk #Tools
import requests #geturl
from nltk.tokenize import word_tokenize #tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory #Stemmer
from textblob import TextBlob #Tools - IDF - Analisis Sentiment
from indoNLP.preprocessing import replace_slang #slank word
from googletrans import Translator #Translator
from nltk.corpus import stopwords #stopword
################################################################
#Unduh korpus
nltk.download('stopwords') #Mendapatkan list stopword indonesia
nltk.download('punkt') #Mendapatkan kata yang tidak terdapat pada kamus

################################################################
#Modul

#Membuat list sementara
tweets_list = []
positif = []
negatif = []

#Mengolah kata
class Prepocessing:
    def __init__(self):
        factory = StemmerFactory()
        sastrawi = factory.create_stemmer()
        self.listStopword =  set(nltk.corpus.stopwords.words('indonesian'))
        self.stemmer = sastrawi 

    def remove_emoji(self, string): #remove emoji
        emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r' ', string)

    def remove_unwanted(self, document): #clean text
        # remove user mentions
        document = re.sub("@[A-Za-z0-9_]+"," ", document)
        # remove URLS 
        document = re.sub(r'http\S+', ' ', document)
        # remove hashtags
        document = re.sub("#[A-Za-z0-9_]+","", document)
        # remove emoji's
        document = self.remove_emoji(document)
        # remove punctuation
        document = re.sub("[^0-9A-Za-z ]", "" , document)
        # remove double spaces
        document = document.replace('  '," ")
        return document.strip()
    
    def tokenize(self, text): #tokenize -> memisah kalimat 
        return word_tokenize(text.translate(str.maketrans('', '', string.punctuation)).lower())
    
    def stopWord(self, text): #stopword -> menghapus kata hubung
        return [kata for kata in text if kata not in self.listStopword]
    
    def slank_word(self, text): #slank word -> mengganti kata yang tidak baku
        return [replace_slang(kata) for kata in text]

    def stemming(self, text): #stemming -> mengganti kata menjadi kata dasar
        return " ".join([self.stemmer.stem(kata) for kata in text])

def translate_text(text, source_lang, target_lang):
    translator = Translator()
    translation = translator.translate(text, src=source_lang, dest=target_lang)
    return translation.text

def translate_word(word, target_lang):
    url = 'https://translate.googleapis.com/translate_a/single'
    params = {
        'client': 'gtx',
        'sl': 'auto',
        'tl': target_lang,
        'dt': 't',
        'q': word
    }
    response = requests.get(url, params=params)
    translation = response.json()[0][0][0]
    return translation
# hasil_terjemahan = translate_word(kata, bahasa)

def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\s+)"," ",tweet).split())

def clean_text(text):
    clean = Prepocessing()
    result = clean.remove_unwanted(clean_tweet(text))
    result = clean.tokenize(result)
    result = clean.stopWord(result)
    result = clean.slank_word(result)
    result = clean.stemming(result)
    
    return result

# sentimens analisis dengan polarity menggunakan library textblob
def analize_sentiment(tweet):
    result = clean_text(tweet)
    #tld = translate_text(result, 'id', 'en') #Translate tweet(sering error)
    tld = translate_word(result, 'en') #alternatif translate
    analysis = TextBlob(tld)
    if analysis.sentiment.polarity > 0:
        return 'Positif'
    elif analysis.sentiment.polarity == 0:
        pass
    else:
        return 'Negatif'

def filter_Pos(t_list):
    if t_list != [] or t_list != None :
        temp = []
        for i in range(len(t_list)):
            if t_list[i][1] == "Positif":
                temp.append(t_list[i])
            else:
                pass
        return temp
    else:
        return None

def filter_Neg(t_list):
    if t_list != [] or t_list != None :
        temp = []
        for j in range(len(t_list)):
            if t_list[j][1] == "Negatif":
                temp.append(t_list[j])
            else:
                pass
        return temp
    else :
        return None

def filter_pharsePos(comment):
    temp = []
    word_prep = Prepocessing
    word_temp = comment.tolist()
    for i in range(len(word_temp)):
        word_temp = "".join(word_temp[i][0])
        word_temp = word_prep.tokenize(word_temp)
        temp.append(word_temp)
    return temp
        
def Crawling_tweets(jumlah, tokoh = "", tokoh2 = "", tokoh3 = ""):
    pos = 0
    neg = 0
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper( tokoh or tokoh2 or tokoh3 ).get_items()):
        if i > jumlah :
            break
        if analize_sentiment(tweet.content) != "Netral" :
            if analize_sentiment(tweet.content) == "Positif" and pos != 10:
                tweets_list.append([clean_text(tweet.content), analize_sentiment(tweet.content)])
                pos += 1
            elif analize_sentiment(tweet.content) == "Negatif" and neg != 10:
                tweets_list.append([clean_text(tweet.content), analize_sentiment(tweet.content)])
                neg += 1
            else:
                pass
      
    if tweets_list != [] :
        pass
    else:
        Crawling_tweets(jumlah, tokoh, tokoh2, tokoh3)
    
################################################################
#GUI

st.title("Aplikasi PBA-NLP")

# inisialisasi data
tab1, tab2 = st.tabs(["Description data", "Processing"])

with tab1:
    st.subheader("Deskripsi")
    st.write(
        "Analisis Sentimen Terhadap Bakal Calon Presiden 2024 dengan Algoritma Naïve Bayes")
    st.caption("""Presiden di Indonesia dipilih melalui masyarakat dengan melalui proses demokrasi yaitu
pemilihan presiden (pilpres) yang dilaksanakan setiap 5 tahun sekali. Menjadi seorang presiden memiliki beberapa
persyaratan yang dimana persyaratan tersebut adalah seseorang tidak diperbolehkan menjadi presiden apabila orang
tersebut sebelumnya telah menjadi presiden selama 2 periode secara berturut – turut, yang dalam hal ini presiden
Indonesia saat ini sudah tidak bisa mencalonkan kembali menjadi Presiden pada pilpres selanjutnya yang akan terlaksana
pada tahun 2024. Berdasarkan tersebut banyak bermunculan survei elektabilitas terhadap beberapa tokoh publik yang memiliki
elektabilitas baik yang menjadikan tokoh ini bisa dijadikan bakal calon presiden Indonesia di pilpres pada tahun 2024. Berdasarkan penjelasan tersebut penelitian ini akan dilakukan sebuah analisa sentimen terhadap bakal calon
presiden 2024 dengan menggunakan algoritme naïve bayes yang nantinya hasil penelitian ini dapat dimanfaat masyarakat
sebagai bahan referensi dalam memilih pemimpinnya di kemudian hari pada tahun 2024.""")

with tab2:
    st.subheader("Processing Data")
    if st.button("Generate Data"):
        st.write("Output :")
        Crawling_tweets(10, "Ganjar Pranowo", "Prabowo")
        st.write(pd.DataFrame(np.array(tweets_list)))
        st.write("Data yang diperoleh akan di pisah berdasarkan section 'Positif' dan 'Negatif'")
        positif = np.array(filter_Pos(tweets_list))
        negatif = np.array(filter_Neg(tweets_list))
        pos_cln = positif[:]
        neg_cln = negatif[:]
        st.write("Positif :")
        positif = pd.DataFrame(positif)
        positif.columns = ["Comment","Section"]
        st.write(positif)
        st.write("Negatif :")
        negatif = pd.DataFrame(negatif)
        negatif.columns = ["Comment","Section"]
        st.write(negatif)
        st.markdown("Kesimpulan dari beberapa data diatas, analisis sentiment belum dapat mengidentifikasi kalimat sarkas atau kalimat yang bermakna ambigu, sebab tools ini hanya mengidentifikasi berdasarkan polarity saja")
        st.write("Mencoba untuk mengambil kata positif dan negatif di setiap comment")
        splt_pos = filter_pharsePos(pos_cln)
        st.write(splt_pos)
        
    else:
        st.write("Output :")
########################################################################
