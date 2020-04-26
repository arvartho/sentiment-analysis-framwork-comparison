'''
Library developed to accomodate tweeter data preprocessing and feature selection for sentiment
analysis
'''
import re
import pandas as pd
import emoji
from lxml import html
import requests
import os
import string

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import preprocessing

from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize, TweetTokenizer
from nltk.sentiment.util import extract_unigram_feats, mark_negation, parse_tweets_set

class TweetProcessor():
   def __init__(self, handle_negation=False):
      """
      Class constructor used to initialize the module.
      Optional argument 'handle_negation' allows to append '_NEG' as a suffix to words following
      'not'
      """      
      self.handle_negation = handle_negation
      # Custom features
      self.custom_features_score = {'<EXCLAMATION>': 1.6, '<PERIODS>': 1.3, '<QUESTION>': 1.2}
      self.custom_features_pos = {'<EXCLAMATION>': '!', '<PERIODS>': '.', '<QUESTION>': '?'}
      # Load emoji DataFrame
      cur_path = os.path.dirname(os.path.abspath(__file__))
      self.emoji_sentiment_df = pd.read_csv(os.path.join(cur_path, 'EmojiSentimentRanking.csv'))
      emoticons = [':)', ':(', ':-(', ':-)', ':0', ':|', ';-)', ':\'(', ';)', '<3', ':D', ':p', ';p']
      self.emoji_vocabulary = list(emoji.UNICODE_EMOJI.keys())+emoticons
      # Negative word list
      with open(os.path.join(cur_path, './word_lists/negative-words.txt'), 'r', encoding='cp1252') as f:
         self.negative_words = [word.strip() for word in f if word.strip()!='' and not word.startswith(';')]
      # Positive word list   
      with open(os.path.join(cur_path, './word_lists/positive-words.txt'), encoding='cp1252') as f:
         self.positive_words = [word.strip() for word in f if word.strip()!='' and not word.startswith(';')]
      # Stop words
      self.custom_stopwords = {'a', 'about', 'after', 'again', 'get', 'all', 'am', 'an', 'and', 'any', 'are', 'around', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'between', 'both', 'by', 'can', 'd', 'did', 'do', 'does', 'doing', 'during', 'each', 'for', 'from', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'may', 'me', 'my', 'myself', 'now', 'o', 'of', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'own', 're', 's', 'rt', 'same', 'she', "she's", 'should', "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'under', 'until', 've', 'very', 'was', 'w', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'you', "you'd", "you'll", "you're", 'your', 'yours', 'yourself', 'yourselves', "'s", "'ve", "'ll", "'m'"}

      
   def get_pos_tag(self, pos):
      """
      Convert between the PennTreebank tags to simple Wordnet tags
      """      
      if pos.startswith('J'):
         return wn.ADJ
      elif pos.startswith('N'):
         return wn.NOUN
      elif pos.startswith('R'):
         return wn.ADV
      elif pos.startswith('V'):
         return wn.VERB
      return None
   
   def tweet_preprocessing(self, doc):
      """
      Tweet preprocessing method
      """
      # Handle emojis
      doc = emoji_parser(doc)
      # Handle negetion
      doc = re.sub(r' isnt ', r' is not ', doc).strip()
      doc = re.sub(r' arent ', r' are not ', doc).strip()
      doc = re.sub(r' aint ', r' is not ', doc).strip()
      doc = re.sub(r' ain ', r' is not ', doc).strip()
      doc = re.sub(r' wasnt ', r' was not ', doc).strip()
      doc = re.sub(r' wasn ', r' was not ', doc).strip()
      doc = re.sub(r' werent ', r' were not ', doc).strip()
      doc = re.sub(r' dont ', r' do not ', doc).strip()
      doc = re.sub(r' doesnt ', r' does not ', doc).strip()
      doc = re.sub(r' didnt ', r' did not ', doc).strip()
      doc = re.sub(r' wont ', r' will not ', doc).strip()
      doc = re.sub(r' havent ', r' have not ', doc).strip()
      doc = re.sub(r' hasnt ', r' has not ', doc).strip()
      doc = re.sub(r' hadnt ', r' had not ', doc).strip()
      doc = re.sub(r' wouldnt ', r' would not ', doc).strip()
      doc = re.sub(r' shouldnt ', r' should not ', doc).strip()
      doc = re.sub(r' shallnt ', r' shall not ', doc).strip()
      doc = re.sub(r' cannot ', r' can not ', doc).strip()
      doc = re.sub(r' cant ', r' can not ', doc).strip()
      doc = re.sub(r' couldnt ', r' could not ', doc).strip()
      doc = re.sub(r'([a-zA-Z].+)n\?t', r' \1 not ', doc).strip()
      doc = re.sub(r'([a-zA-Z].+)n\'t', r' \1 not ', doc).strip()
      # capture apostrofe suffixes
      doc = re.sub(r'([a-zA-Z].+)\'ve', r' \1 have ', doc).strip()
      doc = re.sub(r'([a-zA-Z].+)\'re', r' \1 are ', doc).strip()
      doc = re.sub(r'([a-zA-Z].+)\'s', r' \1 \'s ', doc).strip()          
      # capture explamation mark (!)
      doc = re.sub(r'(!{2,})', '<EXCLAMATION>.', doc).strip()
      # capture question mark (?)
      doc = re.sub(r'(\?{2,})', '<QUESTION>.', doc).strip()
      # remove numbers
      doc = re.sub(r'[0-9]+', '', doc).strip()    
      # remove links
      doc = re.sub(r'http[s]?.+\b', '', doc).strip() 
      # remove underscores
      doc = re.sub(r'_+', '', doc).strip()
      # remove single letters
      doc = re.sub(r' [a-zA-Z] ', ' ', doc).strip()
      # remove periods (.)
      doc = re.sub(r'(\.)', '', doc).strip()      
      
      # Tweet tokenization with TweetTokenizer module
      tk = TweetTokenizer(strip_handles=True, reduce_len=True)
      tokens = tk.tokenize(doc)
      if self.handle_negation:
         tokens = mark_negation(tokens)
      return tokens
   
   def stop_punct_remover(self, tokens):
      """
      Stop words and punctuation removal method
      """
      punctuation = list(string.punctuation)
      cleaned_tokens = [token for token in tokens 
                           if token not in punctuation and token.lower() not in self.custom_stopwords]
      return cleaned_tokens

   def tweet_pipeline(self, doc):
      """
      Method combining the main tweet preprocessing tasks
      """
      tokens = self.tweet_preprocessing(doc)
      cleaned_tokens = self.stop_punct_remover(tokens)
      return cleaned_tokens
      
   def update_emoji_sentiment_ranking(self,):
      '''
      Updates Emoji Ranking with the latest sentiment score from \
      'http://kt.ijs.si/data/Emoji_sentiment_ranking/index.html'
      '''
      mainUrl = 'http://kt.ijs.si/data/Emoji_sentiment_ranking/index.html'
      print('Parsing %s...' % mainUrl)      
      mainPage = requests.get(mainUrl)
      mainTree = html.fromstring(mainPage.content)
      char = mainTree.xpath('//tbody/tr/td[1]/text()')
      unicode = mainTree.xpath('//tbody/tr/td[3]/text()')
      sentiment = mainTree.xpath('//tbody/tr/td[9]/text()')
      label = mainTree.xpath('//tbody/tr/td[11]/text()')
      print('Upadating  Emoji Sentiment Ranking')
      self.emoji_sentiment_df = pd.DataFrame(data={'character': char, 
                                                   'unicode': unicode, 
                                                   'sentiment':sentiment,
                                                   'label': label})
      self.emoji_sentiment_df.to_csv('EmojiSentimentRanking.csv', index=False)
      print('Emoji Sentiment Ranking updated.')
   
   #
   # Feature selection methods using TF-IDF features and word counts
   #
   def sentiment_vectorizer(self, doc):
      '''
      Conducts TF-IDF feature selection using a vocabulary with the negative and potive words
      '''
      vocabulary = set(self.negative_words+self.positive_words)
      vectorizer = TfidfVectorizer(analyzer='word', 
                                tokenizer=self.tweet_preprocessing,
                                norm='l2',
                                lowercase=False,   
                                vocabulary=vocabulary)
      vectorizer = vectorizer.fit(doc)                    
      return vectorizer
      
   def emoji_vectorizer(self, doc):
      '''
      Conducts TF-IDF feature selection using an emoji vocabulary
      '''
      vocabulary = self.emoji_vocabulary
      vectorizer = TfidfVectorizer(analyzer='word', 
                                tokenizer=self.tweet_preprocessing,
                                norm='l2',
                                lowercase=False,
                                vocabulary=vocabulary)
      vectorizer = vectorizer.fit(doc)
      return vectorizer
   
   def special_vectorizer(self, doc):
      '''
      Conducts TF-IDF feature selection using a vocabulary of the special punctuation intendifiers 
      '''
      vocabulary = self.custom_features_score.keys()
      vectorizer = TfidfVectorizer(analyzer='word', 
                                tokenizer=self.tweet_preprocessing,
                                norm='l2',
                                lowercase=False,
                                vocabulary=vocabulary)
      vectorizer.fit(doc)
      return vectorizer
               
   def tfidf_vectorizer(self, doc):
      '''
      Conducts simple TF-IDF feature selection with predefined parameters
      '''
      doc = list2string(doc)
      vectorizer = TfidfVectorizer(analyzer='word', 
                                min_df=5, 
                                ngram_range=(1, 3),
                                norm='l2', 
                                max_features=5000)
      vectorizer.fit(doc)
      return vectorizer
   
   def special_count_vectorizer(self, doc):
      '''
      Conducts BoW feature using a vocabulary of the special punctuation intendifiers 
      '''
      vocabulary = self.custom_features_score.keys()
      vectorizer = CountVectorizer(analyzer='word', 
                                  lowercase=False,
                                  tokenizer=self.tweet_preprocessing,
                                  vocabulary=vocabulary)
      vectorizer.fit(doc)
      return vectorizer
   
   def emoji_count_vectorizer(self, doc):
      '''
      Conducts BoW feature using an emoji vocabulary
      '''
      vocabulary = self.emoji_vocabulary
      vectorizer = CountVectorizer(analyzer='word', 
                                  lowercase=False,
                                  tokenizer=self.tweet_preprocessing,
                                  vocabulary=vocabulary)
      vectorizer = vectorizer.fit(doc)                             
      return vectorizer
   
   def sentiment_count_vectorizer(self, doc):
      '''
      Conducts BoW feature using a vocabulary with the negative and potive words
      '''
      vocabulary = set(self.negative_words+self.positive_words)
      vectorizer = CountVectorizer(analyzer='word', 
                                  lowercase=False,
                                  tokenizer=self.tweet_preprocessing,
                                  vocabulary=vocabulary)
      vectorizer = vectorizer.fit(doc)
      return vectorizer      

#
# Helper functions  
#
def emoji_parser(doc):
   emoji_list = emoji.emoji_lis(doc)
   doc = doc.encode("ascii", "replace").decode('utf-8')
   for em in emoji_list:
      doc_list = list(doc)
      doc_list[em['location']] = em['emoji']
      doc = ''.join(doc_list)
   return doc
        
def list2string(doc):
   if type(doc) is list:
      doc = ' '.join(doc)
   return doc
