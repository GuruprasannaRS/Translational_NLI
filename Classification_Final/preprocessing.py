import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import nltk
import re
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec

class datapreprocessing:
    def __init__(self, filepath):
        self.filepath = filepath
        
    def parquet_to_pd(self):
        table = pq.read_table(self.filepath)
        df = table.to_pandas()
        return df
    
    def eng_fre_dataset(self,df):
        df_english = df.drop(columns = ['premise','hypothesis'], axis = 1)
        df_french = df.iloc[:, 2:]
        df_french = df_french[['premise', 'hypothesis', 'label']]
        return df_english, df_french
    
    def english_pipeline(self,df_english):
        # Case Normalization and Tokenization
        df_english['premise_original'] = df_english['premise_original'].str.lower().apply(word_tokenize)
        df_english['hypothesis_original'] = df_english['hypothesis_original'].str.lower().apply(word_tokenize)
        # Special Characters Removal
        df_english['premise_original'] = df_english['premise_original'].apply(lambda x: [re.sub(r'\W+','', item) for item in x])
        df_english['hypothesis_original'] = df_english['hypothesis_original'].apply(lambda x: [re.sub(r'|W+','',item) for item in x])
        # Stemming 
        # stemmer = PorterStemmer()
        # df_english['premise_original'] = df_english['premise_original'].apply(lambda x: [stemmer.stem(word) for word in x])
        # df_english['hypothesis_original'] = df_english['hypothesis_original'].apply(lambda x: [stemmer.stem(word) for word in x])
        #Word2Vec Model
        premise_sentences_e = df_english['premise_original'].tolist()
        hypothesis_sentences_e = df_english['hypothesis_original'].tolist()
        sentences_e = premise_sentences_e + hypothesis_sentences_e
        # print(sentences_e[:2])
        model = Word2Vec(sentences_e)
        model.save('w2ve.model')
        print('Word 2 vec Model for English saved successfully...')
        # Train Dev Test Split
        X_english = df_english.drop(columns = 'label', axis = 1)
        y_english = df_english['label']
        X_train_temp_e, X_test_e, y_train_temp_e, y_test_e = train_test_split(X_english, y_english, test_size=0.2, random_state=42)
        # Split the temporary set into validation and final testing sets
        X_train_e, X_val_e, y_train_e, y_val_e = train_test_split(X_train_temp_e, y_train_temp_e, test_size=0.25, random_state=42)
        print(len(X_train_e))
        print(len(X_val_e))
        print(len(X_test_e))
        print(len(y_train_e))
        print(len(y_val_e))
        print(len(y_test_e))

        return df_english, X_train_e, X_val_e, X_test_e, y_train_e, y_val_e, y_test_e
    
    def french_pipeline(self,df_french, pred_df):
        # Case Normalization and Tokenization
        df_french['premise'] = df_french['premise'].str.lower().apply(lambda x: nltk.word_tokenize(x, language='french'))
        df_french['hypothesis'] = df_french['hypothesis'].str.lower().apply(lambda x: nltk.word_tokenize(x, language='french'))

        # Case Normalization and Tokenization
        pred_df['premise'] = pred_df['premise'].str.lower().apply(lambda x: nltk.word_tokenize(x, language='french'))
        pred_df['hypothesis'] = pred_df['hypothesis'].str.lower().apply(lambda x: nltk.word_tokenize(x, language='french'))


        # # Stemming
        # stemmer = PorterStemmer()
        # df_french['premise'] = df_french['premise'].apply(lambda x: [stemmer.stem(word) for word in x])
        # df_french['hypothesis'] = df_french['hypothesis'].apply(lambda x: [stemmer.stem(word) for word in x])

        #Word2Vec Model
        premise_sentences_f =  df_french['premise'].tolist()
        hypothesis_sentences_f = df_french['hypothesis'].tolist()
        sentences_f = premise_sentences_f + hypothesis_sentences_f
        model = Word2Vec(sentences_f)
        model.save('w2vf.model')
        print('Word 2 vec Model for French saved successfully..')

        # pred_prem_sent = pred_df['pred_premises'].tolist()
        # pred_hypo_sent = pred_df['pred_hypothesis'].tolist()
        # sent = pred_prem_sent + pred_hypo_sent
        # model = Word2Vec(sent)
        # model.save('w2vf.model')
        # print('Word 2 vec Model for French saved successfully..')



        # Train Dev Test Split
        X_french = df_french.drop(columns = 'label', axis = 1)
        y_french = df_french['label']

        pred_X_test = pred_df.drop(columns = 'label', axis = 1)
        pred_y_test = pred_df['label']

        X_train_temp_f, X_test_f, y_train_temp_f, y_test_f = train_test_split(X_french, y_french, test_size=0.2, random_state=42)
        # Split the temporary set into validation and final testing sets
        X_train_f, X_val_f, y_train_f, y_val_f = train_test_split(X_train_temp_f, y_train_temp_f, test_size=0.25, random_state=42)



        # print(len(X_train_f))
        # print(len(X_val_f))
        # print(len(X_test_f))
        # print(len(y_train_f))
        # print(len(y_val_f))
        # print(len(y_test_f))
        # print(X_test_f[:10])
        # Print the sizes of each set
        # print(X_train_f.head(2))
        # print(X_val_f.head(2))
        # # print(X_test_f.head(2))
        # print("Train set size:", len(X_train_f))
        # print("Validation set size:", len(X_val_f))
        # print("Test set size:", len(X_test_f))

        return df_french, X_train_f, X_val_f, pred_X_test, y_train_f, y_val_f, pred_y_test 
    





    


        




        
        


    









