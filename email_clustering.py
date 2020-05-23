#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD 
from sklearn.preprocessing import normalize 
import nltk.stem.porter
porter_stemmer=nltk.stem.porter.PorterStemmer()
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer() 
import nltk
from sklearn.metrics.pairwise import linear_kernel


def parse_into_emails(messages):
    emails = [raw_message(message) for message in messages]
    return {
        'body': map_to_list(emails, 'body'), 
        'to': map_to_list(emails, 'to'), 
        'from_': map_to_list(emails, 'from')
    }

#This function will extract raw message from the message object
def raw_message(raw_message):
    lines = raw_message.split('\n')  #Here, the paragraph contains "\n" indicating the new line. Hence, the sentence tokenizer is not used.
    email = {}
    message = ''
    keys_to_extract = ['from', 'to']
    for line in lines:
        if ':' not in line:
            message += line.strip()		#removes all the spaces and appends only the text body to the variable "message"
            email['body'] = message
        else:
            pairs = line.split(':')	#This enables to divide each sentence into 2 parts
            key = pairs[0].lower()
            val = pairs[1].strip()
            if key in keys_to_extract:
                email[key] = val
    return email			#returns an array with key-value pairs


def map_to_list(emails, key):
    results = []
    for email in emails:
        if key not in email:
            results.append('')
        else:
            results.append(email[key])
    return results


#reading the dataset
emails = pd.read_csv('split_emails.csv')
#creating a dataframe
email_df = pd.DataFrame(parse_into_emails(emails.message))	

# Drop emails with empty body, to or from_ columns.
email_df.drop(email_df.query("body == '' | to == '' | from_ == ''").index, inplace=True)

#finding top words
stopwords = ENGLISH_STOP_WORDS.union(['ect', 'hou', 'com', 'recipient'])

vect = TfidfVectorizer(analyzer='word', stop_words=stopwords, max_df=0.3, min_df=2)

# max_df=0.5 means "ignore all terms that appear in more then 50% of the documents"
# min_df=2 means "ignore all terms that appear in less then 2 documents"