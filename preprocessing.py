# -*- coding: utf-8 -*-


# Load csv file using pandas.
import pandas as pd
file_path = './Reviews.csv'
data = pd.read_csv(file_path)

# Check for total no. of missings values in data
data.isnull().sum()

# Drop row, if values in Summary is missing. 
data.dropna(subset=['Summary'],inplace = True)

# Only summary and text are useful.
data = data[['Summary', 'Text']]

# Select only those summaries whose length is less than 200 and greater than 100
raw_texts = []
raw_summaries = []

for text, summary in zip(data.Text, data.Summary):
    if 100 < len(text) < 200:
        raw_texts.append(text)
        raw_summaries.append(summary)
        
#for text, summary in zip(raw_texts[:5], raw_summaries[:5]):
#    print('Text:\n', text)
#    print('Summary:\n', summary, '\n\n')
        

# Contractions
contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}

### Text Cleaning

# 1.Convert everything to lowercase

# 2.Remove HTML tags

# 3.Contraction mapping

# 4.Remove (‘s)

# 5.Remove any text inside the parenthesis ( )

# 6.Eliminate punctuations and special characters

# 7.Remove stopwords

# 8.Remove short words


# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english')) 


from bs4 import BeautifulSoup
import re

def cleaning_text(text, num):
    s = text.lower()
    s = BeautifulSoup(s, "lxml").text
    s = re.sub(r'\([^)]*\)', '', s) 
    s = re.sub('"','', s) 
    s = ' '.join([contractions[t] if t in contractions else t for t in s.split(" ")])
    s = re.sub(r"'s\b","",s)
    s = re.sub("[^a-zA-Z]", " ", s) 
    s = re.sub('[m]{2,}', 'mm', s)
    if(num == 0):
        tokens = [w for w in s.split() if not w in stop_words]
    else:
        tokens = s.split()
    long_words = []
    for i in tokens:
        if len(i) > 1:                                                
            long_words.append(i)   
    return (" ".join(long_words)).strip()


# Call the cleaning function on each text
cleaned_text = []
for text in raw_texts:
    cleaned_text.append(cleaning_text(text, 0))
    
# print(cleaned_text[:5])
    
# Call the cleaning function on each summary
cleaned_summary = []
for summary in raw_summaries:
    cleaned_summary.append(cleaning_text(summary, 1))

# print(cleaned_summary[:10])
#print(len(cleaned_text))
#print(len(cleaned_summary))    
    
    
# Make a clean review dataset 
from pandas import DataFrame
reviews = {'text': cleaned_text,
        'summary': cleaned_summary
        }

new_data = DataFrame(reviews, columns= ['text', 'summary'])

# Save dataset in csv format
new_data.to_csv(r'./cleaned_reviews.csv')