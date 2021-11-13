import re
import demoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import spacy

class Data_Preprocessing: 
    def __init__(self,model=None):
        if(model!=None):
            self.nlp = spacy.load(model)
    
    def removeEmojis(self,text):
        try:
            return demoji.replace(text,repl="")
        except Exception as e:
            return text

    def removeUrls(self,text):
        result =re.sub(r"http\S+", "", text)
        return result

    def removeSpecialChar(self,text):
        s = re.sub(r'[^\w\s]','',text) # remove punctutations
        res = re.sub('[^a-zA-Z.\d\s]', '', s)
        return res

    def removeStopWords(self,text,stop_words):
        try:
            word_tokens = word_tokenize(text)
            filtered_text=[w for w in word_tokens if not w in stop_words]
            # add space to all but last word
            res = ''.join([w+' ' for w in filtered_text[:-1]])
            res+=filtered_text[-1]
            return res
        except Exception as e:
            return text

    def lemmatise(self,text):
        try:
            doc = self.nlp(text)
            res =" ".join([token.lemma_ for token in doc])
            return res
        except Exception as e:
            return text

## Testing the functions 
# text = "Hello 🥳😂😎🙂🤩🙏😍🤬😡😘😘😗🥰😉😌🤪🧐😧😬😈"
# text2 ="This is a tweet with a url: http://googlegroups.com./"
# text3 = "Hel@#o$%^&g*e()ga_+!<aga>?eg:e][aga';/.ege,~~.ag\|*ag-e a."
# text4 = "This is a sample sentence, showing off the stop words filtration."
# text5 = "kites babies rocks rocky feet driving"

# Stop Words Generation
# stop_words = set(stopwords.words('english'))
# for different languages
# https://advertools.readthedocs.io/en/master/advertools.stopwords.html

# Ref on Data Cleaning
# https://radiant-brushlands-42789.herokuapp.com/towardsdatascience.com/all-you-need-to-know-about-text-preprocessing-for-nlp-and-machine-learning-bc1c5765ff67


