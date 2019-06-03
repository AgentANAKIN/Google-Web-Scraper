# acknowledgements
# https://github.com/vprusso/youtube_tutorials/blob/master/web_scraping_and_automation/beautiful_soup/beautiful_soup_and_requests.py
# https://github.com/llSourcell/web_scraper_live_demo/blob/master/main.py
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/
# https://stackoverflow.com/questions/5598524/can-i-remove-script-tags-with-beautifulsoup
# http://zetcode.com/python/beautifulsoup/
# https://github.com/AgentANAKIN/Dual-Twitter-Sentiment-Analysis-with-4-Text-Summary-Tools-and-Stopwords-Scrubbed-Keywords
# https://github.com/llSourcell/twitter_sentiment_challenge/blob/master/demo.py
# https://youtu.be/qTyj2R-wcks
# https://www.youtube.com/watch?v=8p9nHmtwk0o
# https://github.com/Jcharis/Natural-Language-Processing-Tutorials/blob/master/Text%20Summarization%20with%20Sumy%20Python%20.ipynb
# Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
# https://codeburst.io/python-basics-11-word-count-filter-out-punctuation-dictionary-manipulation-and-sorting-lists-3f6c55420855
# https://pythonspot.com/nltk-stop-words/



# import dependencies: scrape the links off search results pages and the text off web pages
import requests
from bs4 import BeautifulSoup
import re
# import dependencies: sentiment analysis
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# import dependency: pie chart
import matplotlib.pyplot as plt
# import dependencies: text summarization
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import nltk; nltk.download('punkt')
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
# import dependencies: keywords
import nltk; nltk.download('stopwords')
from nltk.corpus import stopwords



# scrape a search engine's homepage
URL = "https://www.google.com/"
result = requests.get(URL)
src = result.content
soup = BeautifulSoup(src, 'lxml')



# create a list of the links on the selected search engine's homepage
links = soup.find_all("a")
stopURLs = []
for link in links:
    if link not in stopURLs:
        stopURLs.append(link.attrs['href'])



# scrape the results page; unfortunately, this URL has to be updated manually
URL = "https://www.google.com/search?source=hp&ei=V6zsXN2RCcemoATqs7OoAw&q=chocolate&oq=chocolate&gs_l=mobile-gws-wiz-hp.3..0j46l2j0l5.3285.5362..5639...0.0..0.219.1763.0j5j4......0....1.......8..41j0i131j46i275.mNHGFzr_jX8"
result = requests.get(URL)
src = result.content
soup = BeautifulSoup(src, 'lxml')



# create a list of links that are not in the list of homepage links, because the homepage links are not results links
links = soup.find_all("a")
URL_List = []
# only add absolute hyperlinks (starting with "http")
pattern = r"http"
for link in links:
    if link not in URL_List:
        if re.match(pattern, str(link.attrs['href'])) is not None:
            URL_List.append(link.attrs['href'])



# improves performance
SIA = SentimentIntensityAnalyzer()



# set some variables
negative = 0
positive = 0
neutral = 0
unknown = 0



# create text files to append tweets to
txtNegative = open('negative.txt','a+')
txtPositive = open('positive.txt','a+')
txtNeutral = open('neutral.txt','a+')
txtUnknown = open('unknown.txt','a+')



i = 0
size_of_URL_List = len(URL_List) 
while i < size_of_URL_List:
    URL = URL_List[i]
    result = requests.get(URL)
    src = result.text
    soup = BeautifulSoup(src,'lxml')
    [s.extract() for s in soup(['iframe', 'script', 'style'])]
    text = soup.get_text()
# classify tweets as negative, positive, neutral, or unknown
# TextBlob and VADER must agree, or the result is "unknown"
    analysisTB = TextBlob(text)
    analysisVS = SIA.polarity_scores(text)
    if ((analysisTB.sentiment.polarity < -0.05) & 
        (analysisVS['compound'] < -0.05)):
        txtNegative.write(text)
        negative += 1
    elif ((analysisTB.sentiment.polarity > 0.05) & 
              (analysisVS['compound'] > 0.05)):
        txtPositive.write(text)
        positive += 1
    elif ((analysisTB.sentiment.polarity > -0.05) & 
              (analysisTB.sentiment.polarity < 0.05) & (analysisVS['compound'] > -0.05) & (analysisVS['compound'] < 0.05)):
        txtNeutral.write(text)
        neutral += 1
    else:
        txtUnknown.write(text)
        unknown += 1
    i += 1



# open file to append summaries to
txtSummary = open('summary.txt','a+')



# print totals on screen and to file
total = negative + positive + neutral + unknown
negative_pct = ((negative / total) * 100)
txtSummary.write("Negative: ")
txtSummary.write(str(negative))
txtSummary.write(" ")
txtSummary.write(str(negative_pct))
txtSummary.write("%")
print("Negative: "+str(negative))

positive_pct = ((positive / total) * 100)
txtSummary.write("\n\nPositive: ")
txtSummary.write(str(positive))
txtSummary.write(" ")
txtSummary.write(str(positive_pct))
txtSummary.write("%")
print("Positive: "+str(positive))

neutral_pct = ((neutral / total) * 100)
txtSummary.write("\n\nNeutral: ")
txtSummary.write(str(neutral))
txtSummary.write(" ")
txtSummary.write(str(neutral_pct))
txtSummary.write("%")
print("Neutral: "+str(neutral))

unknown_pct = ((unknown / total) * 100)
txtSummary.write("\n\nUnknown: ")
txtSummary.write(str(unknown))
txtSummary.write(" ")
txtSummary.write(str(unknown_pct))
txtSummary.write("%")
print("Unknown: "+str(unknown))



# displays the results as a pie chart and explodes the largest slice
labels = ['negative', 'positive', 'neutral', 'unknown']
sizes = [negative, positive, neutral, unknown]
colors = ['red', 'green', 'yellow', 'gray']
if ((negative > positive) & 
    (negative > neutral) & 
    (negative > unknown)):
    explode = [.1, 0, 0, 0]
elif ((positive > negative) & 
      (positive > neutral) & 
      (positive > unknown)):
    explode = [0, .1, 0, 0]
elif ((neutral > negative) & 
      (neutral > positive) & 
      (neutral > unknown)):
    explode = [0, 0, .1, 0]
else:
    explode = [0, 0, 0, .1]
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()



# close files and reopen for reading
txtNegative.close()
txtPositive.close()
txtNeutral.close()
txtUnknown.close()
txtNegative = open('negative.txt','r')
txtPositive = open('positive.txt','r')
txtNeutral = open('neutral.txt','r')
txtUnknown = open('unknown.txt','r')



# stop words are common English words, such as "the;" the lists of keywords will exclude these stop words
stopWords = set(stopwords.words('english'))



# summarize tweets with LexRank, Luhn, LSA, Stop Words
parser = PlaintextParser.from_file("negative.txt", Tokenizer("english"))
LRSummarizer = LexRankSummarizer()
summary = LRSummarizer(parser.document, 1)
txtSummary.write("\n\n*** LEXRANK NEGATIVE ***\n")
print("*** LEXRANK NEGATIVE ***")
for sentence in summary:
    txtSummary.write(str(sentence))
    print(sentence)

LSummarizer = LuhnSummarizer()
summary = LSummarizer(parser.document, 1)
txtSummary.write("\n\n*** LUHN NEGATIVE ***\n")
print("")
print("*** LUHN NEGATIVE ***")
for sentence in summary:
    txtSummary.write(str(sentence))
    print(sentence)

LSASummarizer = LsaSummarizer()
summary = LSASummarizer(parser.document, 1)
txtSummary.write("\n\n*** LSA NEGATIVE ***\n")
print("")
print("*** LSA NEGATIVE ***")
for sentence in summary:
    txtSummary.write(str(sentence))
    print(sentence)

LSA2Summarizer = LsaSummarizer()
LSA2Summarizer = LsaSummarizer(Stemmer("english"))
LSA2Summarizer.stop_words = get_stop_words("english")
txtSummary.write("\n\n*** LSA W/ STOP WORDS NEGATIVE ***\n")
print("")
print("*** LSA W/ STOP WORDS NEGATIVE ***")
for sentence in LSA2Summarizer(parser.document, 1):
    txtSummary.write(str(sentence))
    print(sentence)

# clean text and convert all words to lowercase
Text = open('negative.txt').read()
for char in '-.,:;?!\n':
    Text = Text.replace(char,' ')
Text = Text.lower()
word_list = Text.split()
# initialize dictionary
d = {}
# count instances of each word
for word in word_list:
    if word not in d:
        d[word] = 0
    d[word] += 1
# reverse the key and values so they can be sorted using tuples
# discard common words and words that appear only once
word_freq = []
for key, value in d.items():
    if (value > 1) and (key not in stopWords):
        word_freq.append((value, key))
word_freq.sort(key=lambda tup:(-tup[0], tup[1]))
txtSummary.write("\n\n*** KEYWORDS NEGATIVE ***\n")
print("")
print("*** KEYWORDS NEGATIVE ***")
for word in word_freq:
    txtSummary.write(str(word))
    txtSummary.write("\n")
    print(word)

parser = PlaintextParser.from_file("positive.txt", Tokenizer("english"))
LRSummarizer = LexRankSummarizer()
summary = LRSummarizer(parser.document, 1)
txtSummary.write("\n\n*** LEXRANK POSITIVE ***\n")
print("")
print("*** LEXRANK POSITIVE ***")
for sentence in summary:
    txtSummary.write(str(sentence))
    print(sentence)

LSummarizer = LuhnSummarizer()
summary = LSummarizer(parser.document, 1)
txtSummary.write("\n\n*** LUHN POSITIVE ***\n")
print("")
print("*** LUHN POSITIVE ***")
for sentence in summary:
    txtSummary.write(str(sentence))
    print(sentence)

LSASummarizer = LsaSummarizer()
summary = LSASummarizer(parser.document, 1)
txtSummary.write("\n\n*** LSA POSITIVE ***\n")
print("")
print("*** LSA POSITIVE ***")
for sentence in summary:
    txtSummary.write(str(sentence))
    print(sentence)

LSA2Summarizer = LsaSummarizer()
LSA2Summarizer = LsaSummarizer(Stemmer("english"))
LSA2Summarizer.stop_words = get_stop_words("english")
txtSummary.write("\n\n*** LSA W/ STOP WORDS POSITIVE ***\n")
print("")
print("*** LSA W/ STOP WORDS POSITIVE ***")
for sentence in LSA2Summarizer(parser.document, 1):
    txtSummary.write(str(sentence))
    print(sentence)

# clean text and convert all words to lowercase
Text = open('positive.txt').read()
for char in '-.,:;?!\n':
    Text = Text.replace(char,' ')
Text = Text.lower()
word_list = Text.split()
# initialize dictionary
d = {}
# count instances of each word
for word in word_list:
    if word not in d:
        d[word] = 0
    d[word] += 1
# reverse the key and values so they can be sorted using tuples
# discard common words and words that appear only once
word_freq = []
for key, value in d.items():
    if (value > 1) and (key not in stopWords):
        word_freq.append((value, key))
word_freq.sort(key=lambda tup:(-tup[0], tup[1]))
txtSummary.write("\n\n*** KEYWORDS POSITIVE ***\n")
print("")
print("*** KEYWORDS POSITIVE ***")
for word in word_freq:
    txtSummary.write(str(word))
    txtSummary.write("\n")
    print(word)

parser = PlaintextParser.from_file("neutral.txt", Tokenizer("english"))
LRSummarizer = LexRankSummarizer()
summary = LRSummarizer(parser.document, 1)
txtSummary.write("\n\n*** LEXRANK NEUTRAL  ***\n")
print("")
print("*** LEXRANK NEUTRAL ***")
for sentence in summary:
    txtSummary.write(str(sentence))
    print(sentence)

LSummarizer = LuhnSummarizer()
summary = LSummarizer(parser.document, 1)
txtSummary.write("\n\n*** LUHN NEUTRAL  ***\n")
print("")
print("*** LUHN NEUTRAL ***")
for sentence in summary:
    txtSummary.write(str(sentence))
    print(sentence)

LSASummarizer = LsaSummarizer()
summary = LSASummarizer(parser.document, 1)
txtSummary.write("\n\n*** LSA NEUTRAL  ***\n")
print("")
print("*** LSA NEUTRAL ***")
for sentence in summary:
    txtSummary.write(str(sentence))
    print(sentence)

LSA2Summarizer = LsaSummarizer()
LSA2Summarizer = LsaSummarizer(Stemmer("english"))
LSA2Summarizer.stop_words = get_stop_words("english")
txtSummary.write("\n\n*** LSA W/ STOP WORDS NEUTRAL  ***\n")
print("")
print("*** LSA W/ STOP WORDS NEUTRAL ***")
for sentence in LSA2Summarizer(parser.document, 1):
    txtSummary.write(str(sentence))
    print(sentence)

# clean text and convert all words to lowercase
Text = open('neutral.txt').read()
for char in '-.,:;?!\n':
    Text = Text.replace(char,' ')
Text = Text.lower()
word_list = Text.split()
# initialize dictionary
d = {}
# count instances of each word
for word in word_list:
    if word not in d:
        d[word] = 0
    d[word] += 1
# reverse the key and values so they can be sorted using tuples
# discard common words and words that appear only once
word_freq = []
for key, value in d.items():
    if (value > 1) and (key not in stopWords):
        word_freq.append((value, key))
word_freq.sort(key=lambda tup:(-tup[0], tup[1]))
txtSummary.write("\n\n*** KEYWORDS NEUTRAL ***\n")
print("")
print("*** KEYWORDS NEUTRAL ***")
for word in word_freq:
    txtSummary.write(str(word))
    txtSummary.write("\n")
    print(word)

parser = PlaintextParser.from_file("unknown.txt", Tokenizer("english"))
LRSummarizer = LexRankSummarizer()
summary = LRSummarizer(parser.document, 1)
txtSummary.write("\n\n*** LEXRANK UNKNOWN  ***\n")
print("")
print("*** LEXRANK UNKNOWN ***")
for sentence in summary:
    txtSummary.write(str(sentence))
    print(sentence)

LSummarizer = LuhnSummarizer()
summary = LSummarizer(parser.document, 1)
txtSummary.write("\n\n*** LUHN UNKNOWN  ***\n")
print("")
print("*** LUHN UNKNOWN ***")
for sentence in summary:
    txtSummary.write(str(sentence))
    print(sentence)

LSASummarizer = LsaSummarizer()
summary = LSASummarizer(parser.document, 1)
txtSummary.write("\n\n*** LSA UNKNOWN  ***\n")
print("")
print("*** LSA UNKNOWN ***")
for sentence in summary:
    txtSummary.write(str(sentence))
    print(sentence)

LSA2Summarizer = LsaSummarizer()
LSA2Summarizer = LsaSummarizer(Stemmer("english"))
LSA2Summarizer.stop_words = get_stop_words("english")
txtSummary.write("\n\n*** LSA W/ STOP WORDS UNKNOWN  ***\n")
print("")
print("*** LSA W/ STOP WORDS UNKNOWN ***")
for sentence in LSA2Summarizer(parser.document, 1):
    txtSummary.write(str(sentence))
    print(sentence)

# clean text and convert all words to lowercase
Text = open('unknown.txt').read()
for char in '-.,:;?!\n':
    Text = Text.replace(char,' ')
Text = Text.lower()
word_list = Text.split()
# initialize dictionary
d = {}
# count instances of each word
for word in word_list:
    if word not in d:
        d[word] = 0
    d[word] += 1
# reverse the key and values so they can be sorted using tuples
# discard common words and words that appear only once
word_freq = []
for key, value in d.items():
    if (value > 1) and (key not in stopWords):
        word_freq.append((value, key))
word_freq.sort(key=lambda tup:(-tup[0], tup[1]))
txtSummary.write("\n\n*** KEYWORDS UNKNOWN ***\n")
print("")
print("*** KEYWORDS UNKNOWN ***")
for word in word_freq:
    txtSummary.write(str(word))
    txtSummary.write("\n")
    print(word)



# close files
txtNegative.close()
txtPositive.close()
txtNeutral.close()
txtUnknown.close()
txtSummary.close()
