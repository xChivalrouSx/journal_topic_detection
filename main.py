import datetime

import re
import numpy as np
import pandas as pd
from pprint import pprint

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import spacy

import pyLDAvis
import pyLDAvis.gensim 
import matplotlib.pyplot as plt

import os
import operator
from gensim.models.wrappers import LdaMallet

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

from wordcloud import WordCloud

from nltk.corpus import stopwords


stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 've'])

full_path = os.path.realpath(__file__)
execution_directory = os.path.dirname(full_path)
print("Execution Path : " + execution_directory)

os.environ.update({'MALLET_HOME': execution_directory + '/mallet-2.0.8/'})

df = pd.read_csv(execution_directory + '/test_data/data_en.csv', error_bad_lines=False)

print(df.head(10))
print("==========================================")
print(df.shape[0])
print(df.shape[1])
print("==========================================")

# Convert to list
# data = df.text.values.tolist()
data = df.text.values.tolist()

print("remove start")
print(datetime.datetime.now())
# Remove Emails
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]

# Remove distracting double quotes
data = [re.sub("\"", "", sent) for sent in data]

# Remove url
url_regex = "(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"
data = [re.sub(url_regex, "", sent) for sent in data]

# Remove Turkish Characters
data = [re.sub('\S*[öÖüÜığĞçÇşŞ]\S*\s?', ' ', sent) for sent in data]

# Remove (*)
data = [re.sub('\(\w*\)', ' ', sent) for sent in data]

# Remove [*]
data = [re.sub('\[\w*\]', ' ', sent) for sent in data]    

# Remove asci
data = [re.sub("[^\x00-\x7F]", "", sent) for sent in data]
print("remove end")
print(datetime.datetime.now())

##### pprint(data[:1])

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  
        
# deacc=True removes punctuations
data_words = list(sent_to_words(data))

##### print(data_words[:1])

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
##### print(trigram_mod[bigram_mod[data_words[0]]])

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

##### print(data_lemmatized[:1])

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
##### print(corpus[:1])

##### print(corpus[:1])
##### [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

print("LDA model start")
print(datetime.datetime.now())
# Build LDA model
topic_count = 30

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=topic_count, 
                                        random_state=100,
                                        update_every=1,
                                        chunksize=100,
                                        passes=10,
                                        alpha='auto',
                                        per_word_topics=True)
print("LDA model end")
print(datetime.datetime.now())

# Print the Keyword in the 10 topics
print(lda_model.print_topics())
# lda_model.save(execution_directory + '/result_data/lda.model')
# doc_lda = lda_model[corpus]

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# Visualize the topics
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
# vis
# pyLDAvis.save_html(vis, execution_directory + '/result_data/lda.html')

# Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
# update this path
mallet_path = execution_directory + '/mallet-2.0.8/bin/mallet'
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=topic_count, id2word=id2word)
# Show Topics
print(ldamallet.show_topics(formatted=False))

coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        print("start...")
        print("num of topic:" + str(num_topics))
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        print("end...")

    return model_list, coherence_values

# Can take a long time to run.

lmt = 50
strt = 20
stp = 10

# print("################################")
# print("how many topic:")
# print(datetime.datetime.now())
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=strt, limit=lmt, step=stp)
# # print("This is end...")
# # print(datetime.datetime.now())

# # Show graph
# limit=lmt; start=strt; step=stp
# x = range(start, limit, step)
# plt.figure()
# plt.plot(x, coherence_values)
# plt.xlabel("Num Topics")
# plt.ylabel("Coherence score")
# plt.legend(("coherence_values"), loc='best')
# plt.savefig(execution_directory + '/result_data/optimal_number_of_topic.png')

# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print(">>>>> Num Topics =", m, " has Coherence Value of", round(cv, 4))

optimal_model = ldamallet
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))

def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
print(df_dominant_topic.head(25))
df_dominant_topic.to_csv(execution_directory + '/result_data/df_dominant_topic.csv')


sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                            grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

# Show
print(sent_topics_sorteddf_mallet.head())
sent_topics_sorteddf_mallet.to_csv(execution_directory + '/result_data/sent_topics_sorteddf_mallet.csv')


# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)

# Topic Number and Keywords
topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Change Column names
df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

# Show
print(df_dominant_topics)
df_dominant_topics.to_csv(execution_directory + '/result_data/df_dominant_topics.csv')

print(df.shape[0])
print(df_dominant_topics.shape[0])


# Get Result csv files
result_csv_path = execution_directory + '/result_data/df_dominant_topic.csv'
topic_csv_path = execution_directory + '/result_data/sent_topics_sorteddf_mallet.csv'

print(result_csv_path)
print(topic_csv_path)


df_result = pd.read_csv(result_csv_path, error_bad_lines=False)
df_topic = pd.read_csv(topic_csv_path, error_bad_lines=False)

# set the journal names to dataframe
df_result["journal_name"] = df['journal_name']

# Get req values for result
journal_list = df_result.journal_name.unique()
topic_list = df_topic.Topic_Num.unique()

# Empty dictionary for result
result_dic = {journal : {topic : 0 for topic in topic_list} for journal in journal_list}

# Calculate article topic numbers for journal
for journal in journal_list: 
    df_loop = df_result[df_result["journal_name"] == journal]
    
    for index, row in df_loop.iterrows():
        result_dic[journal][row['Dominant_Topic']] = result_dic[journal][row['Dominant_Topic']] + 1

# Sort topic DESC for journals
sorted_result_dic = {}
for key, value in result_dic.items():
    sorted_result_dic[key] = dict( sorted(value.items(), key=operator.itemgetter(1), reverse=True))

# Create result dictionary
dic_for_save = {
    "journal_name" : [],
    "total_article" : [],
    "topic_1" : [],
    "topic_1_how_many_times" : [],
    "topic_1_keywords" : [],
    "topic_2" : [],
    "topic_2_how_many_times" : [],
    "topic_2_keywords" : [],
    "topic_3" : [],
    "topic_3_how_many_times" : [],
    "topic_3_keywords" : []
}

for key, value in sorted_result_dic.items():
    dic_for_save["journal_name"].append(key)
    
    dic_for_save["total_article"].append(len(df[df["journal_name"] == key]))
    
    index = 0
    for inner_key, inner_value in value.items():
        if index == 0:
            dic_for_save["topic_1"].append(inner_key)
            dic_for_save["topic_1_how_many_times"].append(inner_value)
            dic_for_save["topic_1_keywords"].append(df_topic.iloc[int(inner_key)]["Keywords"])
        elif index == 1:
            if inner_value > 0:
                dic_for_save["topic_2"].append(inner_key)
                dic_for_save["topic_2_how_many_times"].append(inner_value)
                dic_for_save["topic_2_keywords"].append(df_topic.iloc[int(inner_key)]["Keywords"])
            else:
                dic_for_save["topic_2"].append(None)
                dic_for_save["topic_2_how_many_times"].append(None)
                dic_for_save["topic_2_keywords"].append(None)
        elif index == 2:
            if inner_value > 0:
                dic_for_save["topic_3"].append(inner_key)
                dic_for_save["topic_3_how_many_times"].append(inner_value)
                dic_for_save["topic_3_keywords"].append(df_topic.iloc[int(inner_key)]["Keywords"])
            else:
                dic_for_save["topic_3"].append(None)
                dic_for_save["topic_3_how_many_times"].append(None)
                dic_for_save["topic_3_keywords"].append(None)
        elif index == 3:
            break
        index = index + 1

# create result data frame and save
result_data_frame = pd.DataFrame(dic_for_save, columns = ["journal_name", "total_article", "topic_1", "topic_1_how_many_times", "topic_1_keywords", "topic_2", "topic_2_how_many_times", "topic_2_keywords", "topic_3", "topic_3_how_many_times", "topic_3_keywords"])
result_data_frame.to_csv(execution_directory + '/result_data/result_top_3.csv')

result_frame = pd.DataFrame(result_dic)
result_frame = result_frame.T
result_frame.to_csv(execution_directory + '/result_data/result.csv')

for index, row in sent_topics_sorteddf_mallet.iterrows():
    folder_path = '/result_images/'
    
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(row["Keywords"])
    # Save word cloud
    filename = str(index) + '_topic-cloud.png'
    wordcloud.to_file(execution_directory + folder_path + filename)
