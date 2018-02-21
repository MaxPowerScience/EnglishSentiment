"""Author: Manuel Reinbold, Maximilian Renk
Date: 21/11/17
Version: 1.0
"""

import glob
from xml.dom import minidom

# Delivers the unpreprocessed raw data for all social media comments.
def get_raw_data():
    # parse an xml file by name
    data_path_positive = '../data/movieReviews/negativeFiles.txt'
    data_path_negative = '../data/movieReviews/positiveFiles.txt'

    positive_texts = []
    with open(data_path_positive, 'r', encoding='UTF-8') as file:
        for line in file:
            positive_texts.append(line.strip('\n'))

    negative_texts = []
    with open(data_path_negative, 'r', encoding='UTF-8') as file:
        for line in file:
            negative_texts.append(line.strip('\n'))

    sorted_texts = positive_texts + negative_texts

    # Sort texts according to sentiment
    return sorted_texts, positive_texts, negative_texts

# Reads data from a xml file and return text and sentiment of social media comment.
def get_data_from_xml(filepath):
    my_doc = minidom.parse(filepath)
    document_items = my_doc.getElementsByTagName('Document')
    texts, sentiments = [], []
    for document in document_items:
        for child in document.childNodes:
            if child.nodeName == 'sentiment':
                sentiments.append(child.firstChild.data)
            if child.nodeName == 'text':
                texts.append(child.firstChild.data)

    return texts, sentiments

# Sorting the texts according to their sentiments (order: positive, negative, neutral).
def sort_text_by_sentiment(texts, sentiments):
    sorted_positive_texts, sorted_negative_texts, sorted_neutral_texts = [], [], []
    sorted_positive_sentiment, sorted_negative_sentiment, sorted_neutral_sentiment = [], [], []

    for idx, text in enumerate(texts):
        if sentiments[idx] == 'positive':
            sorted_positive_texts.append(text)
            sorted_positive_sentiment.append(sentiments[idx])
        elif sentiments[idx] == 'negative':
            sorted_negative_texts.append(text)
            sorted_negative_sentiment.append(sentiments[idx])
        else:
            sorted_neutral_texts.append(text)
            sorted_neutral_sentiment.append(sentiments[idx])

    sorted_texts = sorted_positive_texts + sorted_negative_texts + sorted_neutral_texts
    sorted_sentiments = sorted_positive_sentiment + sorted_negative_sentiment + sorted_neutral_sentiment

    return sorted_texts, sorted_positive_texts, sorted_negative_sentiment, sorted_neutral_texts, sorted_sentiments

def convert_to_one_file():
    pos_path = '../data/movieReviews/positiveReviews/*.txt'
    neg_path = '../data/movieReviews/negativeReviews/*.txt'
    pos_files = glob.glob(pos_path)
    neg_files = glob.glob(neg_path)

    for name in pos_files:
        with open('../data/movieReviews/positiveFiles.txt', 'a', encoding='utf-8') as writer:
            with open(name, 'r', encoding='utf-8') as reader:
                writer.write(reader.readline()+'\n')

    for name in neg_files:
        with open('../data/movieReviews/negativeFiles.txt', 'a', encoding='utf-8') as writer:
            with open(name, 'r', encoding='utf-8') as reader:
                writer.write(reader.readline()+'\n')

def main():
    print("Please execute some code")

if __name__ == "__main__":
    main()