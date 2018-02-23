"""Author: Manuel Reinbold, Maximilian Renk
Date: 21/11/17
Version: 1.0
"""

import glob
from xml.dom import minidom

# Delivers the unpreprocessed raw data for all social media comments.
def get_raw_data():
    data_path_positive = '../data/movieReviews/negativeFiles.txt'
    data_path_negative = '../data/movieReviews/positiveFiles.txt'

    positive_texts = get_data_from_txt(data_path_positive)
    negative_texts = get_data_from_txt(data_path_negative)
    sorted_texts = positive_texts + negative_texts

    return sorted_texts, positive_texts, negative_texts

# Reads data from a xml file and return text and sentiment of social media comment.
def get_data_from_txt(filepath):
    texts = []
    with open(filepath, 'r', encoding='UTF-8') as file:
        for line in file:
            texts.append(line.strip('\n'))
    return texts

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