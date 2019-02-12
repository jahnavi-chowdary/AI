#!/usr/bin/env python
import argparse
from collections import Counter
import math
from optparse import OptionParser
import numpy as np
import sys
import csv
# Class for Naive Bayes for spam filter
stopwords = ["ourselves", "hers", "between", "yourself", "but", "again", 
             "there", "about", "once", "during", "out", "very", "having", 
             "with", "they", "own", "an", "be", "some", "for", "do", "its", 
             "yours", "such", "into", "of", "most", "itself", "other", "off", 
             "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", 
             "themselves", "until", "below", "are", "we", "these", "your", "his",
             "through", "don", "nor", "me", "were", "her", "more", "himself", "this",
             "down", "should", "our", "their", "while", "above", "both", "up", "to", 
             "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them",
             "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", 
             "then", "that", "because", "what", "over", "why", "so", "can", "did", 
             "not", "now", "under", "he", "you", "herself", "has", "just", "where", 
             "too", "only", "myself", "which", "those", "i", "after", "few", "whom", 
             "t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", 
             "how", "further", "was", "here", "than"]

class SpamFilter:
    def __init__(self, spam_train, spam_test):
        self.laplace_smooth = 0.1
        self.spam_train = spam_train
        self.spam_test = spam_test
        self.total_spam_words = 0
        self.total_ham_words = 0
        self.spam_prob = 1.0
        self.ham_prob = 1.0
        self.ham_map = Counter()
        self.spam_map = Counter()
        self.SpamHamFunc()

    def ProbabilityClass():
        global dict_word
        global totalMailCount
        wordCounts.setdefault("spam", {})
        wordCounts.setdefault("ham", {})

    # Training function for spam classifier
    def SpamHamFunc(self):
        spam_sum_total = 0
        ham_sum_total = 0
        # read training file
        train_data = open(self.spam_train, "r")
        count = 0
        spam = 0
        #for each training data
        for row in train_data:
            k = row.split(' ')
            ans = k[1]
            for words in range(2, len(k), 2):
                if words in stopwords:
                    continue
                if ans == "spam":
                    spam += 1
                    self.spam_map[k[words]] += int(k[words + 1])
                else:
                    self.ham_map[k[words]] += int(k[words + 1])
                count += 1     
        self.spam_prob = float(spam) / float(count)
        self.ham_prob = 1 - self.spam_prob
        spam_sum_total = sum(self.spam_map.itervalues())      
        self.total_spam_words += float(spam_sum_total)
        ham_sum_total = sum(self.ham_map.itervalues())
        self.total_ham_words += float(ham_sum_total)

    # To calculate probability of the word given it is spam
    def probWordSpamfunc(self, word, count, spam_check, counter):
        x = counter
        if spam_check == False:
            if word in self.ham_map:
                prob = float(self.ham_map[word]) / float(self.total_ham_words)
                effective_prob = np.log10(prob) * count
                return effective_prob
            else:
                length_ham = len(self.ham_map)
                smoothing = float(length_ham * self.laplace_smooth) + self.total_ham_words
                word_prob = np.log10(self.laplace_smooth / smoothing)
                return word_prob
        else:
            if word in self.spam_map:
                prob = float(self.spam_map[word]) / float(self.total_spam_words)
                effective_prob = np.log10(prob) * count
                return effective_prob
            else:
                length_spam = len(self.spam_map)
                smoothing = float(length_spam * self.laplace_smooth) + self.total_spam_words
                word_prob = np.log10(self.laplace_smooth / smoothing)
                return word_prob 
            

    def spamfilterFunc(self, rest_email, isham):
        isham = False
        check_spam = 0.0
        for i in range(0, len(rest_email), 2):
            given_words = rest_email[i]
            if given_words in stopwords:
                continue
            word_count = int(rest_email[i + 1])
            word_prob = self.probWordSpamfunc(given_words, word_count, True, i)
            if word_prob is not None:
                check_spam += word_prob

        check_ham = 0.0
        for i in range(0, len(rest_email), 2):
            given_words = rest_email[i]
            if given_words in stopwords:
                continue
            word_count = int(rest_email[i + 1])
            word_prob = self.probWordSpamfunc(given_words, word_count, False, i)
            if word_prob is not None:
                check_ham += word_prob
        check_spam = np.log10(self.spam_prob) + check_spam
        check_ham = np.log10(self.ham_prob) + check_ham
        if check_ham < check_spam:
            return True
        else:
            return False


if __name__ == "__main__":
    #defining variables
    ans = 0
    faulty = 0
    number_arg = 0

    #parsing arguements/parametrs given in command line
    arg_array = []
    argf = sys.argv
    for arguement in argf:
        len_arg =  len(arguement)
        if len_arg > 2 and arguement.startswith('-'):
            arguement = '-' + arguement
        arg_array.append(arguement)
    sys.argv = arg_array

    predicted_spam = 0
    predicted_ham = 0
    actual_predicted_spam = 0
    actual_predicted_ham = 0
    spam_count = 0
    ham_count = 0

    commandparser = OptionParser()
    commandparser.add_option('-t', '--f1', dest="training", help="train data csv")
    commandparser.add_option('-q', '--f2', dest="testing", help="test data csv")
    commandparser.add_option('-o', '--output', dest="output", help="output file")

    (options, args) = commandparser.parse_args()

    spam_train = options.training
    spam_test = options.testing
    output = options.output
    object_bayes = SpamFilter(spam_train, spam_test)

    # Openened the file for reading
    testdata = open(spam_test, 'r')
    #defining a list
    test_data_list = list()
    for row in testdata:
        row = row.split(' ')
        email_id = row[0]
        given = row[1]
        rest_mail = row[2:]
        #by default parameter
        isham = True
        spam_pred = object_bayes.spamfilterFunc(rest_mail, isham)
        pred_result = ""

        if spam_pred == True:
            predicted_spam += 1
            pred_result = "spam"
        else:
            predicted_ham += 1
            pred_result = "ham"

        if pred_result == given:
            if pred_result == "spam":
                actual_predicted_spam += 1
            else:
                actual_predicted_ham += 1
            ans += 1
        else:
            faulty += 1

        test_data_list.append((email_id, pred_result))
        #counting number of spams and hams
        if given == "ham":
            ham_count += 1
        else:
            spam_count += 1

    # Write the output to a csv
    with open(output + ".csv", 'wb') as output_csv:
        k = csv.writer(output_csv)
        for row in test_data_list:
            k.writerow(row)
    accuracy = (float(ans)/float(ans + faulty))
    precision_spam = (float(actual_predicted_spam) / float(predicted_spam))
    precision_ham = (float(actual_predicted_ham) / float(predicted_ham))
    recall = float(actual_predicted_spam) / float(spam_count)
    fmeasure = (2*precision_spam*recall)/(precision_spam+recall) * 100
    # Print the accuracy, f1 and recall
    print "Accuracy of the result is", round((accuracy * 100),2) , "%"
    print "Precision of spam is: ", round((precision_spam * 100) , 2), "%"
    print "Precision of not spam is: ", round((precision_ham * 100) , 2), "%"
    print "Recall parameter is: ", round((recall * 100), 2), "%"
    print "fmeasure is :", round(fmeasure, 2), "%"