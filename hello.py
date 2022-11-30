from flask import Flask, render_template, request
import pandas as pd
import math
import pandas as pd
import numpy as np
#import re
import math
#from collections import Counter
import math
import nltk
import csv
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def home():
  #Simple_Recommendation_system
  ##################STOPWORDS##########
  #establishing punctuation
  punct =  set(string.punctuation)
  #establishing stopwprds
  my_stopwords = set(stopwords.words('english'))
  #Combining stopwords with puntuations
  my_stopwords |= punct

  ##################CSV FILES####################
  #CSV File
  #Importing excel
  filename = "SBA_Loan_V3.csv"

  #Setting up lists for csv
  Fields = []
  rows = []

  with open(filename, 'r') as csvfile:
      # creating a csv reader object
      csvreader = csv.reader(csvfile)
      
      # extracting field names through first row
      fields = next(csvreader)
  
      # extracting each data row one by one
      for row in csvreader:
          rows.append(row)

  ##############################################
  ################CSV FILE END#################

  #############################################
  ################KEY FUNCTIONS################
  ################Bigram Creator Function#####
  #Function to create bigrams
  def big_bigram(list):
      counter_a_a = 0
      counter_a_b = 1
      counter_b_a = 1
      counter_b_b = 2
      loop_counter = 0
      temp_list = []
      big_len = len(list)
      if big_len % 2 == 0:
          while big_len % 2 == 0 and loop_counter != big_len // 2:
              temp_list.append(list[counter_a_a] + " " + list[counter_a_b])
              counter_a_a += 2
              counter_a_b += 2
              loop_counter += 1
          return temp_list
      if big_len % 2 > 0:
          while big_len % 2 > 0 and loop_counter != big_len // 2:
              temp_list.append(list[counter_a_a] + " " + list[counter_a_b])
              temp_list.append(list[counter_b_a] + " " + list[counter_b_b])
              counter_a_a += 2
              counter_a_b += 2
              counter_b_a += 2
              counter_b_b += 2
              loop_counter += 1
          return temp_list
  ##################Jaccards Similarity Function
  def get_jaccard(list1, list2):
      s1 = set(list1)
      s2 = set(list2)
      return float(len(s1.intersection(s2)) / len(s1.union(s2)))
      
  ######################COSINE SIMILARITY######
  # Program to measure the similarity between 
  # two sentences using cosine similarity.

  def get_cosine(string_a,string_b): 
      # X = input("Enter first string: ").lower()
      # Y = input("Enter second string: ").lower()
      X = string_a
      Y = string_b
        
      # tokenization
      X_list = word_tokenize(X) 
      Y_list = word_tokenize(Y)
        
      # sw contains the list of stopwords
      sw = my_stopwords 
      l1 =[];l2 =[]
        
      # remove stop words from the string
      X_set = {w for w in X_list if not w in sw} 
      Y_set = {w for w in Y_list if not w in sw}
        
      # form a set containing keywords of both strings 
      rvector = X_set.union(Y_set) 
      for w in rvector:
          if w in X_set: l1.append(1) # create a vector
          else: l1.append(0)
          if w in Y_set: l2.append(1)
          else: l2.append(0)
      c = 0
        
      # cosine formula 
      for i in range(len(rvector)):
              c+= l1[i]*l2[i]
      cosine = c / float((sum(l1)*sum(l2))**0.5)
      return cosine
  ##############################################
  ###############KEY FUNCTION END###############

  #############INPUT PREP#####################
  test_input = request.form['k']

  ############Prep for jaccard###################

  #Jaccard Prep
  def jaccard_prep(x):
      temp_list = []
      my_tokens = word_tokenize(x)
      for word in my_tokens:
          if word not in my_stopwords:
              temp_list.append(word.lower())
      return temp_list

  my_jaccard = jaccard_prep(test_input)
  ##############Prep for Cosine#############
  #Cosine Prep
  def cosine_prep(x):
      blank_string = ""
      lower_temp = x.lower()
      tokens = word_tokenize(lower_temp)
      for x in tokens:
          if x not in my_stopwords:
              blank_string += " "
              blank_string += x
      return blank_string

  my_cosine = cosine_prep(test_input)
  ##################################################
  ##############INPUT PREP END#####################
  ##########DATABASE PREP#######################

  #establishing rows database copy called working rows
  working_rows = rows

  #lowering keywords and text fields
  for x in working_rows:
      x[2]=x[2].lower()
      x[4]=x[4].lower()

  #removing stop words from text
  #for str in working_rows:
  #    temp_str = str[5]
  #    temp_str = cosine_prep(temp_str)
  #    str[5] = temp_str


  #adding split keywords to index 6 Jaccard prep Unique
  for word in working_rows:
      word.append(word[2].split("|"))

  #adding split keywords to index 7 Jaccard prep Keywords
  for word in working_rows:
      word.append(word[3].split("|"))

  #bigrams of input Index 8
  for word in working_rows:
      word.append(big_bigram(my_jaccard))

  #Bigram Jaccard Score Index 9
  for word in working_rows:
      word.append(get_jaccard(word[8], word[7]))

  #Unique Jaccard Score Index 10
  for word in working_rows:
      word.append(get_jaccard(word[6], my_jaccard))
      
  #Key terms Jaccard Score Index 11
  for word in working_rows:
      word.append(get_jaccard(word[7], my_jaccard))

  #Cosine Similarity Score Index 12
  for word in working_rows:
      word.append(get_cosine(word[5], my_cosine))

  #Combined score Index 13
  for word in working_rows:
      word.append(word[9] + word[10] + word[11] + word[12])

  #Enumerate
  for num,word in enumerate(working_rows):
      word.append(num)
  #####################End of Database Prep################
  #######################Display of Info###############

  #Putting index and combined scores in dictionary
  my_dict = {}
  for x in working_rows:
      if x[0] not in my_dict:
          my_dict[x[14]] = x[13]

  #Sorting in high to low in a list
  final_dict = sorted(my_dict.items(), key=lambda x:          x[1],     reverse=True)


  #Getting indexes
  ranked_index = []
  for x in final_dict:
      ranked_index.append(x[0])

  #Getting full entries from working list
  full_list = []
  for x in ranked_index:
      full_list.append(working_rows[x])

  #Getting full clean list
  clean_full_list = []
  clean_temp = []
  for x in full_list:
      clean_temp.append(x[0])
      clean_temp.append(x[1])
      clean_temp.append(x[4])
      clean_temp.append(x[13])
      clean_full_list.append(clean_temp)
      clean_temp = []

  #Final Outcomes
  if len(clean_full_list) <= 4:
      big_final_list = clean_full_list

  if len(clean_full_list) > 4:
      big_final_list = clean_full_list[0:3]


            
  #transfer = big_final_list

    


  return render_template('after.html', data = big_final_list)
if __name__ == "__main__":
  app.run(debug=True)
