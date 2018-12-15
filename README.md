# Sentiment-Analysis-by-combining-Machine-Learning-and-Lexicon-Based-methods
This project is on twitter sentimental analysis by combining lexicon based and machine learning approaches. A supervised lexicon-based approach for extracting sentiments from tweets was implemented. Various supervised machine learning approaches were tested using scikit-learn libraries in python and implemented Decision Trees and Naive Bayes techniques.

The entire code for preprocessing, implementation and post-processing of the project was done in Python 2.7.

## Overview of the Project

      



## Requirements
The packages required for running the code are listed below.
* Sklearn
* Pandas
* Numpy 
* Math
* io
* os
* Nltk 


## Installations
Most of the packages can be installed using normal pip  commands. Installing NLTK may require special instructions which can be found at https://www.nltk.org/install.html

The preprocessing files which are required to run the code are as follows:

1. tweetylabel.csv            #contains the input tweets
2. dic.csv	                  #contains the dictionary created and merged
3. intense.csv.               #contains the intensifiers
4. bucket.csv.                #creates the bucket
5. positive-words.txt         #contains the positive word list as text file
6. negabuse.txt               #contains the negative and abusive word list as text file


## Instruction for running the code

Keep all the above mentioned preprocessing files in the same folder and change the directory to that folder. lexi_plus_ml.py file contains the entire code for the project. Open the code and specify the working directory on line 17 of the code.
