The code is programed based on Python 3.6.4 |Anaconda, Inc.

Required library:
========================================================================================
import numpy as np
import csv
import sys
import pandas as pd
import re
import nltk

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

import matplotlib.pyplot as plt

All this package and library can be directly install from Anaconda or using pip command.