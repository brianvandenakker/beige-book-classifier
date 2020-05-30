# -*- coding: utf-8 -*-

#%%
from bs4 import BeautifulSoup 
import requests
import os
import re
import json
import pandas as pd
import numpy as np
import random
import sys
import scipy
import sklearn
import seaborn as sns
import sklearn.model_selection
import sklearn.preprocessing
from sklearn.metrics import confusion_matrix as skm_conf_mat
from sklearn.model_selection import train_test_split as skl_traintest_split
from sklearn import metrics
import datetime as DT
import time
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
from io import StringIO
from html.parser import HTMLParser
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
#%%

projFld = "C:/Users/brian/Documents/Github/beige-book-classifier/"
codeFld = os.path.join(projFld, "PyCode").replace('\\', '/')
fnsFld = os.path.join(codeFld,"_Functions").replace('\\', '/')
outputFld = os.path.join(projFld, "Output").replace('\\', '/')
rawDataFld = os.path.join(projFld, "RawData").replace('\\', '/')
savedDataFld = os.path.join(projFld, "SavedData").replace('\\', '/')

#%%

fnList = [
         "fn_logMyInfo"
        ,"fn_confusionMatrixInfo"
        ,"fn_MakeDummies"
        ,"fn_InfoFromTree"
        ,"fn_ROC"
        ,"fn_forewardStepWise"
        ] 
#for fn in fnList:
#    exec(open(os.path.join(fnsFld, fn + ".py")).read())