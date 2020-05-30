# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 10:45:48 2019

@author: RV
"""

def fn_MakeDummies(dsin=None, varsToDummy=None, dsout=None):
    """
  dsin = dataset input
  varsToDummy = list of variable names which are to be transformed into dummies
  dsout = dataset output
"""
# testting - make sure it is commented out when done!!!
#dsin = Carseats.copy()
#varsToDummy = ['ShelveLoc']
#dsout = Carseats2
#
    dsout = dsin.copy()
    
    for var in [x for x in varsToDummy if x in dsin.columns]: 
    #    cat_list='var'+'_'+var
        cat_list = pd.get_dummies(dsout[var], prefix=var) # this creates dummy variables from a categorical/character variable
        dsout1 = dsout.join(cat_list)
        dsout  = dsout1
    
    dsout = dsout.drop(columns=varsToDummy)
    #data_vars=dsout.columns.values.tolist()
    #to_keep=[i for i in data_vars if i not in varsToDummy]
    #dsout = dsout[to_keep].copy()
    return(dsout)
