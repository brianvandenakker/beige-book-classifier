# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 11:09:50 2019

@author: RV
"""

#%% write a simple function to multiply two numbers

def fn_test_product(a,b, debug_TF = False): 
    # def means "start a function", fn_test_product is its name, a and b are the inputs
    # debug_TF is a boolean (True/False) with a default value of False
# input: two numbers, a and b
# output: the product of the number
    print('inputs are a=' + str(a) + ' and b=' + str(b)) # + between strings appends the strings, print does print...
    if debug_TF:
        print('and I really want to see something else like the time...')
        print('and a movie...')
    c = a * b
    return(c)
    
#%% test the function
#fn_test_product(5,10)
#fn_test_product(2,3)
#fn_test_product(10,100, debug_TF = True)
    
