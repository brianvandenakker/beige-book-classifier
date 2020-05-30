# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 23:55:27 2019

@author: RV
"""

def fn_InfoFromTree(treeIn=None, testData=None):
    """
    Credits/authors: 
        https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
        
    Purpose: 
        explain / provide insights into the structure of a decision tree
        
    Inputs: 
        treeIn = sklearn.tree.DecisionTreeClassifier fitted (trained) tree
        testData = pandas dataframe of the same structure as the one used to trained the tree. Only the first row will be "mapped" through the tree
        
    Note:
        The original link has an expansion (commented below) for identifying the closest node shared by two trained samples. It would be nice to expand that code to cover any two test samples...
    """
    # Using those arrays, we can parse the tree structure:
    
    n_nodes = treeIn.tree_.node_count
    children_left = treeIn.tree_.children_left
    children_right = treeIn.tree_.children_right
    feature = treeIn.tree_.feature
    threshold = treeIn.tree_.threshold
    
    
    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1
    
        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
    
    print("The binary tree structure has %s nodes and has "
          "the following tree structure:"
          % n_nodes)
    for i in range(n_nodes):
        if is_leaves[i]:
            print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
        else:
            print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                  "node %s."
                  % (node_depth[i] * "\t",
                     i,
                     children_left[i],
                     feature[i],
                     threshold[i],
                     children_right[i],
                     ))
    print()
    
    
    if testData is not None:
    
        # First let's retrieve the decision path of each sample. The decision_path
        # method allows to retrieve the node indicator functions. A non zero element of
        # indicator matrix at the position (i, j) indicates that the sample i goes
        # through the node j.
        
        node_indicator = treeIn.decision_path(testData)
        
        # Similarly, we can also have the leaves ids reached by each sample.
        
        leave_id = treeIn.apply(testData)
        
        # Now, it's possible to get the tests that were used to predict a sample or
        # a group of samples. First, let's make it for the sample.
    
        sample_id = 0 #@@RV deliberately left it at 0 - assume function will be used with one test row at a time
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                            node_indicator.indptr[sample_id + 1]]
        print('Rules used to predict sample %s: ' % sample_id)
        for node_id in node_index:
            if leave_id[sample_id] == node_id:
                continue
            if (testData.iloc[sample_id, feature[node_id]] <= threshold[node_id]):
                threshold_sign = "<="
            else:
                threshold_sign = ">"
        
            print("decision id node %s : (testData[%s, %s] (= %s) %s %s)"
                  % (node_id,
                     sample_id,
                     feature[node_id],
                     testData.iloc[sample_id, feature[node_id]],
                     threshold_sign,
                     threshold[node_id]))
        
        # For a group of samples, we have the following common node.
#        sample_ids = [0, 1]
#        common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) ==
#                        len(sample_ids))
#        
#        common_node_id = np.arange(n_nodes)[common_nodes]
#        
#        print("\nThe following samples %s share the node %s in the tree"
#              % (sample_ids, common_node_id))
#        print("It is %s %% of all nodes." % (100 * len(common_node_id) / n_nodes,))
