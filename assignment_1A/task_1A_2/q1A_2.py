""" This file is created as a suggested solution template for question 1.2 in DD2434 - Assignment 1A.

    We encourage you to keep the function templates.
    However, this is not a "must" and you can code however you like.
    You can write helper functions etc. however you want.

    If you want, you can use the class structures provided to you (Node and Tree classes in Tree.py
    file), and modify them as needed. In addition to the data files given to you, it is very important for you to
    test your algorithm with your own simulated data for various cases and analyse the results.

    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format. Let us know if you face any problems.

    Also, we are aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). We wanted to keep the template codes as simple as possible.
    You can change the file names however you want.

    For this task, we gave you three different trees (q1A_2_small_tree, q1A_2_medium_tree, q1A_2_large_tree).
    Each tree has 5 samples (the inner nodes' values are masked with np.nan).
    We want you to calculate the likelihoods of each given sample and report it.

    Note:   The alphabet "K" is K={0,1,2,3,4}.

    Note:   A VERY COMMON MISTAKE is to use incorrect order of nodes' values in CPDs.
            theta is a list of lists, whose shape is approximately (num_nodes, K, K).
            For instance, if node "v" has a parent "u", then p(v=Zv | u=Zu) = theta[v][Zu][Zv].

            If you ever doubt your useage of theta, you can double-check this marginalization:
            \sum_{k=1}^K p(v = k | u=Zu) = 1
"""

import numpy as np
from Tree import Tree
from Tree import Node


def calculate_likelihood(tree_topology, theta, beta):
    """
    This function calculates the likelihood of a sample of leaves.
    :param: tree_topology: A tree topology. Type: numpy array. Dimensions: (num_nodes, )
    :param: theta: CPD of the tree. Type: list of numpy arrays. Dimensions (approximately): (num_nodes, K, K)
    :param: beta: A list of node assignments. Type: numpy array. Dimensions: (num_nodes, )
                Note: Inner nodes are assigned to np.nan. The leaves have values in [K]
    :return: likelihood: The likelihood of beta. Type: float.

    This is a suggested template. You don't have to use it.
    """

    # TODO Add your code here
    N = len(beta)   # Number of nodes
    K = 5           # Number of classes in each vertex



    s_arr = np.zeros((N,K))
    leaf_vec_idx = [i for i,x in enumerate(beta) if np.isnan(x) == False]   # Takes out the index of every leaf node

    # Initiating the s values for the leaf nodes for which they are 1
    for i in leaf_vec_idx:
        s_arr[i][int(beta[i])] = 1


    # for i in leaf_vec_idx:
    #     s_arr[i] = np.ones(K)

    
    # Want to see if a leaf have the same parents, in that case we can calculate the s(parent,*)
    J = len(leaf_vec_idx)-1      # last index of vector
    iter = 1
    start_idx = leaf_vec_idx[J]
    next_start_idx = leaf_vec_idx[J-1]
    current_parent = int(tree_topology[start_idx])

    while(J != -1):                 # Stops when vector is empty
        
        
        child_idx = np.where(tree_topology[leaf_vec_idx] == current_parent)[0] # Select index 0 which is array of index for children, otherwise returns tuple

        if len(child_idx) < 2:
            iter += 1
            current_parent = tree_topology[leaf_vec_idx].astype(int)[-iter]  
        
        elif len(child_idx) == 0: 
            print("You are now at the root") 
            leaf_vec_idx = []
        elif len(child_idx) > 2: raise ValueError('The number of children for a vertex cannot be more than 2')
        
        else: 
            first_child = leaf_vec_idx[child_idx[1]]
            second_child = leaf_vec_idx[child_idx[0]]
            s_temp = [ np.dot(theta[first_child][l],s_arr[first_child]) * np.dot(theta[second_child][l],s_arr[second_child])  for l in range(K)]
            s_arr[current_parent] = s_temp
            
            # Remove the two leafs that was just added, start with largest index to not make the list to short
            if child_idx[1] > child_idx[0]: 
                leaf_vec_idx.pop(child_idx[1])
                leaf_vec_idx.pop(child_idx[0])
            else: 
                leaf_vec_idx.pop(child_idx[0])
                leaf_vec_idx.pop(child_idx[1])
            

            # Add the parent which is now a leaf
            leaf_vec_idx.append(current_parent)
            # first_child = current_parent
            if current_parent == 0: leaf_vec_idx = []
        
            else: 
                current_parent = tree_topology[leaf_vec_idx].astype(int)[-1]    # Selects last element in index vector
                
                iter = 1
            
        J = len(leaf_vec_idx) -1
        
    # Start: Example Code Segment. Delete this segment completely before you implement the algorithm.
    print("\tCalculating the likelihood...")
    likelihood = np.dot(s_arr[0],theta[0])
    # End: Example Code Segment

    return likelihood

def calculate_likelihod_recursive(tree_topology, theta, beta):
    N = len(beta)   # Number of nodes
    K = 5           # Number of classes in each vertex





    # Start: Example Code Segment. Delete this segment completely before you implement the algorithm.
    print("\tCalculating the likelihood...")
    likelihood = np.random.rand()
    # End: Example Code Segment

    return likelihood


def main():
    print("Hello World!")
    print("This file is the solution template for question 1.2.")

    print("\n1. Load tree data from file and print it\n")

    #filename = "task_1A_2/data/q1A_2/q1A_2_small_tree.pkl"  # "data/q1A_2_medium_tree.pkl", "data/q1A_2_large_tree.pkl"
    filename = "task_1A_2/data/q1A_2/q2_2_medium_tree.pkl"
    #filename = "task_1A_2/data/q1A_2/q2_2_large_tree.pkl"
    print("filename: ", filename)

    t = Tree()
    t.load_tree(filename)
    t.print()
    print("K of the tree: ", t.k, "\talphabet: ", np.arange(t.k))

    print("\n2. Calculate likelihood of each FILTERED sample\n")
    # These filtered samples already available in the tree object.
    # Alternatively, if you want, you can load them from corresponding .txt or .npy files

    for sample_idx in range(t.num_samples):
        beta = t.filtered_samples[sample_idx]
        print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
        sample_likelihood = calculate_likelihood(t.get_topology_array(), t.get_theta_array(), beta)
        print("\tLikelihood: ", sample_likelihood)
        #print("Tree topology indentation",t.print_topology())

    # sample_idx = 0
    # beta = t.filtered_samples[sample_idx]
    # print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
    # sample_likelihood = calculate_likelihood(t.get_topology_array(), t.get_theta_array(), beta)
    # print("\tLikelihood: ", sample_likelihood)
    



if __name__ == "__main__":
    main()
