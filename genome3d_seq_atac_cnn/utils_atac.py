import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from matplotlib import colors
from scipy.stats.mstats import mquantiles
import scipy.stats as ss
import math
import scipy.linalg
import itertools
import copy
import random
import gzip

# builds adjacency matrix 
def BuildMatrix(PromoterFile, InteractionsFile, datatype):

    REFrag_dict={}
    index=0
    # Assign indices to all promoter HindIII sites.
    for line in open(PromoterFile,'r'):
        words=line.rstrip().split()
        key=(words[0], words[1], words[2])
        if words[0] in ['chr1']: # only chr1
           REFrag_dict[key]=index
           index+=1

    # Initialize matrix (promoter x promoter)
    PPMatrix=np.zeros((len(REFrag_dict), len(REFrag_dict))) #  number of promoters in chr 1

    # Fill (promoter x promoter) matrix with q-values of promoter-promoter interaction
    max_score=0
    for line in gzip.open(InteractionsFile,'r'):
        words=line.rstrip().split()
        if float(words[6])!=0:  
            if (-1)*math.log10(float(words[6]))>max_score:
                max_score=(-1)*math.log10(float(words[6])) 
            
    for line in gzip.open(InteractionsFile,'r'):
        words=line.rstrip().split()
        if words[0] in ['chr1']: #only chr1
            i=REFrag_dict[(words[0], words[1], words[2])]
            j=REFrag_dict[(words[3], words[4], words[5])]
            if datatype == 'HiC':
                if float(words[6])!=0:                        
                    q_values=(-1)*math.log10(float(words[6])) # for HiC
                else:
                     q_values=max_score
            else:
                q_values=float(words[6])  # for CaptureC
            if PPMatrix[i,j] != 0:
                PPMatrix[i,j]=PPMatrix[i,j]/2+q_values/2
                PPMatrix[j,i]=PPMatrix[j,i]/2+q_values/2
            else:
                PPMatrix[i,j]=q_values
                PPMatrix[j,i]=q_values
                    # take -1*log(Q) for non-zero entries
    #mask = PPMatrix != 0
    #PPMatrix[mask] = np.log10(PPMatrix[mask])*(-1)

    # list of non-zero q-values
    q_values=list(filter((0.0).__ne__,list(itertools.chain.from_iterable(np.array(PPMatrix).tolist()))))

    # Some tests:
    print "Some tests on adjacency matrix:"
    # 1. Check if the matrix is symmetric:
    if (PPMatrix.transpose() == PPMatrix).all() == True:
        print "Adjacency matrix is symmetric"
    # 2. Print out average q-values:
    print "Average q-value with zeros: ", str(np.average(PPMatrix))
    print "Average q-value w/o zeros: ", np.mean(q_values)

    # Print distribution of q-values
    plt.hist(q_values)
    plt.show()

    return PPMatrix

def printMatrix(Matrix, ylabel, QuantileValue, LowerUpperLimit, title=''):
    #vmaxLim=mquantiles(Matrix,[0.99])[0]
    Lim=mquantiles(Matrix,[QuantileValue])[0]
    print Matrix.max()
    print np.shape(Matrix)
    print "Limit:", Lim
    fig, ax = plt.subplots()
    fig.subplots_adjust(top=0.8)
    if LowerUpperLimit == 'lower':
        m = ax.matshow(Matrix, origin="bottom", #norm=colors.LogNorm(),  #norm=colors.SymLogNorm(1),
               cmap="RdYlBu_r", vmin=Lim)
    else:
        m = ax.matshow(Matrix, origin="bottom", #norm=colors.LogNorm(),  #norm=colors.SymLogNorm(1),
               cmap="RdYlBu_r", vmax=Lim) # cmap="RdYlBu_r"


    ax.axhline(-0.5, color="#000000", linewidth=1, linestyle="--")
    ax.axvline(-0.5, color="#000000", linewidth=1, linestyle="--")

    cb = fig.colorbar(m)
    cb.set_label(ylabel)

    ax.set_ylim((-0.5, len(Matrix) - 0.5))
    ax.set_xlim((-0.5, len(Matrix) - 0.5))

    plt.title(title)
    plt.show()
    return

def binarize(matrix):
    copy_matrix=copy.copy(matrix)
    copy_matrix[copy_matrix > 0] = 1
    return copy_matrix

def get_matrix(nodes):
    size=nodes.shape[0]
    matrix=np.zeros((size,size))
    for i in range(size):
        for j in range(i):
            matrix[i,j]=nodes[i]*nodes[j]
    #return (matrix+matrix.T)/2
    return matrix

def get_nodeDegree_ForEachNode(matrix):
    size=matrix.shape[0]
    nodeDegree=np.zeros(size)
    for i in range(size):
        nodeDegree[i] = sum(matrix[:,i])
    return nodeDegree

def get_NodeDegree(nodes, indxs):
    true_indxs = np.where(indxs==True)[0]
    nodeDegree = nodes[true_indxs] 
    return nodeDegree

def get_confidentLinks(matrix, thres):
    matrix_copy = copy.copy(matrix)
    matrix_copy[matrix_copy < thres] = 0
    return matrix_copy
