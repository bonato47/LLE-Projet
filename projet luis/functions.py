#Import Libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from IPython.display import Image
from IPython.core.display import HTML 
import os
from mpl_toolkits.mplot3d import Axes3D

import cv2
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from matplotlib import rc

from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import Isomap
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import classification_report

from scipy.stats import spearmanr
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist
from scipy.signal import argrelextrema
from scipy import ndimage, misc


import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

import pickle
from time import time


font = {'family' : 'times new roman',
        'weight' : 400,
        'size'   : 19}

plt.rc('font', **font)



def creat_s_curve(n_points,string):
    """
    Input : Number of points and dataset name
    Output: data point and colors
    """
    
    if string == "s_curve" : 
        print("S-curve with : ",n_points," Datapoint")
        X, color = datasets.make_s_curve(n_points, random_state=0)

        # Create figure
        fig = plt.figure(figsize=(8, 8))
        fig.suptitle("S_curve with %i points"
                     % (n_points))

        # Add 3d scatter plot
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
        ax.view_init(4, -72)
    if string == "swiss":
        print("Swiss Roll with : ",n_points," Datapoint")
        X, color = datasets.make_swiss_roll(n_samples=n_points)
        # Create figure
        fig = plt.figure(figsize=(8, 8))
        fig.suptitle("Swiss Roll with %i points" % (n_points))
        # Add 3d scatter plot
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
        ax.view_init(4, -72)
    return X,color

def residual_variance(X, Y_reduc):
    """
    Input : Data set and embedding
    Output: residual variance
    """
    #pairwise euclidean distance
    Dx = pdist(X, 'euclidean')
    Dy = pdist(Y_reduc, 'euclidean')

    #Pearson correlation
    pxy, _ = pearsonr(Dx, Dy)

    #residual variance
    res_var = 1 - pxy**2
    
    return res_var

def Spearsman_s_Rho (X,Y):
    """
    Input : Data set and embedding
    Output: Spearsman_s_Rho
    """
    
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    nbrs2 = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(Y)
    distances2, indices2 = nbrs2.kneighbors(Y)

    distances  = np.delete(distances, 0, 1)
    distances2 = np.delete(distances2, 0, 1)
    coef, p = spearmanr(distances, distances2,axis=None)
    return  np.mean(coef)

def Spearsman_s_Rho_graph (X, n_neighbors,d,string):
    """
    Input : Data set, Max_k, dimension and type of embedding
    Output: Spearsman_s_Rho and Residual variance graph
    """

    matrix       = np.zeros((10,n_neighbors-4))
    matrix_error = np.zeros((10,n_neighbors-4))
    matrix_residual       = np.zeros((10,n_neighbors-4))
    for j in range(10):
        liste1 = []
        liste2 = []
        residual= []
        for i in range (4,n_neighbors):
            clf = manifold.LocallyLinearEmbedding(i, n_components=d,method='standard')
            clf_M = manifold.LocallyLinearEmbedding(i, n_components=d,method='modified')
            if string =="LLE":
                embbeding  = clf.fit_transform(X)
                liste2.append(clf.reconstruction_error_)
            else:
                embbeding = clf_M.fit_transform(X) 
                liste2.append(clf_M.reconstruction_error_)
            residual.append(residual_variance(X, embbeding))
            value = Spearsman_s_Rho (X,embbeding)
            liste1.append(value)
            if max(liste1)<=value : 
                K_max = i
                embbeding_best = embbeding
            if min(liste1)>=value :
                K_min = i
                embbeding_worst = embbeding
                

        matrix[j,:] = liste1
        matrix_residual[j,:] = residual
        matrix_error[j,:]=liste2
    print("Embedding with K_max = ",K_max)
    print("Embedding with K_min = ",K_min)
    mean_list                = matrix.mean(axis=0) 
    mean_list_error          = matrix_error.mean(axis=0) 
    mean_list_residual       = matrix_residual.mean(axis=0) 
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(4,n_neighbors), mean_list, color='darkgreen', linestyle='dashed', marker='o',
             markerfacecolor='black', markersize=6,label="Spearsman")
    
    plt.plot(range(4,n_neighbors), mean_list_residual, color='darksalmon', linestyle='dashed', marker='o',
             markerfacecolor='black', markersize=6,label="Residual Variance")

    plt.title("Spearsman's Rho and Residual Variance for K between [4;%i] for %s "% (n_neighbors,string))
    plt.xlabel('K Value')
    plt.ylabel("Spearsman's Rho and Residual Variance")
    plt.legend()
    
    print("K =",np.argmax(mean_list)+4,"Produce the biggest Spearson's Rho: ", np.amax(mean_list),"for", string)
    print("K =",np.argmin(mean_list)+4,"Produce the smallest Spearson's Rho: ", np.amin(mean_list),"for", string)
    
    return embbeding_best,embbeding_worst,mean_list_error,K_max,K_min


def creat_graph(X1,LLE,MLLE,color,kLLE,kMLLE,string):
    
    """
    Input : Embeddings
    Output: Graph Embeddings
    """
    fig = plt.figure(figsize=(10,5))
    ax0 = fig.add_subplot(131, projection='3d')
    ax0.scatter(X1[:, 0], X1[:, 1], X1[:, 2],c=color ,cmap=plt.cm.Spectral)
    ax0.view_init(4, -72)
    ax0.axis('off')
    ax1 = fig.add_subplot(132)
    ax1.scatter(LLE[:, 0], LLE[:, 1], c=color, cmap=plt.cm.Spectral)
    ax1.axis('off')
    ax2 = fig.add_subplot(133)
    ax2.scatter(MLLE[:, 0], MLLE[:, 1], c=color, cmap=plt.cm.Spectral)
    ax2.axis('off')
    
    #ax0.set_title("S-curve")
    ax0.set_title("%s results" %(string))
    ax1.set_title("LLE, k= %i" %(kLLE))
    ax2.set_title("MLLE, k= %i" %(kMLLE))

def creat_graph2(X1,LLE,MLLE,color,kLLE,kMLLE,string):
   
    """
    Input : Embeddings
    Output: Graph Embeddings
    """
    fig = plt.figure(figsize=(10,5))
    ax0 = fig.add_subplot(141, projection='3d')
    ax0.scatter(X1[:, 0], X1[:, 1], X1[:, 2],c=color ,cmap=plt.cm.Spectral)
    ax0.view_init(4, -72)
    ax0.axis('off')
    ax1 = fig.add_subplot(142)
    ax1.scatter(LLE[:, 0], LLE[:, 1], c=color, cmap=plt.cm.Spectral)
    ax1.axis('off')
    ax2 = fig.add_subplot(143)
    ax2.scatter(MLLE[:, 0], MLLE[:, 1], c=color, cmap=plt.cm.Spectral)
    ax2.axis('off')
    
    PCA_r= PCA(n_components=3).fit_transform(X1)
    ax3 = fig.add_subplot(144)
    ax3.scatter(PCA_r[:, 1], PCA_r[:, 2], c=color, cmap=plt.cm.Spectral)
    ax3.axis('off')
    
    
    #ax0.set_title("S-curve")
    ax0.set_title("%s results" %(string))
    ax1.set_title("LLE, k= %i" %(kLLE))
    ax2.set_title("MLLE, k= %i" %(kMLLE))
    ax2.set_title("MLLE, k= %i" %(kMLLE)) 
    ax3.set_title("PCA")


def LLE_MLLE(X,n_neighbors,d):
    
    """
    Input : Dataset, Hyperparameter k and d
    Output: Embeddings and error
    """
    
    clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=d,method='standard')
    clf_M = manifold.LocallyLinearEmbedding(n_neighbors, n_components=d,method='modified')
    LLE  = clf.fit_transform(X)
    MLLE = clf_M.fit_transform(X)
    LLE_error = clf.reconstruction_error_
    MLLE_error = clf_M.reconstruction_error_
    return LLE,MLLE,LLE_error,MLLE_error

def Spearsman_s_Rho_graph2 (X, n_neighbors,d,string): # MEAN and Error and merge graph!!!
    
    """
    Input : Dataset, Hyperparameter k and d, type of embedding (LLE or MLLE)
    Output: Spearsman_s_Rho_graph, best and worst embedding
    """
    
    matrix       = np.zeros((10,-4+n_neighbors))
    matrix_error = np.zeros((10,-4+n_neighbors))
    for j in range(10):
        liste1 = []
        liste2 = []
        for i in range (4,n_neighbors):
            clf = manifold.LocallyLinearEmbedding(i, n_components=d,method='standard')
            clf_M = manifold.LocallyLinearEmbedding(i, n_components=d,method='modified')
            if string =="LLE":
                embbeding  = clf.fit_transform(X)
                liste2.append(clf.reconstruction_error_)
            else:
                embbeding = clf_M.fit_transform(X) 
                liste2.append(clf_M.reconstruction_error_)
            value = Spearsman_s_Rho (X,embbeding)
            liste1.append(value)
            if max(liste1)<=value : 
                K_max = i
                embbeding_best = embbeding
            if min(liste1)>=value :
                K_min = i
                embbeding_worst = embbeding
                

        matrix[j,:] = liste1
        matrix_error[j,:]=liste2
    print("Embedding with K_max = ",K_max)
    print("Embedding with K_min = ",K_min)
    mean_list       = matrix.mean(axis=0) 
    mean_list_error = matrix_error.mean(axis=0) 
    print("K =",np.argmax(mean_list)+4,"Produce the biggest Spearson's Rho: ", np.amax(mean_list),"for", string)
    print("K =",np.argmin(mean_list)+4,"Produce the smallest Spearson's Rho: ", np.amin(mean_list),"for", string)
    return embbeding_best,embbeding_worst,mean_list_error,mean_list


def multiclass(classes, df):
    """
    Returns a dataset with chosen classes from input list 
    """
    df_classes = df[df.label.isin(classes)]
    df_classes.reset_index(inplace=True, drop=True)
    return df_classes


def im_rotation(df, n):
    """
    Rotation function
    in : df = dataframe of shape (nproducts x 785) 
         n  = number of rotations to apply
    out: df_labels = pandas series with all n*nproducts labels, 
         df_images = pandas dataframe (n*nproducts x 784)
    """
    #separation images and labels
    df_labels = df.iloc[0:, 0].astype(int)
    df_images = df.iloc[0:, 1:]
    
    #conversion to array
    images = df_images.to_numpy()

    # making a list of image arrays of 28x28
    list_images = [[]]*len(images)
    for i in range(len(images)):
        list_images[i]=images[i].reshape(28,28)

    #list of images multiplied by number of rotations
    rotated = [[]]*len(images)*n

    #rotate each image n times
    for img in range(len(list_images)):
        for angle in range(n):
            rotated[img+(len(list_images)*angle)] = ndimage.rotate(list_images[img], int((360/n)*angle),       reshape=False)

    #convert to a stacked array
    stacked = np.vstack(rotated).reshape(len(rotated),28,28)
    stacked = stacked.reshape(len(rotated),784)
    
    #convert to df
    df_rot = pd.DataFrame(data=stacked)
    
    #update labels and images for training
    df_images = df_rot.copy()

    labels = df_labels.values
    for i in range(n-1):
        labels = np.concatenate((labels, df_labels.values), axis=0)

    df_labels =  pd.Series(labels)    
    
    return df_labels, df_images


def intensity(df_images, i):
    """
    Changes pixels intensity in input dataframe with images only and returns it
    """
    df_images = df_images*i
    return df_images

def noisy(image, noise_type = 'gauss'):
    #function for noise addition
    """
    types:
    gaussian -> gauss
    salt & pepper -> sp
    poisson -> poisson
    speckle -> spec
    """
    if noise_type == "gauss":
        row,col = image.shape
        mean = 0
        var = 65536
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        return noisy
    
    elif noise_type == "sp":
        row,col = image.shape
        s_vs_p = 0.5
        amount = 0.10
        out = np.copy(image)
        #salt
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1
        #pepper
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    
    elif noise_type =="spec":
        row,col = image.shape
        gauss = np.random.randn(row,col)
        gauss = gauss.reshape(row,col)        
        noisy = image + image * gauss
        return noisy
    
def plot_embedding_v(X, X_origin,y_label,size=10 ,title=None, dims=[None, 28, 28]):
    
    """
    Input : Data set and it embedding
    Output: Spearsman_s_Rho_graph, best and worst embedding
    """
    dims[0] = X.shape[0]
    X_origin = X_origin.values.astype(np.float).reshape(dims)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(size,size))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y_label.values[i]),
                 color=plt.cm.Set1(y_label.values.astype(np.int)[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 12})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 8e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(X_origin[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    ax.axis('off')    
    

def KNN(x_train, x_test, y_train, y_dev, k):
    """Take train and test set, run Knn and get prediction y_pred_knn"""
    #train
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    #predict
    y_pred_knn = knn.predict(x_test)
    
    return knn, y_pred_knn


def graph_F1(string,X_transformed_train,X_transformed_test,y_train,y_dev,X_transformed_train_m,X_transformed_test_m):   
    """
    Input : Embeddings
    Output: F1 measurement
    """
    
    knn_f1        = []
    knn_accuracy =  []
    
    knn_f1_LLE        = []
    knn_accuracy_LLE =  []

    knn_f1_MLLE        = []
    knn_accuracy_MLLE = []

    for k in range(1,30):
        
        # For LLE
        knn_emb, y_pred_emb_LLE = KNN(X_transformed_train, X_transformed_test, y_train, y_dev, k)
        knn_f1_LLE.append(metrics.f1_score( y_dev,  y_pred_emb_LLE, average= "weighted"))
        knn_accuracy_LLE .append(metrics.accuracy_score(y_dev,  y_pred_emb_LLE))
        
        # For MLLE
        knn_emb, y_pred_emb_MLLE = KNN(X_transformed_train_m, X_transformed_test_m, y_train, y_dev, k)
        knn_f1_MLLE.append(metrics.f1_score( y_dev, y_pred_emb_MLLE, average= "weighted"))
        knn_accuracy_MLLE .append(metrics.accuracy_score(y_dev, y_pred_emb_MLLE))
        
        if string == True : 
            # No embedding
            knn_emb, y_pred_emb = KNN(X_train_normalized, X_dev_normalized, y_train, y_dev, k)
            knn_f1.append(metrics.f1_score( y_dev,  y_pred_emb, average= "weighted"))
            knn_accuracy.append(metrics.accuracy_score(y_dev,  y_pred_emb))
        


    plt.figure(figsize=(12, 6))
    plt.plot(range(1,30), knn_f1_LLE, color='darkgreen', linestyle='dashed', marker='o',markerfacecolor='black', markersize=6,label="F1 LLE")

    plt.plot(range(1,30), knn_f1_MLLE, color='darksalmon', linestyle='dashed', marker='o',
                 markerfacecolor='black', markersize=6,label="F1 MLLE")
    if string == True : 
        plt.plot(range(1,30), knn_f1, color='royalblue', linestyle='dashed', marker='o',
                     markerfacecolor='black', markersize=6,label="F1 no embedding")

    plt.title("Knn classification F1 in function of K for LLE and MLLE ")
    plt.xlabel('K Value')
    plt.ylabel("F1 Score")
    plt.legend()
    
    best_k_LLE  = np.argmax(knn_f1_LLE)+1
    best_k_MLLE = np.argmax(knn_f1_MLLE)+1
    
    print("K =",np.argmax(knn_f1_LLE)+1,"give the best F1 score: ", np.amax(knn_f1_LLE),"for LLE")
    print("K =",np.argmax(knn_f1_MLLE)+1,"give the best F1 score: ", np.amax(knn_f1_MLLE),"for MLLE")
    return best_k_LLE,best_k_MLLE

def classification_rate (y_dev,y_pred,y_pred_emb):
    """
    Input : prediction knn with and whithout embeddings
    Output: classification_rate
    """
    knn_cm           = metrics.confusion_matrix(y_dev, y_pred)
    knn_cm_embedding = metrics.confusion_matrix(y_dev, y_pred_emb)
    rate             = (np.trace(knn_cm )-np.trace(knn_cm_embedding))/np.trace(knn_cm )
    return rate,knn_cm_embedding



def printmd(string):
    # Helper function
    # Print markdown style
    display(Markdown(string))
    

def plot_confusion_matrix(cm, names,string, cmap=plt.cm.Blues):
    """
    Plots confusion matrix for the KNN classification
    Input:  cm is the confusion matrix, names are the names of the classes and string is for the title
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix for %s'%(string))
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=90)
    plt.yticks(tick_marks, names)
    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    thresh = cm.max() / 2.

    for i in range (cm.shape[0]):
        for j in range (cm.shape[1]):
            plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
            
def Spearsman_s_Rho_graph3 (df_images, n_neighbors,d,string): 
    """
    Input : Image
    Output: graph_spearsman
    """

    X_train, X_dev, y_train, y_dev = train_test_split(df_images, df_labels, test_size=test_split)
    X_train_standardized = StandardScaler().fit_transform(X_train)
    X_dev_standardized = StandardScaler().fit_transform(X_dev)
    X_train_normalized = Normalizer().fit_transform(X_train_standardized)
    X_dev_normalized = Normalizer().fit_transform(X_dev_standardized)
    
    
    matrix       = np.zeros((10,-4+n_neighbors))
    matrix_error = np.zeros((10,-4+n_neighbors))
    for j in range(10):
        liste1 = []
        liste2 = []
        for i in range (4,n_neighbors):
            clf = manifold.LocallyLinearEmbedding(i, n_components=d,method='standard')
            clf_M = manifold.LocallyLinearEmbedding(i, n_components=d,method='modified')
            if string =="LLE":
                embbeding  = clf.fit_transform(X_dev_normalized)
                liste2.append(clf.reconstruction_error_)
            else:
                embbeding = clf_M.fit_transform(X_dev_normalized) 
                liste2.append(clf_M.reconstruction_error_)
            value = Spearsman_s_Rho (X_dev_normalized,embbeding)
            liste1.append(value)
            if max(liste1)<=value : 
                K_max = i
                embbeding_best = embbeding
            if min(liste1)>=value :
                K_min = i
                embbeding_worst = embbeding
                

        matrix[j,:] = liste1
        matrix_error[j,:]=liste2
    print("Embedding with K_max = ",K_max)
    print("Embedding with K_min = ",K_min)
    mean_list       = matrix.mean(axis=0) 
    mean_list_error = matrix_error.mean(axis=0) 
    print("K =",np.argmax(mean_list)+4,"Produce the biggest Spearson's Rho: ", np.amax(mean_list),"for", string)
    print("K =",np.argmin(mean_list)+4,"Produce the smallest Spearson's Rho: ", np.amin(mean_list),"for", string)
    return embbeding_best,embbeding_worst,mean_list_error,mean_list

def graph_spearsman_s (LLE_list,MLLE_list,n_neighbors):
    """
    Input : List LLE and MLLE 
    Output: graph_spearsman
    """
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(4,n_neighbors), LLE_list, color='darkgreen', linestyle='dashed', marker='o',
             markerfacecolor='black', markersize=6,label="LLE")
    plt.plot(range(4,n_neighbors), MLLE_list, color='darksalmon', linestyle='dashed', marker='o',
             markerfacecolor='black', markersize=6,label="MLLE")

    plt.title("Spearson's Rho for K between [4;%i] for LLE and MLLE "% (n_neighbors))
    plt.xlabel('K Value')
    plt.ylabel("Spearsman's Rho")
    plt.legend()