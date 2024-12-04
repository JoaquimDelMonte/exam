# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:53:16 2024

@author: nc182257
"""
#%% importer les bibliothèques
import numpy as np

from numpy.linalg import eig,svd

#%% Exercice 1
#######################
#%%créer le jeu de donnée
X=np.array([3,3,4,6])
Y=np.array([21,32,43,64])
A2=np.array([X,Y])
print(A2)
A2=A2.T
print(A2)
A=np.array([[3,21],[3,32],[4,43],[6,64]])
print(A)


#%%calcul de la moyenne
m=A.mean(axis=0) #axis=0 correspond aux colonnes
print("mean=",m)
print(X.mean())
print(Y.mean())

#%%calcul de l'écart type
s=A.std(axis=0)#c'est la formule de l'écart type avec la division par n
print('std=',s)

#%%standardiser A
A_std=(A-m)/s
A_std

#%%calcul de la matrice de variance-covariance
V=1/3 * A_std.T @ A_std
print(V)
#%%ou bien
V1=np.cov(A_std.T)
print(V1)

#%%calcul des valeurs propres et vecteurs propres
eig_val,eig_vect=eig(V)#bon
print(eig_val,eig_vect,sep='\n')

#%%# Using np.linalg.svd function

U, lambdas, _ = svd(V, full_matrices=True)
print(U)
print(lambdas)
#%%calcul des taux de varainces expliquées
taux=eig_val/sum(eig_val)*100
print(taux)

#test de l'ordre des valeurs propres et vecteurs propres V@U==lambda@U ?
np.allclose(V@eig_vect[0],eig_val[0]*eig_vect[0])#faux
np.allclose(V@eig_vect[0],eig_val[1]*eig_vect[0])#vrai
#=>les vecteurs ne sont pas ordonnés de la même façon que les valeurs propres

eigenvalues, eigenvectors = np.linalg.eig(V)

# Trier les valeurs propres par ordre décroissant
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_indices
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Afficher les valeurs propres triées et les vecteurs propres correspondants
print("Valeurs propres triées :", sorted_eigenvalues)
print("Vecteurs propres correspondants :\n", sorted_eigenvectors)
#ça ne marche pas!!
#car on ne connait pas quel vecteur propre correspond à quelle valeur propre


print(lambdas)
T1=U[0]/U.sum()
T2=U[1]/U.sum()
#np.allclose(V@U[0],lambdas[0]*U[0])
#là ils sont triés
#%% calcul du nuage de point X_pca projeté sur le plan des vecteurs propres 

X_pca=A_std@U
X_pca

#%% application avec sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#%%
scaler=StandardScaler()
A_std=scaler.fit_transform(A)
A_std#est obtenue avec la formul
pca = PCA(n_components=2)

X_pca = pca.fit_transform(A_std)
X_pca



X_pca_origine=pca.inverse_transform(X_pca)
pca.components_
pca.explained_variance_
pca.explained_variance_ratio_
pca.n_components
pca.mean_
#%%###################  Exercice 2   ##########################################
# 
###############################################################################
#%% créer le jeu de donnée
rng = np.random.RandomState(1)

X = (rng.rand(2, 2)@rng.randn(2, 200)).T
X
mean=X.mean(axis=0)
print("mean=",mean)
print('std=',X.std(axis=0))
#[0.823873  , 0.31358832]
#on remarque que les données sont légèrement dispersé
#=> ils n'ont pas besoin d'être standardisée
#%% tracer le nuage de point d'origine

plt.scatter(X[:, 0], X[:, 1])

plt.axis('equal')
plt.axhline(0, color='red',linewidth=0.5)  # Axe horizontal
plt.axvline(0, color='red',linewidth=0.5)  # Axe vertical
plt.show()
#%% standardiser les données
scaler=StandardScaler()
X_std=scaler.fit_transform(X)
X_std
plt.scatter(X_std[:, 0], X_std[:, 1], alpha=0.2,color='blue')
plt.show()
#%% appliquer la pca sur 2 composantes
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_std)
print(pca.components_)
print(pca.explained_variance_)
X_pca = pca.transform(X_std)
X_pca
#%% projeter les X_pca dans le repère d'origine
X_new = pca.inverse_transform(X_pca) 
X_new

#%%tracer le nouveau nuage de point avec les directions des composantes principale

plt.scatter(X_std[:, 0], X_std[:, 1], alpha=0.2,color='red')
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.3,color='blue')
plt.scatter(mean[0],mean[1],color='yellow',marker='x')
plt.axis('equal')#important car sinon l'axe ne sera pas dans la bonne direction

plt.quiver(pca.mean_[0], pca.mean_[1], pca.components_[0, 0], pca.components_[0, 1], scale=3,color='r', label='Composante principale 1')
plt.quiver(pca.mean_[0], pca.mean_[1], pca.components_[1, 0], pca.components_[1, 1], scale=3,color='g', label='Composante principale 2')
plt.grid(True)
plt.legend()
plt.show()

#%% print the explained variances
print("Variances (Percentage):")
print(pca.explained_variance_ratio_ * 100)
print()
print("Cumulative Variances (Percentage):")
print(pca.explained_variance_ratio_.cumsum() * 100)
print()
#%% visualiser le diagramme en bar des variances expliquées

# plot a scree plot
components = len(pca.explained_variance_ratio_) 
components
x=list(range(1,pca.n_components+1))
x
y=pca.explained_variance_ratio_ * 100
y
plt.bar(x,y,color='b')
plt.xlabel("Number of components")
plt.ylabel("Explained variance (%)")
plt.show()
#on voit que la première composante principale apporte 94.4% d'information
#il suffit donc de projeter nos données sur une composante



#%%
from sklearn.decomposition import PCA
#%%
pca = PCA(n_components=1)

X_pca = pca.fit_transform(X_std)
X_pca
#%% projeter les X_pca dans le repère d'origine
X_new = pca.inverse_transform(X_pca) 
X_new

#%%tracer le nouveau nuage de point avec les directions des composantes principale
m=X_pca.mean(axis=0)
m
pca.mean_
plt.scatter(X_std[:, 0], X_std[:, 1], alpha=0.2,color='blue')
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8,color='green')
#plt.scatter(mean[0],mean[1],color='red',marker='x')
plt.axis('equal')#important car sinon l'axe ne sera pas dans la bonne direction
plt.quiver(pca.mean_[0], pca.mean_[1], pca.components_[0, 0], pca.components_[0, 1], scale=3,color='r', label='Composante principale 1')

plt.axhline(0, color='red',linewidth=0.5)  # Axe horizontal
plt.axvline(0, color='red',linewidth=0.5)  # Axe vertical
plt.legend()
plt.show()

#%% en standardisant les données
scaler=StandardScaler()
X_std=scaler.fit_transform(X)
X_std
plt.scatter(X_std[:, 0], X_std[:, 1], alpha=0.2,color='blue')
plt.show()

pca = PCA(n_components=1)
m=X_pca.mean(axis=0)
X_pca = pca.fit_transform(X_std)
X_pca

X_new = pca.inverse_transform(X_pca) 
X_new

plt.scatter(X_std[:, 0], X_std[:, 1], alpha=0.2,color='blue')
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8,color='green')
#plt.scatter(mean[0],mean[1],color='red',marker='x')
plt.axis('equal')
plt.quiver(pca.mean_[0], pca.mean_[1], pca.components_[0, 0], pca.components_[0, 1], scale=3,color='r', label='Composante principale 1')

plt.legend()
plt.show()
pca.components_
