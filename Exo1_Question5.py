#5. Quelles sont les taux de variances 
# expliqués par chaque vecteur propre ?

# Le taux de variance expliqué par 
# le vecteur u1 (noté U1)
# tau1 = (lambda1 / (lambda1 + lambda2)) * 100 
# (lambda noté d mais tu finis pas la boucle du d en bas)
# tau1 = (2.63 / (2.63 + 0.05)) * 100 ≈ 98%
# Le taux de variance expliqué par le vecteur u2 (noté U2)
# tau2 = (lambda2 / (lambda1 + lambda2)) * 100
# tau2 = (0.05 / (2.63 + 0.05)) * 100 ≈ 2%

# Conclusion :
# La projection sur l'axe porté par u1 
# conserve environ 98% des variations des données.
# La projection sur u1 est donc suffisante 
# pour représenter les données sans perdre une grande quantité d'informations.
# Cela permet de réduire la dimension des données de 2D à 1D.

