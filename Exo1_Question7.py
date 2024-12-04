# Étapes de calcul pour la projection sur les axes u1 et u2

# C1 = A_std * u1 =
# [
#   -0.82  -1.2   0.7
#   -0.82  -0.15  0.7
#    0      0.19  0.7
#    1.64   1.51  0.7
# ] * [0.7] =
# [
#   -0.82 * 0.7 - 1.2 * 0.7
#   -0.82 * 0.7 - 0.15 * 0.7
#    0 * 0.7 + 0.19 * 0.7
#    1.64 * 0.7 + 1.51 * 0.7
# ] =
# [
#   -1.414
#   -0.924
#    0.133
#    2.205
# ]

# C2 = A_nd * u2 =
# [
#   -0.82  -1.2  -0.7
#   -0.82  -0.15 -0.7
#    0      0.19 -0.7
#    1.64   1.51 -0.7
# ] * [-0.7] =
# [
#    0.82 * 0.7 + 1.2 * -0.7
#    0.82 * 0.7 - 0.15 * -0.7
#   -0.19 * -0.7
#    1.64 * -0.7 - 1.51 * 0.7
# ] =
# [
#    0.266
#   -0.224
#   -0.133
#    0.091
# ]

# D’où A_pca =
# [
#   -1.414   0.266
#    0.924  -0.224
#    0.133  -0.133
#    2.205   0.091
# ]

# Les coordonnées suivant u1 et u2 sont tracées sur le graphe ci-dessous.