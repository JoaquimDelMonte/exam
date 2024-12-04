'''
Calcul des valeurs propres :

λ(λ) = det(V - λI) =
| 1.34-λ  1.29 | 
| 1.29  1.34-λ |
= (1.34-λ)² - (1.29)²
= (1.34-λ+1.29)(1.34-λ-1.29)
= (0.05-λ)(2.63-λ)

D'où les valeurs propres de V sont :
 λ₁ = 2.63 et λ₂ = 0.05

Calcul des vecteurs propres :
Vu = λu <=> (V - λI)u = 0
Pour λ₁ = 2.63 :

| 1.34-2.63  1.29 | |x| = |0|
| 1.29     1.34-2.63 | |y| = |0|

<=> { 1.34x + 1.29y = 2.63x
1.29x + 1.34y = 2.63y

<=> x = y

Donc u₁ = (1, 1)

• soit μ₁ = [x y] / Vu2  0.05 μ2 <=> 
{ 1.34x + 1.29y = 0.05x
{ 1.29x + 1.34y = 0.05y 
on prend μ₂ = (1, -1)

pour former un repère orthonormé, 
les vecteurs directeurs des axes doivent être
orthogonaux (⊥) et normés (de normes 1)

μ₁ • μ₂ = xx' + yy' 
= 1×1 + 1×(-1) = 0 ⇒ μ₁ ⊥ μ₂.

||μ₁|| = √(x² + y²) = √(1² + 1²) 
= √2 ≠ 1 de même pour μ₂ ⇒ il faut normaliser
μ₁' = μ₁ / ||μ₁||
'''