using Test

include("gram-schmidt.jl")
include("gram-schmidt-robuste.jl")

# Matrice linéairement indépendante
X_ind = [ -1.0  -1.0   1.0;
           1.0   3.0   3.0;
          -1.0  -1.0   5.0 ]

# Matrice linéairement dépendante
X_dep = [ -1.0  -1.0  -1.0;
           1.0   3.0   3.0;
          -1.0  -1.0  -1.0 ]

Q_ind = gramschmidt(X_ind)
Q_dep = gramschmidt(X_dep)

Q_ind_rob = gramschmidtrobust(X_ind)
Q_dep_robust = gramschmidtrobust(X_dep)

@test size(Q_ind, 2) == 3                
@test size(Q_dep, 2) == 3 #ne prend pas en compte les vecteurs colinéaire

@test length(Q_ind_rob) == 3            
@test length(Q_dep_robust) == 2

@test norm(Q_dep_robust[1]) ≈ 1    #approximatif pour les floating points inexact
@test norm(Q_dep_robust[2]) ≈ 1

@test abs(dot(Q_dep_robust[1], Q_dep_robust[2])) ≈ 0  
