"""
Src: Primal-Dual-Interior-Point-Method, Page 218, 219
"""

using LinearAlgebra
using SparseArrays

# Skip small pivots
function cholesky_skip(M, tol=1e-15)
    m,n = size(M)
    @assert m==n "To factorize, M should be a squared matrix"

    L = zeros(m,m)
    max_pivot = Float64(0.0)

    for i in 1:m
        max_pivot = max(max_pivot, M[i,i])
        if M[i,i] >= max_pivot * tol
            L[i,i] = sqrt(M[i,i])
            for j in i + 1 : m
                L[j,i] = M[j,i] / L[i,i]
                for k in i+1 : j
                     M[j,k] = M[j,k] - L[j,i] * L[k,i]
                end
            end
        else
            # skip 
            continue
        end  
    end
    return L
end

function cholesky_big(M, tol=1e-15)
    m,n = size(M)
    @assert m==n "To factorize, M should be a squared matrix"

    L = zeros(m,m)
    println("Type of L: ",typeof(L))
    max_pivot = 0.0
    for i in 1:m
        max_pivot = max(max_pivot, M[i,i])
        if M[i,i] >= max_pivot * tol
            L[i,i] = sqrt(M[i,i])
            for j in i + 1 : m
                L[j,i] = M[j,i] / L[i,i]
                for k in i+1 : j
                     M[j,k] = M[j,k] - L[j,i] * L[k,i]
                end
            end
        else
            L[i,i] = Float64(1e64)
            L[i+1:m, i] .= Float64(1e-64)
            for j in i + 1 : m
                for k in i+1 : j
                     M[j,k] = M[j,k] - Float64(1e-64)
                end
            end

        end
    end
    return L
end

"""Sovle LL' x = b
Sometimes, L will not be exact LowerTriangular. This function takes care of this situation.
"""
function hpcg_cholesky_solve(L, b)
    m,n = size(L)
    @assert m == n "L should be a squared matrix"
    @assert m >= 1 "L should not be an empty matrix"

    x = fill(Float64(0.0), m)
    L_rank = rank(L)
    if m == L_rank
        L = LowerTriangular(L)
        x = L' \ (L \ b)
    else
        # Solve LL' x = b
        L_upper = copy(L')

        # Solve L y = b
        y = fill(Float64(0.0), m)
        # LowerTriangular 
        for col in 1:m
            if all(x->x==0.0, L[:,col])
                continue
            end

            y[col] = b[col] / L[col,col]
            
            if col == m
                break
            end

            for r in col+1:m
                b[r] = b[r] - L[r, col] * y[col]
            end
        end

        # UpperTriangular
        for col in m:-1:1
            if all(x->x==0.0, L[:,col])
                continue
            end

            x[col] = y[col] / L_upper[col, col]

            if col == 1
                break
            end 

            for r in col - 1 :-1:1
                y[r] = y[r] - L_upper[r,col] * x[col]
            end
        end
    end

    return x
end