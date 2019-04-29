module HPCGLpSolver

using Plots
using LinearAlgebra
using SparseArrays

include("HPCGCholeskySolver.jl")

const Infinity = 1.0e308

mutable struct IplpProblem
     c::Vector{Float64}
     A::SparseMatrixCSC{Float64}
     b::Vector{Float64}
     lo::Vector{Float64}
     hi::Vector{Float64}
 end
 
mutable struct IplpSolution
     # Solution vector
     x::Vector{Float64}
     # A true/false flag indicating convergence or not
     flag::Bool         
     # The objective vector in standard form
     cs::Vector{Float64}
     # The constraint matrix in standard form
     As::SparseMatrixCSC{Float64}
     # The right hand side (b) in standard form
     bs::Vector{Float64}
     # The solution in standard form
     xs::Vector{Float64}
     # The solution lambda in standard form
     lam::Vector{Float64}
     # The solution s in standard form
     s::Vector{Float64}
 end

# Compute the residual of the KKT condition Ax = b
function residual_c(s::Vector{Float64},
                    As::SparseMatrixCSC{Float64},
                    lam::Vector{Float64},
                    cs::Vector{Float64})
     return s + As' * lam .- cs
end
function residual_c(p_sol::IplpSolution)
     return residual_c(p_sol.s, p_sol.As, p_sol.lam, p_sol.cs)
end

# Compute the residual of the KKT condition A'lam + s = c
function residual_b(As::SparseMatrixCSC{Float64},
                    xs::Vector{Float64},
                    bs::Vector{Float64})
     return As * xs .- bs
end
function residual_b(p_sol::IplpSolution)
     return residual_b(p_sol.As, p_sol.xs, p_sol.bs)
end

"""
Standard form of the step equation
"""
function compute_predictor_standard(p_sol::IplpSolution, sigma)
     # Compute the steps 
     n = length(p_sol.cs)
     m = size(p_sol.As, 1)

     # Jacobian matrix of the function for Newton method
     J = [zeros(n,n)             p_sol.As'  Matrix{Float64}(I,n,n); 
          p_sol.As               zeros(m,m) zeros(m,n);
          Diagonal(vec(p_sol.s)) zeros(n,m) Diagonal(vec(p_sol.xs))]

     mu = dot(p_sol.xs, p_sol.s)/n

     # We want to find the zero of this function
     Fc = [residual_c(p_sol);
           residual_b(p_sol); 
           p_sol.xs .* p_sol.s .- sigma * mu]

     # Newton method: Direction in which we perform the line search
     d = J\-Fc

     # Split the components of the direction vector
     dx = d[1:n]
     dlambda = d[n + 1:m + n]
     ds = d[m + n + 1:end]

     return dx, dlambda, ds
end

"""
Augmented form of the step equation
"""
function compute_predictor_augmented(p_sol::IplpSolution, sigma)
     # Compute the steps 
     n = length(p_sol.cs)
     m = size(p_sol.As, 1)

     # Precompute D matrix
     mu = dot(p_sol.xs, p_sol.s)/n
     D2 = Diagonal(vec(p_sol.s ./ p_sol.xs))
     X_inv = Diagonal(vec(1.0 ./ p_sol.xs))
     residual_xs = p_sol.xs .* p_sol.s .- sigma * mu

     # Modified Jacobian matrix
     J = [zeros(m, m)  p_sol.As;
          p_sol.As'    -D2]

     # Modified function for Newton method
     Fc = [residual_b(p_sol); residual_c(p_sol) - X_inv * residual_xs]

     # Direction in which we perform the line search
     d = J \ -Fc

     dlambda = d[1:m]
     dx = d[m + 1:m + n]
     ds = -X_inv * (residual_xs + p_sol.s .* dx)

     return dx, dlambda, ds
end

"""
HPCG Cholesky
Given a symmetric semi-positive matrix M, this function computes the factorization form of cholesky
Return a L matrix
"""
function hpcg_cholesky(M::SparseMatrixCSC{Float64}, cholesky_solver::Function, tol = 1e-12)
    return cholesky_solver(M, tol)
end

"""
Return the left-hand side of the step equation in normal form
Factored using Cholesky
Return a L matrix
"""
function normal_equation_factored(p_sol::IplpSolution)
     A = p_sol.As
     X = Diagonal(vec(p_sol.xs))
     S = Diagonal(vec(p_sol.s))
     D2 = Diagonal(vec(p_sol.xs ./ p_sol.s))
     M = (A * D2 * A')
          
     # Factorization using Cholesky
     return hpcg_cholesky(M, cholesky_skip)
end

"""
Solve the normal form of the step equation with residuals [rb; rc; rxs]
normal_factored_lhs is the left-hand side of the step equation in normal form
Use normal_equation_factored(p_sol::IplpSolution) to compute it
"""
function solve_normal_equation(p_sol::IplpSolution, normal_factored_lhs, rb, rc, rxs)
    # Solve delta_lambda, delta_s, delta_x
    A = p_sol.As
    X = Diagonal(vec(p_sol.xs))
    S = Diagonal(vec(p_sol.s))
    D2 = Diagonal(vec(p_sol.xs ./ p_sol.s))
    M = (A * D2 * A')
         
    # Solve using Cholesky
    b = -rb + A * (-S \ X * rc + S \ rxs)
    dlambda = hpcg_cholesky_solve(normal_factored_lhs, b)

    ds = -rc - A' * dlambda
    dx = -S\(rxs + X * ds)

    return dx, dlambda, ds
end

function compute_corrector(p_sol::IplpSolution, normal_factored_lhs, sigma, dx_affine, ds_affine)
     # Compute the steps 
     n = length(p_sol.cs)
     m = size(p_sol.As, 1)

     mu = dot(p_sol.xs, p_sol.s)/n
     rb = zeros(m)
     rc = zeros(n)
     rxs = dx_affine .* ds_affine .- sigma * mu;

     return solve_normal_equation(p_sol, normal_factored_lhs, rb, rc, rxs)
end

function compute_predictor(p_sol::IplpSolution, normal_factored_lhs, sigma)
     # Compute the steps 
     n = length(p_sol.cs)
     m = size(p_sol.As, 1)

     mu = dot(p_sol.xs, p_sol.s)/n
     rb = residual_b(p_sol)
     rc = residual_c(p_sol)
     rxs = p_sol.xs .* p_sol.s .- sigma * mu;

     return solve_normal_equation(p_sol, normal_factored_lhs, rb, rc, rxs)
end

function alpha_max(x, dx, hi = 1.0)
     n = length(x)
     alpha = hi
     ind = -1
 
     for i=1:n
          if dx[i] < 0.0
               curr_alpha = -x[i]/dx[i]
               if curr_alpha < alpha
                    alpha = curr_alpha
                    ind = i
               end
         end
     end
 
     return alpha, ind
 end

function check_end_condition(p_sol::IplpSolution, tolerance::Float64)
     # Compute the duality measure
     mu = dot(p_sol.xs, p_sol.s) / length(p_sol.xs)
     # Compute the normalized residual
     residual = norm([residual_c(p_sol); residual_b(p_sol); p_sol.xs .* p_sol.s]) / norm([p_sol.bs; p_sol.s])
     # End condition
     return (mu <= tolerance) && (residual <= tolerance)
end

function feasibility_diagnostic(p_sol::IplpSolution, tolerance::Float64)
     println("Feasibility diagnostic")
     println("x > 0: ", all(p_sol.xs .> 0))
     println("s > 0: ", all(p_sol.s .> 0))
     println("Norm of residual_c: ", norm(residual_c(p_sol)))
     println("Norm of residual_b: ", norm(residual_b(p_sol)))
     residual = norm([residual_c(p_sol); residual_b(p_sol); p_sol.xs .* p_sol.s]) / norm([p_sol.bs; p_sol.s])
     println("Norm of residual: ", residual)
     mu = dot(p_sol.xs, p_sol.s) / length(p_sol.xs)
     println("Mu: ", mu)
     println("Tolerance test for residual: ", residual <= tolerance)
     println("Tolerance test for mu: ", mu <= tolerance)
end

"""
Adaptation of sigma
Page 196
"""
function predictor_corrector(p_sol::IplpSolution, tolerance::Float64, max_iterations::Int)
     n = length(p_sol.cs)
     
     step = 1
     while (step < max_iterations && !check_end_condition(p_sol, tolerance))
          println("Step: ", step)
          feasibility_diagnostic(p_sol, tolerance)
          
          # Precompute the left-hand side of the step equation in normal form
          normal_factored_lhs = normal_equation_factored(p_sol)

          # Predictor step
          affine_dx, affine_dlambda, affine_ds = compute_predictor(p_sol, normal_factored_lhs, 0.0) 
          
          cur_mu = dot(p_sol.xs, p_sol.s)/n
          
          affine_primal_alpha, _ = alpha_max(p_sol.xs, affine_dx)
          affine_dual_alpha, _ = alpha_max(p_sol.s, affine_ds)
          
          mu_aff = dot(p_sol.xs + affine_primal_alpha * affine_dx, p_sol.s + affine_dual_alpha * affine_ds) / n
               
          affine_sigma = clamp((mu_aff / cur_mu)^3, 0.0, 1.0)

          # Corrector step
          dx_c, dlambda_c, ds_c = compute_corrector(p_sol, normal_factored_lhs, affine_sigma, affine_dx, affine_ds)
          dx = affine_dx + dx_c
          dlambda = affine_dlambda + dlambda_c
          ds = affine_ds + ds_c

          max_primal_alpha, max_primal_alpha_ind = alpha_max(p_sol.xs, dx, Infinity) 
          max_dual_alpha, max_dual_alpha_ind = alpha_max(p_sol.s, ds, Infinity)

          # Pick alpha in theory
          primal_alpha = min(0.9 * max_primal_alpha, 1.0)
          dual_alpha = min(0.9 * max_dual_alpha, 1.0)

          # Pick alpha in practice (if max_alpha indices are defined we can use the heuristic)
          if max_primal_alpha_ind >= 0 && max_dual_alpha_ind >= 0
               gamma_f = 0.05
               mu_plus = dot(p_sol.xs + max_primal_alpha * affine_dx, p_sol.s + affine_dual_alpha * affine_ds) / n

               f_primal = (((gamma_f * mu_plus) / (p_sol.s[max_primal_alpha_ind] + max_dual_alpha*ds[max_primal_alpha_ind])) - p_sol.xs[max_primal_alpha_ind]) / (max_primal_alpha * dx[max_primal_alpha_ind])
               f_dual = (((gamma_f * mu_plus) / (p_sol.xs[max_dual_alpha_ind] + max_primal_alpha*dx[max_dual_alpha_ind])) - p_sol.s[max_dual_alpha_ind]) / (max_dual_alpha * ds[max_dual_alpha_ind])

               primal_alpha = max(1.0 - gamma_f, f_primal)*max_primal_alpha
               dual_alpha = max(1.0 - gamma_f, f_dual)*max_dual_alpha
          end

          # Step towards the descent direction
          primal_alpha = clamp(primal_alpha, 0.0, 1.0)
          dual_alpha = clamp(dual_alpha, 0.0, 1.0)
          p_sol.xs += primal_alpha * dx
          p_sol.lam += dual_alpha * dlambda
          p_sol.s += dual_alpha * ds

          # @show primal_alpha
          # @show dual_alpha
          # @show mu_aff
          # @show cur_mu
          # @show affine_sigma
          
          step += 1
     end
     
     # Check if the problem is solved
     # If so, set x and the flag
     if check_end_condition(p_sol, tolerance)
          # The solution is xs without the slack variables
          p_sol.x = p_sol.xs[1:length(p_sol.x)]
          # Problem solved => flag = true
          p_sol.flag = true
     end

     return p_sol
end

function remove_useless_rows(problem::IplpProblem)::IplpProblem
     # Histogram of values in the rows
     hist = zeros(size(problem.A, 1))

     rows = rowvals(problem.A)
     vals = nonzeros(problem.A)
     m, n = size(problem.A)
     for i = 1:n
          for j in nzrange(problem.A, i)
               row = rows[j]
               hist[row] += 1
          end
     end
     # Indices of rows with at least one non zero
     ind = findall(hist .> 0.0)
     nb_zeros = count(hist .== 0.0)

     if nb_zeros > 0
          problem.A = problem.A[ind, :]
          problem.b = problem.b[ind]
          println("Removed ", nb_zeros, " rows in matrix A and vector b")
     end

     return problem
end

"""
Solve min 0.5 * ||x|| st Ax = b
By using the augmented system of normal equations
"""
function solve_constrained_least_norm(A, b)
     m, n = size(A)
     # Step 1, form the block system
     M = [Matrix{Float64}(I,n,n) A'; A zeros(m,m)]
     # Step 2, solve
     z = M\[zeros(n); b]
     # Step 3, extract 
     return z[1:n]
end

"""
Mehrotra section 7 page 15 of 27
"""
function find_starting_point(A, b, c)
     m, n = size(A)

     x_hat = solve_constrained_least_norm(A, b)
     lambda_s_hat = solve_constrained_least_norm([A' Matrix{Float64}(I,n,n)], c)
     lambda_hat = lambda_s_hat[1:m]
     s_hat = lambda_s_hat[m + 1:m + n]

     delta_x = max(-1.5 * minimum(x_hat), 0.0)
     delta_s = max(-1.5 * minimum(s_hat), 0.0)

     numerator_x_term = x_hat + delta_x * ones(n, 1)
     numerator_s_term = s_hat + delta_s * ones(n, 1)     

     delta_x_hat = delta_x + 0.5 * (dot(numerator_x_term, numerator_s_term) / sum(numerator_s_term))
     delta_s_hat = delta_s + 0.5 * (dot(numerator_x_term, numerator_s_term) / sum(numerator_x_term))
     
     xs = vec(x_hat + delta_x_hat * ones(n, 1))
     lam = vec(lambda_hat)
     s = vec(s_hat + delta_s_hat * ones(n, 1))

     return xs, lam, s
end

"""
soln = iplp(Problem,tol) solves the linear program:

   minimize c'*x where Ax = b and lo <= x <= hi

where the variables are stored in the following struct:

   Problem.A
   Problem.c
   Problem.b   
   Problem.lo
   Problem.hi

and the IplpSolution contains fields

  [x,flag,cs,As,bs,xs,lam,s]

which are interpreted as   
a flag indicating whether or not the
solution succeeded (flag = true => success and flag = false => failure),

along with the solution for the problem converted to standard form (xs):

  minimize cs'*xs where As*xs = bs and 0 <= xs

and the associated Lagrange multipliers (lam, s).

This solves the problem up to 
the duality measure (xs'*s)/n <= tol and the normalized residual
norm([As'*lam + s - cs; As*xs - bs; xs.*s])/norm([bs;cs]) <= tol
and fails if this takes more than maxit iterations.
"""
function iplp(problem::IplpProblem, tolerance::Float64; maxit=100)::IplpSolution
     sigma = 0.5

     # Remove rows of A and b populated only with zeros.
     problem = remove_useless_rows(problem)

     m, n = size(problem.A)

     # If the contraint has inequality constraints on x, we reformulate it.
     
     if (any(problem.lo .!= 0.0) || any(problem.hi .< Infinity))
          println("Convert to standard form")

          noninf_constraint_indice = findall(problem.hi .< Infinity)
          noninf_hi_num = length(noninf_constraint_indice)
          noninf_hi = problem.hi[noninf_constraint_indice] - problem.lo[noninf_constraint_indice]

          # Modify A to take account for the slacks
          right_block_slacks = Matrix{Float64}(I, noninf_hi_num, noninf_hi_num)
          left_block_constraints = zeros(noninf_hi_num, n)
          left_block_constraints[:, noninf_constraint_indice] = right_block_slacks
          
          As = [problem.A               zeros(m, noninf_hi_num); 
                left_block_constraints  right_block_slacks]
          bs = [problem.b - problem.A * problem.lo; noninf_hi]
          cs = [problem.c; zeros(noninf_hi_num)]
     else
          println("No conversion needed, the problem is already in standard form")
          cs = problem.c
          As = problem.A
          bs = problem.b
     end

     # By default the solution vector is zero
     x = zeros(n)

     # Find a starting point
     xs, lam, s = find_starting_point(As, bs, cs)

     initial_solution = IplpSolution(x, false, cs, As, bs, xs, lam, s)

     # Solve the problem
     standard_solution = predictor_corrector(initial_solution, tolerance, maxit)

     # Solution is shifted by problem.lo, we need to shift it back
     standard_solution.x = standard_solution.x[1:n] + problem.lo

     return standard_solution
end

export IplpProblem
export IplpSolution
export iplp
end
