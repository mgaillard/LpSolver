module HPCGLpSolver

using Convex
using Plots
using LinearAlgebra
using SparseArrays

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

function compute_direction(p_sol::IplpSolution, sigma)
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

function check_alpha_condition(p_sol::IplpSolution, dx, dlambda, ds, initial_residual, alpha)
     # Constants useful for the computation
     n = length(p_sol.cs)
     # Beta >= 1
     beta = 1e4
     # Gamma in [0, 1]
     gamma = 1e-4

     # Compute the potential location
     xs = p_sol.xs + alpha * dx
     lambda = p_sol.lam + alpha * dlambda
     s = p_sol.s + alpha * ds

     # Mean of the x_i*s_i coefficients
     current_mu = dot(p_sol.xs, p_sol.s)/n
     mu = dot(xs, s)/n
     rb = residual_b(p_sol.As, xs, p_sol.bs)
     rc = residual_c(s, p_sol.As, lambda, p_sol.cs)

     return (norm([rb; rc]) > initial_residual*beta*mu
          || any(xs .< 0)
          || any(s .< 0)
          || any((xs.*s) .< gamma*mu)
          || any(mu .> (1 - 0.01*alpha)*current_mu))
end

# Choose alpha in ]0; 1] IPF algorithm (page 131/310)
function pick_alpha(p_sol::IplpSolution, dx, dlambda, ds, initial_residual)
     alpha = 1.0
     
     while check_alpha_condition(p_sol, dx, dlambda, ds, initial_residual, alpha)
          alpha /= 2;
     end
     
     return alpha
end

function check_end_condition(p_sol::IplpSolution, sigma::Float64, tolerance::Float64)
     # Compute the duality measure
     mu = dot(p_sol.xs, p_sol.s) / length(p_sol.xs)
     # Compute the normalized residual
     residual = norm([residual_c(p_sol); residual_b(p_sol)]) / norm([p_sol.bs; p_sol.s])
     # End condition
     return (mu <= tolerance) && (residual <= tolerance)
end

function feasibility_diagnostic(p_sol::IplpSolution, tolerance::Float64)
     println("Feasibility diagnostic")
     println("x > 0: ", all(p_sol.xs .> 0))
     println("s > 0: ", all(p_sol.s .> 0))
     println("Norm of residual_c: ", norm(residual_c(p_sol)))
     println("Norm of residual_b: ", norm(residual_b(p_sol)))
     residual = norm([residual_c(p_sol); residual_b(p_sol)]) / norm([p_sol.bs; p_sol.s])
     println("Norm of residual: ", residual)
     println("Tolerance test for residual: ", residual <= tolerance)
     mu = dot(p_sol.xs, p_sol.s) / length(p_sol.xs)
     println("Mu: ", mu)
     println("Tolerance test for mu: ", mu <= tolerance)
end

function interior_point_method(p_sol::IplpSolution, sigma::Float64, tolerance::Float64, max_iterations::Int)
     step = 1

     # Compute initial residuals
     # Used later when choosing alpha
     initial_mu = dot(p_sol.xs, p_sol.s) / length(p_sol.xs)
     initial_residual = norm([residual_b(p_sol); residual_c(p_sol)]) / initial_mu

     try  
          while (step < max_iterations && !check_end_condition(p_sol, sigma, tolerance))
               println("Step: ", step)
               feasibility_diagnostic(p_sol, tolerance)

               # Compute a descent direction biais toward the central path
               dx, dlambda, ds = compute_direction(p_sol, sigma)
               # Perform a line search with the constraint that we need to stay in the feasible region
               alpha = pick_alpha(p_sol, dx, dlambda, ds, initial_residual)
          
               # Step towards the descent direction
               p_sol.xs += alpha * dx
               p_sol.lam += alpha * dlambda
               p_sol.s += alpha * ds
          
               step += 1
          end
     catch e
          println("An exception happened! ")
          println(e)
          frames = stacktrace()
          for (count,f) in enumerate(frames)
               println("[",count, "] ",f)
          end

          return p_sol
     end

     # Check if the problem is solved
     # If so, set x and the flag
     if check_end_condition(p_sol, sigma, tolerance)
          # The solution is xs without the slack variables
          p_sol.x = p_sol.xs[1:length(p_sol.x)]
          # Problem solved => flag = true
          p_sol.flag = true
     end

     return p_sol
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
function iplp(problem::IplpProblem, tolerance::Float64; max_iterations=100)
     sigma = 0.5

     m, n = size(problem.A)

     # If the contraint has inequality constraints on x, we reformulate it.
     # if (any(problem.lo .!= 0.0) || any(problem.hi .< Infinity))
     
     if length(findall(problem.hi .!= Infinity)) > 0
          println("Convert to standard form")
          hi = problem.hi - problem.lo

          noninf_constraint_indice = findall(hi .!= Infinity)
          noninf_hi_num = length(noninf_constraint_indice)
          noninf_hi = hi[noninf_constraint_indice] 
          
          As = [problem.A                zeros(m,noninf_hi_num); 
                zeros(noninf_hi_num, n)  Matrix{Float64}(I, noninf_hi_num, noninf_hi_num)]
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
     ms, ns = size(As)
     zeta = 128.0
     xs = zeta * ones(ns)
     lam = zeros(ms)
     s = zeta * ones(ns)
     initial_solution = IplpSolution(x, false, cs, As, bs, xs, lam, s)

     # Solve the problem
     shifted_solution = interior_point_method(initial_solution, sigma, tolerance, max_iterations)
     # Solution is shifted by problem.lo, we need to shift it back
     shifted_solution.x = shifted_solution.x[1:n] + problem.lo

     return shifted_solution
end

export IplpProblem
export IplpSolution
export iplp
end
