module HPCGLpSolver

using Convex
# TODO: Remove this dependency
using ECOS
using Plots
using LinearAlgebra
using SparseArrays

mutable struct IplpProblem
     c::Vector{Float64}
     # A::SparseMatrixCSC{Float64} 
     A::Matrix{Float64} # TODO: use SparseMatrixCSC instead
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
     # As::SparseMatrixCSC{Float64}
     As::Matrix{Float64} # TODO: use SparseMatrixCSC instead
     # The right hand side (b) in standard form
     bs::Vector{Float64}
     # The solution in standard form
     xs::Vector{Float64}
     # The solution lambda in standard form
     lam::Vector{Float64}
     # The solution s in standard form
     s::Vector{Float64}
 end

# Solve the central-path problem for interior point methods.
function ip_central(c, A, b, tau)
    x = Variable(length(c))
    p = minimize(c'*x - tau*sum(log(x)))
    p.constraints += A*x == b
    p.constraints += x .>= 0
    solve!(p, ECOSSolver(verbose=false))
    return vec(x.value), p
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
     Fc = [p_sol.s + p_sol.As' * p_sol.lam .- p_sol.cs;
           p_sol.As * p_sol.xs .- p_sol.bs;
           p_sol.xs .* p_sol.s .- sigma * mu]

     # Newton method: Direction in which we perform the line search
     d = J\-Fc

     # Split the components of the direction vector
     dx = d[1:n]
     dlambda = d[n + 1:m + n]
     ds = d[m + n + 1:end]

     return dx, dlambda, ds
end

# Choose alpha with the backtracking method
function pick_alpha(x, lambda, s, dx, dlambda, ds)
     alpha = 1.0

     while (any((x + alpha * dx) .< 0) || any((s + alpha * ds) .< 0))
          alpha /= 2;
     end

     return alpha
end

# Find a starting point
function find_starting_point(c, A, b, tau)
     x, prob = ip_central(c, A, b, tau)
     lambda = vec(prob.constraints[1].dual)
     s = tau./x
     return x, lambda, s
end

function check_end_condition(p_sol::IplpSolution, sigma::Float64, tolerance::Float64)
     # Compute the duality measure
     mu = dot(p_sol.xs, p_sol.s) / length(p_sol.xs)
     # Compute the normalized residual
     Fc = [p_sol.s + p_sol.As' * p_sol.lam .- p_sol.cs;
           p_sol.As * p_sol.xs .- p_sol.bs]
     residual = norm([Fc[1]; Fc[2]]) / norm([p_sol.bs; p_sol.s])
     # End condition
     return (all(mu .<= tolerance) && residual <= tolerance)
end

function interior_point_method(p_sol::IplpSolution, sigma::Float64, tolerance::Float64, max_iterations::Int)
     step = 1

     while (step < max_iterations && !check_end_condition(p_sol, sigma, tolerance))
          # Compute a descent direction biais toward the central path
          dx, dlambda, ds = compute_direction(p_sol, sigma)
          # Perform a line search with the constraint that we need to stay in the feasible region
          alpha = pick_alpha(p_sol.xs, p_sol.lam, p_sol.s, dx, dlambda, ds)
     
          # Step towards the descent direction
          p_sol.xs += alpha * dx
          p_sol.lam += alpha * dlambda
          p_sol.s += alpha * ds
     
          step += 1
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

function iplp(problem::IplpProblem, tolerance::Float64)
     sigma = 0.5
     max_iterations = 100
     tau = 10.0
     
     # TODO: check if unbound
     m, n = size(problem.A)
     cs = [problem.c; zeros(m)]
     x = zeros(n)
     As = [problem.A Matrix{Float64}(I, m, m)]
     bs = problem.b

     # TODO: Find the starting point without relying on another solver
     xs, lam, s = find_starting_point(cs, As, problem.b, tau)

     initial_solution = IplpSolution(x, false, cs, As, bs, xs, lam, s)

     return interior_point_method(initial_solution, sigma, tolerance, max_iterations)
end

export IplpProblem
export IplpSolution
export iplp
end