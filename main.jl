push!(LOAD_PATH, "./")
import HPCGLpSolver

using LinearAlgebra
using SparseArrays
using MatrixDepot

function convert_matrixdepot(P::MatrixDepot.MatrixDescriptor)::HPCGLpSolver.IplpProblem
     return HPCGLpSolver.IplpProblem(vec(P.c), P.A, vec(P.b), vec(P.lo), vec(P.hi))
end

function create_problem(name::String)::HPCGLpSolver.IplpProblem
     # Our own test examples
     if name == "HPCG/lp_test"
          c = [-1.0; -2.0; 0.0; 0.0; 0.0]
          A = [-2.0 1.0 1.0 0.0 0.0;
               -1.0 2.0 0.0 1.0 0.0;
                1.0 0.0 0.0 0.0 1.0]
          b = [2.0; 7.0; 3.0]
          lo = [0.0; 0.0; 0.0; 0.0; 0.0]
          hi = [16.0; 16.0; 16.0; 16.0; 16.0]
          return HPCGLpSolver.IplpProblem(c, A, b, lo, hi)

     # Examples from the University of Flordia Sparse Matrix repository
     # URL: http://www.cise.ufl.edu/research/sparse/matrices/LPnetlib
     else
          # List all available matrices: listnames("LPnetlib/*")
          # Display informations about the problem lp_afiro: mdinfo("LPnetlib/lp_afiro")

          md = mdopen(name)
          # Workaround for a bug: https://github.com/JuliaMatrices/MatrixDepot.jl/issues/34
          MatrixDepot.addmetadata!(md.data)
          return convert_matrixdepot(md)
     end
end

# Problem definition
# problem = create_problem("LPnetlib/lp_ganges")
problem = create_problem("LPnetlib/lp_afiro")

# Solve
solution = HPCGLpSolver.iplp(problem, 1e-4; max_iterations=1000)

# Display the solution
if solution.flag
     println("Solution found")
else
     println("Solution not found")
end

println(solution.x)
println("Optimal value: ", problem.c' * solution.x)

