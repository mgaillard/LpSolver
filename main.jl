push!(LOAD_PATH, "./")
import HPCGLpSolver

# Problem definition
c = [-1.0; -2.0]
A = [-2.0 1.0; 
      -1.0 2.0; 
       1.0 0.0]
b = [2.0; 7.0; 3.0]
lo = [0.0; 0.0]
hi = [16.0; 16.0]

problem = HPCGLpSolver.IplpProblem(c, A, b, lo, hi)

# Solve
solution = HPCGLpSolver.iplp(problem, 1e-8)

# Display the solution
if solution.flag
     println("Solution found")
else
     println("Solution not found")
end
display(solution.x)