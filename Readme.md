Interior-point solver for Linear Programming
============================================

## Getting started

### What is the interface
```Julia
# Interface for describing the problem
mutable struct IplpProblem
    c::Vector{Float64}
    A::SparseMatrixCSC{Float64}
    b::Vector{Float64}
    lo::Vector{Float64}
    hi::Vector{Float64}
end

# Solves the linear program
function iplp(problem::IplpProblem, tolerance::Float64; max_iterations=100)
```
### Example
```Julia
# Problem definition
c = [-1.0; -2.0; 0.0; 0.0; 0.0]
A = [-2.0 1.0 1.0 0.0 0.0;
     -1.0 2.0 0.0 1.0 0.0;
      1.0 0.0 0.0 0.0 1.0]
b = [2.0; 7.0; 3.0]
lo = [0.0; 0.0; 0.0; 0.0; 0.0]
hi = [16.0; 16.0; 16.0; 16.0; 16.0]
problem = IplpProblem(c, A, b, lo, hi)

# Solve
solution = iplp(problem, 1e-4)

# Display the solution
if solution.flag
    println("Solution found")
    display(solution.x)
end
```

## Dependencies

- Julia (version >= 1.1)
- Convex
- SparseArrays

## Features
Based on Wright, S. J. (1997). Primal-dual interior-point methods (Vol. 54). Siam.

## Authors
**Mathieu Gaillard** and **Yichen Sheng**

[HPCG Laboratory](http://hpcg.purdue.edu/)  
Department of Computer Graphics Technology  
Purdue University

This software has been developed for the [CS-520: Computational Optimization](https://www.cs.purdue.edu/homes/dgleich/cs520-2019) class at Purdue University.

## Future work

- Check feasible before solving in predictor-corrector method
- Try the first form, use Ldlt
- Better alpha picking
- Better predictor 

## Testing
To see sample testing, please run: 

```
julia HPCGLpSolver_Test.jl
```

For specific problem, please change the problem, tweak the parameters in main, and run:
```
julia main.jl
```


## Licence
See the LICENSE file.
