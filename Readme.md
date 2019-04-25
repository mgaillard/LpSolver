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

- Check feasible before solving
- Try the first form, use Ldlt
- Better alpha picking
- Better predictor 

## Testing
We test our algorithm on the problems from the University of Flordia Sparse Matrix repository.

**tol = 1e-7**

| Problem name  |     Sucess    |     Res           |       Mu                  | Iteration |
| ------------- | ------------- |------------       | ---------------           | --------- |
| lp_afiro      | &#x2611;      |0.000000000111     | 0.000000016570            | N/A       |
| lp_brandy     | &#x2611;      |0.000000022079     | 0.000000071644            | N/A       |
| lp_adlittle   | &#x2611;      |0.000000000000     | 0.000000097575            | N/A       |
| lp_agg        | &#x2611;      |0.000000096385     | 0.000000027158            | N/A       |
| lp_stocfor1   | &#x2611;      |0.000000000001     | 0.000000078902            | N/A       |
| lp_fit1d      | &#x2612;      |0.675714541101     | 142.660889550226          | N/A       |
| lp_25fv47     | &#x2612;      |1.109278128352     | 0.000000000000            | N/A       |
| lp_ganges     | &#x2612;      |0.5622249587237997 | 4.1857702837761206e-5     | N/A       |
| lpi_chemcom   | &#x2612;      |0.5                | 0.5                       | N/A       |

**tol = 1e-8**

| Problem name  |     Sucess    |
| ------------- | ------------- |
| lp_afiro      | &#x2611;      |
| lp_brandy     | &#x2611;      |
| lp_adlittle   | &#x2611;      |
| lp_agg        | &#x2612;      |
| lp_stocfor1   | &#x2611;      |
| lp_fit1d      | &#x2612;      |
| lp_25fv47     | &#x2612;      |
| lp_ganges     | &#x2612;      |
| lpi_chemcom   | &#x2612;      |

## Licence
See the LICENSE file.
