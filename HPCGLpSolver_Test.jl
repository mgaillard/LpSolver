push!(LOAD_PATH, "./")
import HPCGLpSolver
using LinearAlgebra
using SparseArrays
using MatrixDepot
using Printf

function convert_matrixdepot(P::MatrixDepot.MatrixDescriptor)::HPCGLpSolver.IplpProblem
    return HPCGLpSolver.IplpProblem(vec(P.c), P.A, vec(P.b), vec(P.lo), vec(P.hi))
end

# "Ground truth" optimal value is coming from https://www.cise.ufl.edu/research/sparse/matrices/LPnetlib/ 
opt_dict = Dict("lp_afiro"=>-4.6475314286E+02, 
                "lp_brandy"=>1.5185098965E+03, 
                "lp_fit1d"=>-9.1463780924E+03,
                "lp_adlittle"=>2.2549496316E+05,
                "lp_agg"=>-3.5991767287E+07,
                "lp_ganges"=>-1.0958636356E+05,
                "lp_stocfor1"=>-4.1131976219E+04,
                "lp_25fv47"=>5.5018458883E+03)


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

"""
Test problems given by prof 
SRC: https://www.cs.purdue.edu/homes/dgleich/cs520-2019/project.html 
'afiro','brandy','fit1d','adlittle','agg','ganges','stocfor1', '25fv47', 'chemcom'
"""
function test_problems()
     problem_folder = "LPnetlib/"
     problem_sets = ["lp_afiro", "lp_brandy", "lp_fit1d", "lp_adlittle", "lp_agg", "lp_ganges", "lp_stocfor1", "lp_25fv47"] # TODO, "lpi_chemcom"
     success_problem = []
     failed_problem = []
     diff_reference = Dict()
     timing_list = Dict()
     for p_str in problem_sets
        println("Current problem " * p_str)
        problem_name = problem_folder * p_str
        problem = create_problem(problem_name)
        
        start = time_ns()
        solution = HPCGLpSolver.iplp(problem, 1e-4; max_iterations=1000)
        elapsed = time_ns() - start
        timing_list[p_str] = elapsed
        
        # compute differnece w.r.t reference optimal value for debugging
        diff_reference[p_str] = problem.c' * solution.x - opt_dict[p_str]
        if solution.flag # TODO Tolerance with reference optimal value &&  (problem.c' * solution.x - opt_dict[p_str]) < 1e-1
            push!(success_problem, p_str)
        else
            push!(failed_problem, p_str)
        end
     end

    println("************************* Final Test Results *************************")
    println("Sucess problems: ")
    println(success_problem)
    println("Failed problems: ")
    println(failed_problem)
    println("Difference w.r.t reference: ")
    for (key, value) in diff_reference
        @printf("Problem: %10s  Diff: %f Time: %10f \n", key, value, timing_list[key] * 1e-9)        
    end
    
end

test_problems()