using Revise
includet("cutoff_figure.jl")
includet("semi_synthetic.jl")
includet("continuous_model.jl")
includet("regret.jl")

#Figure 1
Random.seed!(1)
createCutoffFigure()

#Figure 2
Random.seed!(1)
generateCouponPlots()

#Figure 3
Random.seed!(1)
regret_simulation(PrefTypes)

#Table 3 and Table 4
Random.seed!(1)
bootstrap_ses()


#Numerical Computations Checking Table 1 and Table 2
Random.seed!(1)
Example1()
Example2()
dro_bs_example()
