using Random, Distributions, Optim, Plots, Statistics
using BayesianOptimization, GaussianProcesses, StatsBase

struct Agent
    θ::Int64
    τ::Float64
    cA::Float64
    y0::Float64
    V::Float64
end

function Agent(ps)
    θ = sample([1, 2, 3], Weights(ps.ρ))
    #rand([1, 2, 3], 1)[1]
    if θ == 1
        cA = 2.0
    elseif θ == 2
        cA = rand(Uniform(ps.Lc, ps.Uc), 1)[1]
    else
        cA = -2.0
    end

    e0, e1 = rand(Uniform(-1, 1), 2)
    return Agent(θ, ps.τ[θ] + e1, cA, ps.y0[θ] + e0, ps.V[θ])
end


function dro_bs_example()
    ps = Parameters([0.0, 1/2, 1/2], [0.0, -5.0, 3.75],[0.0, 10.0, 0.0], 10.0, 0.0, [0.0, 5.0, 3.75] )
    compute_BS_optimal_rule(ps)

end

function cvar(data, s)
    q = quantile(data, s)
    return mean(data[data .> q])

end
function compute_BS_optimal_rule(ps; n=1000000)
    agents = generateSample(n, ps)
    Xs = X(agents, [0.5, 0.5])
    agentsA = agents[Xs .== 1]
    agentsB = agents[Xs .== 0]
    cateA = mean([a.τ for a in agentsA])
    cateB = mean([a.τ for a in agentsB])
    y1A = [a.y0 + a.τ for a in agentsA]
    y1B = [a.y0 + a.τ for a in agentsB]
    y0A = [a.y0  for a in agentsA]
    y0B = [a.y0 for a in agentsB]

    for γ in [1,  1.2, 1.5, 2.0, 3.0, 4.0, 100.0]
        s= 1/(1+ γ)
        HA = (1 - 1/γ)*(cvar(y1A, s) - cvar(y0A, s))
        HB = (1 - 1/γ)*(cvar(y1B, s) - cvar(y0B, s))
        println("Γ = ", γ, ": (CATE(A): ", cateA, " H(A):", HA, ")")
        println("(CATE(B): ", cateB, " H(B):", HB, ")")
        println("Verified CATE(A) > H(A) and CATE(B) < H(B)")
    end

end
struct Parameters
    ρ::Vector
    τ::Vector
    y0::Vector
    Uc::Float64
    Lc::Float64
    V::Vector
end

function X(agents::Vector, π)
    return [X(a, π) for a in agents]
end
function X(agent::Agent, π)::Int64
    if agent.V * (π[2] - π[1]) > agent.cA
        return 1
    else
        return 0
    end
end

function Y(agent, W)::Float64
    if W==1
        return agent.τ .+ agent.y0
    else
        return agent.y0
    end
end

function Parameters(M)
    return Parameters([1/3, 1/3, 1/3], [-5.0, M, 4.0], [0.0, 0.0, 0.0], 2.0, 0.0, [1.0, 1.0, 1.0])
end

function generateSample(n, ps::Parameters)
    return [Agent(ps) for i in 1:n]
end

function Example1()
    println("Solving the Coupon Model for Table 1: ")
    ps = Parameters([0.0, 1/2, 1/2], [0.0, -5.0, 3.75],[0.0, 10.0, 0.0], 10.0, 0.0, [0.0, 5.0, 3.75] )
    computeNumericOptimal(ps; verbose=true)
end

function Example2()
    println("Solving the Product Upgrade Model for Table 2: ")
    ps = Parameters([1/2, 1/2, 0.0], [-4.0, 5.0, 0.0],[5.0, 5.0, 0.0], 10.0, -10.0, [0.0, 5.0, 0.0] )
    computeNumericOptimal(ps; verbose=true)
end
function computeNumericOptimal(ps::Parameters; verbose = false)
    n=1000000
    agents = generateSample(n, ps)
    τs = [a.τ for a in agents]
    function objective(π)
        signals = X(agents, π)
        πs = π[2] .* signals .+ π[1] .* ( 1 .- signals)
        EY = mean(πs .* τs)
        return EY
    end

    function utilities(π)
        signals = X(agents, π)
        output = zeros(length(signals))
        πs = π[2] .* signals .+ π[1] .* ( 1 .- signals)
        for i in 1:length(signals)
            cost = (signals[i]==1)*(agents[i].θ ==2)*(agents[i].cA) *πs[i]
            value = agents[i].V * πs[i]
            output[i] = value - cost
        end
    #    println(output[1:5])
        return mean(output)
    end

    signals = X(agents, [0.5, 0.5])
    πs = 0.5 .* signals .+ 0.5 .* ( 1 .- signals)
    μ = mean(πs .* τs)
    σ2 = var(πs .* τs)/n
    model = ElasticGPE(2,                            # 2 input dimensions
                   mean = MeanConst(0.),
                   kernel = SE([0., 0.], 5.),
                   logNoise = log(σ2),
                   capacity = 3000)              # the initial capacity of the GP is 3000 samples.
    set_priors!(model.mean, [Normal(μ, 2)])

    # Optimize the hyperparameters of the GP using maximum a posteriori (MAP) estimates every 50 steps
    modeloptimizer = NoModelOptimizer()

    opt = BOpt(objective,
           model,
           UpperConfidenceBound(),                   # type of acquisition
           modeloptimizer,
           [0.0, 0.0], [1.0, 1.0],                     # lowerbounds, upperbounds
           repetitions = 1,                          # evaluate the function for each input 5 times
           maxiterations = 30,                      # evaluate at 100 input positions
           sense = Max,                              # minimize the function
           #acquisitionoptions = (method = :LD_LBFGS, # run optimization of acquisition function with NLopts :LD_LBFGS method
            #                     restarts = 5,       # run the NLopt method from 5 random initial conditions each time.
            #                     maxtime = 0.1,      # run the NLopt method for at most 0.1 second each time
            #                     maxeval = 1000),    # run the NLopt methods for at most 1000 iterations (for other options see https://github.com/JuliaOpt/NLopt.jl)
            verbosity = Progress)

    results = boptimize!(opt)
    pistar = results.model_optimizer
    if verbose
        println("Mean Utility with Cutoff: ", utilities([0.0, 1.0]))
        println("Mean Utility with Uniform: ", utilities([0.5, 0.5]))
        println("Mean Utility with Stackelberg: ", utilities(pistar))
        println("Mean Objective with Cutoff: ", objective([0.0, 1.0]) .+ mean([a.y0 for a in agents]))
        println("Mean Objective with Uniform: ", objective([0.5, 0.5]) .+ mean([a.y0 for a in agents]))
        println("Mean Objective with Stackelberg: ", objective(pistar) .+ mean([a.y0 for a in agents]))
    end
    println("Optimal Stackelberg Rule: ", results.model_optimizer)
    return results.model_optimizer
end

function createCutoffFigure()
    Ms = collect(-50:1:15.0)
    πA = zeros(length(Ms))
    πB = zeros(length(Ms))
    for i in 1:length(Ms)
        M = Ms[i]
        πA[i] = computeNumericOptimal(Parameters(M))[2]
    end
    plot(Ms, [πA πB], xlabel="τₘ", ylabel = "Optimal π", linestyle = [:solid :dash], linewidth=5, label=[ "π[A]" "π[B]"])
    savefig("figure1.pdf")
end
