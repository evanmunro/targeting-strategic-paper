using Random, Distributions, Optim, Plots, Statistics
using BayesianOptimization, GaussianProcesses

struct PrefType
    θ::Bool
    M::Bool
    c::Float64
    y1::Float64
    y0::Float64
end

function PrefTypes(n)
    γ = 0.46
    ρ₀ = 0.71
    ρ₁ = 0.44
    b₀ = 4.13
    b₁ = 3.30
    θ = rand(Bernoulli(γ), n)
    M1 = rand(Bernoulli(ρ₁), n)
    M0 = rand(Bernoulli(ρ₀), n)
    c1 = rand(Uniform(0, b₁), n)
    c0 = rand(Uniform(0, b₀), n)
    C = zeros(n)
    M = zeros(n)
    for i in 1:n
        if θ[i] == 1
            M[i] = M1[i]
            C[i] = c1[i]
        else
            M[i] = M0[i]
            C[i] = c0[i]
        end
    end
    y1 = 5 .* θ .- 6 .* (1 .- θ)
    y0 = zeros(n)
    return [PrefType(θ[i], M[i], C[i], y1[i], y0[i]) for i in 1:n]
end

function reportX(type::PrefType, δ, strategic)
    if type.M == 1
        return (δ[2] - δ[1])>type.c
    else
        return 1
    end
end


function generateObservations(agents, δ, strategic)
    n = length(agents)
    x = [reportX(agents[i], δ, strategic) for i in 1:n]
    δs = [δ[x[i]+1] for i in 1:n]
    w = [rand(Bernoulli(δs[i]), 1)[1] for i in 1:n]
    y = [agents[i].y1*w[i] + (1-w[i])*agents[i].y0 for i in 1:n]
    return (y=y, w=w, x=x)
end

function regret_simulation(agentGen)

    strategic=true
    result = computeOptimalPi(strategic, 2000, agentGen)
    println(result.dstar)
    cum = accumulate(+, result.regret)./(1:length(result.regret))
    #repeat the regret calculation 50 times and average over the samples to make plot
    for s in 1:50
        result = computeOptimalPi(strategic, 2000, agentGen)
        println(result.dstar)
        cum .+= accumulate(+, result.regret)./(1:length(result.regret))
    end
    cum ./= 51
    plot(cum[5:length(cum)], xlabel="Period", ylabel="Average Regret", legend=nothing)
    savefig("regret.pdf")
end

#compute optimal targeting rule on a sample of agents
function lineSearch(n, agentGen)
    agents = agentGen(n)
    function objective(π)
        data = generateObservations(agents, [0.0, π], true)
        return -1 * mean(data.y)
    end

    res = Optim.optimize(x -> objective(x), 0, 1, GoldenSection())
    return Optim.minimizer(res)
end

function computeOptimalPi(strategic, n, agentGen)
    opt_for_regret = []
    actual_opt = []
    #get the best policy on a large sample
    pistar = lineSearch(Int(1e6), agentGen)
    println("Optimal: ", pistar)
    function objective(β)
        agents = agentGen(n)
        data = generateObservations(agents, β, strategic)

        #compare to optimal value computed in advance
        dataopt = generateObservations(agents, [0.0, pistar], strategic)
        push!(opt_for_regret, mean(dataopt.y))
        push!(actual_opt, mean(data.y))
        return mean(data.y)
    end
    agents = agentGen(n)
    data = generateObservations(agents, [0.5, 0.5], strategic)
    println(length(data.y))
    μ = mean(data.y)
    σ2 = var(data.y)/length(data.y)
    model = ElasticGPE(2,                            # 2 input dimensions
                   mean = MeanConst(μ),
                   kernel = SEArd([0., 0.], 1.0),
                   logNoise = log(sqrt(σ2)*2),
                   capacity = 3000)              # the initial capacity of the GP is 3000 samples.
    set_priors!(model.mean, [Normal(μ, 1)])

    modeloptimizer = NoModelOptimizer()

    opt = BOpt(objective,
           model,
           UpperConfidenceBound(NoBetaScaling(), 1.0),                   # type of acquisition
           modeloptimizer,
           [0.0, 0.5], [0.5, 1.0],                     # lowerbounds, upperbounds
           repetitions = 1,                          # evaluate the function for each input 1 time
           maxiterations = 500,                      # evaluate at 500 input positions
           sense = Max,                              # max the function
           acquisitionoptions = (method = :LD_LBFGS, # run optimization of acquisition function with NLopts :LD_LBFGS method
                                 restarts = 5,       # run the NLopt method from 5 random initial conditions each time.
                                 maxtime = 0.1,      # run the NLopt method for at most 0.1 second each time
                                 maxeval = 1000),    # run the NLopt methods for at most 1000 iterations (for other options see https://github.com/JuliaOpt/NLopt.jl)
            verbosity = Progress)

    result = boptimize!(opt)
    #print(fieldnames(typeof(opt)))
    return (dstar = result.observed_optimizer, value = result.observed_optimum,
             regret=opt_for_regret .- actual_opt)
end
