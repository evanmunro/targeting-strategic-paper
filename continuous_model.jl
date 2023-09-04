using Random, Distributions, Optim, Plots, Statistics
using BayesianOptimization, GaussianProcesses

struct Theta
    θ::Bool
    z::Float64
    c::Float64
    y1::Float64
    y0::Float64
end

struct SampleTypes
    sample::Array{Theta}
end

function SampleTypes(n)
    z= rand(Normal(0,1), n)
    c = rand(Uniform(1, 10),n)
    θ = rand(Bernoulli(0.5), n)
    z0 = rand(Normal(-1, 2), n)
    z1 = rand(Normal(1, 2), n)
    z = z0.*(1 .- θ)  .+ z1 .*θ
    y1 = 5 .*(0.75.*θ .+ (1 .- θ))
    y0 = (1 .- θ).*10
    return [Theta(θ[i], z[i], c[i], y1[i], y0[i]) for i in 1:n]
end

function generateCouponPlots()
        n = 100000
        agents = SampleTypes(n)
        result = computeOptimalBeta(agents, false)
        βces = result.observed_optimizer
        println(result.observed_optimum)
        result = computeOptimalBeta(agents, true)
        βstar = result.observed_optimizer
        println(result.observed_optimum)
        n = 10000
        agents = SampleTypes(n)
        plotNonStrategic(agents, βces, "Non-Strategic")
        savefig("nonstrategic.pdf")
        plotStrategic(agents, βstar, "Strategic Behavior with Optimal Rule")
        savefig("strategic_optimal.pdf")
        plotStrategic(agents, βces, "Strategic Behavior with Cutoff Rule")
        savefig("strategic_ces.pdf")
end

function plotStrategic(data, β, title)
    n = length(data)
    S = 500
    x = [reportX(data[i], β, true) for i in 1:n]
    δ = [allocation(x[i], β) for i in 1:n]
    w = [rand(Bernoulli(δ[i]), 1)[1] for i in 1:n]
    y = [data[i].y1*w[i] + (1-w[i])*data[i].y0 for i in 1:n]
    sub = 1:S
    println("strategic outcomes: ", mean(y))
    cate = [(data[i].y1 - data[i].y0) >0  for i in 1:n ]
    #xsub = x[cate.<1]
    plot(x[sub], zeros(S) .+ rand(Normal(-0.2,0.05), S), markershape = [:circle :utriangle], group=cate[sub], label = ["Negative ITE" "Positive ITE"],
                    xrange = [-3, 3], yrange = [-0.3, 1.4], alpha=0.5, seriestype=:scatter)
    alloc(x) = allocation(x, β)
    plot!(alloc, -3, 3, lw = 2, label = "Treatment Prob.", linecolor=:black, xlabel="x", ylabel="π(x)")
end

function plotNonStrategic(data, β, title)
    n = length(data)
    S = 500
    sub = 1:S
    x = [data[i].z for i in 1:n]
    δ = [allocation(x[i], β) for i in 1:n]
    w = [rand(Bernoulli(δ[i]), 1)[1] for i in 1:n]
    y = [data[i].y1*w[i] + (1-w[i])*data[i].y0 for i in 1:n]
    println("non-strategic outcomes: ", mean(y))
    cate = [(data[i].y1 - data[i].y0) >0  for i in 1:n ]
    plot(x[sub], zeros(S) .+ rand(Normal(-0.2,0.05), S), markershape = [:circle :utriangle], group=cate[sub], label = ["Negative ITE" "Positive ITE"],
                    xrange = [-3, 3], yrange = [-0.3, 1.4], alpha=0.5, seriestype=:scatter)
    alloc(x) = allocation(x, β)
    plot!(alloc, -3, 3, lw = 2, label = "Treatment Prob.", linecolor=:black, xlabel="x", ylabel="π(x)")
end


function computeOptimalBeta(agents, strategic)

    function objective(β)
        data = generateObservations(agents, β, strategic)
        return mean(data.y)
    end

    data = generateObservations(agents, [0, 0], strategic)
    println(length(data.y))
    μ = mean(data.y)
    σ2 = var(data.y)/length(data.y)
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
           [-50., -50.], [50., 50.],                     # lowerbounds, upperbounds
           repetitions = 1,                          # evaluate the function for each input 5 times
           maxiterations = 500,                      # evaluate at 100 input positions
           sense = Max,                              # minimize the function
           #acquisitionoptions = (method = :LD_LBFGS, # run optimization of acquisition function with NLopts :LD_LBFGS method
            #                     restarts = 5,       # run the NLopt method from 5 random initial conditions each time.
            #                     maxtime = 0.1,      # run the NLopt method for at most 0.1 second each time
            #                     maxeval = 1000),    # run the NLopt methods for at most 1000 iterations (for other options see https://github.com/JuliaOpt/NLopt.jl)
            verbosity = Progress)

    return boptimize!(opt)
end

function generateObservations(agents, β, strategic::Bool)
    n = length(agents)
    x = [reportX(agents[i], β, strategic) for i in 1:n]
    δ = [allocation(x[i], β) for i in 1:n]
    w = [rand(Bernoulli(δ[i]), 1)[1] for i in 1:n]
    y = [agents[i].y1*w[i] + (1-w[i])*agents[i].y0 for i in 1:n]
    return (y=y, w=w, x=x)
end

function reportX(type::Theta, β, strategic)
    if strategic
        if type.θ == 1
            return type.z
        else
            function objective(x)
                return -5*allocation(x, β) + type.c*(x - type.z)^2
            end
            res = Optim.optimize(x -> objective(x), -100, 100, GoldenSection())
            return Optim.minimizer(res)
        end
    else
        return type.z
    end
end

function allocation(x, β)
    return 1 ./ (1 .+ exp.(-1 .* (x.*β[2] + β[1])))
end
