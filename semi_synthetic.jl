using CategoricalArrays, DataFrames, GLM, CSV, Statistics, Random

function recode_raw_survey_data!(df)
    #Q1, Q2, Q3, Q4 are labeled Q10, Q12, Q3, and Q11 in the Qualtrics file
    recode!(df.Q3, 1=> 10.0, 2=> 12.0, 3=> 13.0, 4=> 14.0, 5=> 16.0, 6=> 18.0, 7=> 20.0)
end

function clean_data(datapath)
    data = DataFrame(CSV.File(datapath))
    recode_raw_survey_data!(data)
    recode!(data.Q11, missing => -100)
    data.correct = (data.Q11 .≈ 3) .+ (data.Q11 .≈ -0.5)
    data.lmath = (data.Q10 .> 3)
    ites = [-6, 5]
    data.ites = [ites[Int(i)+1] for i in data.lmath]
    println(nrow(data))
    return data
end

function data_summary(data)
    lmath = mean(data.lmath)
    correct  = mean(data.correct)
    correct_lm = mean(data.correct.*data.lmath)/mean(data.lmath)
    correct_dl = mean(data.correct .*(1 .- data.lmath))/mean((1 .- data.lmath))
    return (lmath=lmath, correct=correct, correct_lm=correct_lm, correct_dl=correct_dl)
end
function compute_outcomes(data1, data2)
#compute each of the relevant outcomes as vector
    data1_summary = data_summary(data1)
    data2_summary = data_summary(data2)
    gamma = (mean(data1.lmath) + mean(data2.lmath))/2
    rho0 = 1 - data1_summary.correct_dl #proportion who respond B and don't like math
    rho1 = 1 - data1_summary.correct_lm #proportion who respond B and like math
    b0 = max(0, 1/(data2_summary.correct_dl - data1_summary.correct_dl))*rho0
    b1 = max(0, 1/(data2_summary.correct_lm - data1_summary.correct_lm))*rho1
    #println(b0)
    #println(b1)
    betahat = (gamma, rho0, rho1, b0, b1)
    return (data1_summary, data2_summary, betahat)
end

function bootstrap_ses(;S=1000)
    Random.seed!(1234)
    data1 = clean_data("data/full_noincentive_external.csv")
    data2 = clean_data("data/full_bonus100_external.csv")
    wave1col = zeros(4, S)
    wave2col= zeros(4, S)
    estparams = zeros(5, S)

    rows = collect(1:nrow(data1))
    for s in 1:S
        r1 = rand(rows, length(rows))
        r2 = rand(rows, length(rows))
        d1 = data1[r1, :]
        d2 = data2[r2, :]
        d1s, d2s, betahat = compute_outcomes(d1, d2)
        wave1col[:, s] .= collect(d1s)
        wave2col[:, s] .= collect(d2s)
        estparams[:, s] .= collect(betahat)
    end

    println("Wave 1 Table 2: ", mean(wave1col, dims=2), " ", std(wave1col, dims=2))
    println("Wave 2 Table 2: ", mean(wave2col, dims=2), " ", std(wave2col, dims=2))
    println("Table 3: ", mean(estparams, dims=2), " ", std(estparams, dims=2))
end
