using POMDPs, Pkg
using Distributions
using Parameters
using POMDPModelTools
using Random
using PyPlot
pygui(:qt5)

struct SleepState3
    complete::Int32
    productivity::Int32
    deadline::Int32
    hours_slept::Int32
    hours_since_slept::Int32
end

@with_kw mutable struct SleepPOMDP3 <: POMDP{SleepState3, Symbol, Float64}
    r_complete::Int32 = 10000
    r_missed::Int32 = -100
    sleep_multiplier::Float64 = 1.5
    work_multiplier::Float64 = 0.8
    relax_multiplier::Float64 = 1.2
    max_deadline::Int32 = 72
    discount::Float64 = 1.0
    obs_sigma::Float64 = 10
end

function sleep_plot(history, path)
    complete = []
    productivity = []
    deadline = []
    observation = []
    actions = []
    i = 0
    for (s, a, o, r) in eachstep(history, "s,a,o,r")
        push!(complete, s.complete)
        push!(productivity, s.productivity)
        push!(deadline, s.deadline)
        push!(observation, o)
        push!(actions, a)
    end
    figure(figsize=(8, 8))
    plot(collect(1:length(complete)), complete, color="green")
    plot(collect(1:length(productivity)), productivity, color="orange")
    plot(collect(1:length(deadline)), deadline, color="red")
    prev = 1
    curr = 1
    curr_seg_action = ""
    # actions = fill(:work, length(actions))
    push!(actions, :end)
    linewidth = 7.0
    for action in actions
        if action != curr_seg_action
            if curr_seg_action == :work
                axvspan(prev, curr, 0, 100, color="black", alpha=.2)
            elseif curr_seg_action == :relax
                axvspan(prev, curr, 0, 100, color="pink", alpha=.2)
            else
                axvspan(prev, curr, 0, 100, color="blue", alpha=.2)
            end
            prev = curr
            curr_seg_action = action
        end
        curr += 1
    end
    savefig(path)
end


function POMDPs.actions(m::SleepPOMDP3)
    return [:work, :sleep, :relax]
end

function POMDPs.actionindex(m::SleepPOMDP3, a::Symbol)
    if a == :work
        return 1
    elseif a == :sleep
        return 2
    elseif a == :relax
        return 3
    end
    error("Invalid Action $a")
end

function POMDPs.reward(m::SleepPOMDP3, s::SleepState3, a::Symbol)
    if s.deadline == 0 && s.complete < 100
        return -100
    elseif s.complete >= 100
        return 100
    elseif a == :relax && s.hours_since_slept < 12
        return 1
    else
        return 0
    end
end

function POMDPs.gen(m::SleepPOMDP3, s::SleepState3, a::Symbol, rng::AbstractRNG)
    deadline = s.deadline - 1
    complete = s.complete
    productivity = s.productivity
    hours_since_slept = s.hours_since_slept + 1
    hours_slept = 0
    if s.deadline == 0 || s.complete >= 100
        complete = 0
        deadline = round(rand(rng, Truncated(Normal(m.max_deadline, 5), 0.0, m.max_deadline)))
    end

    if a == :work
        if s.complete < 100 && s.deadline > 0
            complete = round(s.complete + rand(rng, Truncated(Normal(10 * (s.productivity / 100), 1), 0.0, Inf)))
        end
        productivity = round(s.productivity * rand(rng, Normal(m.work_multiplier, 0.01)))
    elseif a == :sleep
        if s.hours_slept >= 6
            productivity = min(round(s.productivity * rand(rng, Normal(m.sleep_multiplier, 0.01))), 100)
            hours_since_slept = 0
        end
        hours_slept = s.hours_slept + 1
    else
        if s.hours_since_slept < 12
            productivity = min(round(s.productivity * rand(rng, Normal(m.relax_multiplier, 0.01))), 100)
        end
    end
    return (sp=SleepState3(complete, productivity, deadline, hours_slept, hours_since_slept),)
end

function POMDPs.observation(m::SleepPOMDP3, a::Symbol, sp::SleepState3)
    return Truncated(Normal(sp.complete, m.obs_sigma), 0.0, Inf)
end

function POMDPs.discount(m::SleepPOMDP3)
    return 0.95
end

POMDPs.initialstate_distribution(m::SleepPOMDP3) = Deterministic(SleepState3(0, 100, m.max_deadline, 0, 0))


using ParticleFilters
using BasicPOMCP
using ARDESPOT
problem = SleepPOMDP3()
solver = DESPOTSolver(bounds=(DefaultPolicyLB(RandomSolver()), 2000))
planner = solve(solver, problem)
filter = SIRParticleFilter(problem, 1000)
hr = HistoryRecorder(max_steps=15*12)
history = simulate(hr, problem, planner, filter)
discounted_reward(history)

sleep_plot(history, "optimal_impossible.png")
