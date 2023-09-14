using ReinforcementLearningTrajectories
using CircularArrayBuffers, DataStructures
using StableRNGs
using Test
import ReinforcementLearningTrajectories.StatsBase.sample
using CUDA
using Adapt
using Random
import ReinforcementLearningTrajectories.StatsBase.sample
import StatsBase.countmap


struct TestAdaptor end

gpu(x) = Adapt.adapt(TestAdaptor(), x)

Adapt.adapt_storage(to::TestAdaptor, x) = CUDA.functional() ? CUDA.cu(x) : x

@testset "ReinforcementLearningTrajectories.jl" begin
    include("traces.jl")
    include("sum_tree.jl")
    include("common.jl")
    include("samplers.jl")
    include("controllers.jl")
    include("trajectories.jl")
    include("normalization.jl")
    include("episodes.jl")
end
