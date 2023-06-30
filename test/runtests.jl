using ReinforcementLearningTrajectories
using CircularArrayBuffers
using Test
using CUDA
using Adapt

struct TestAdaptor end

gpu(x) = Adapt.adapt(TestAdaptor(), x)

Adapt.adapt_storage(to::TestAdaptor, x) = CUDA.functional() ? CUDA.cu(x) : x

@testset "ReinforcementLearningTrajectories.jl" begin
    include("traces.jl")
    include("common.jl")
    include("samplers.jl")
    include("controllers.jl")
    include("trajectories.jl")
    include("normalization.jl")
    include("episodes.jl")
end
