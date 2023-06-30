module ReinforcementLearningTrajectories

const RLTrajectories = ReinforcementLearningTrajectories
export RLTrajectories

include("patch.jl")
include("traces.jl")
include("common/common.jl")
include("samplers.jl")
include("controllers.jl")
include("trajectory.jl")
include("normalization.jl")
include("episodes.jl")
end
