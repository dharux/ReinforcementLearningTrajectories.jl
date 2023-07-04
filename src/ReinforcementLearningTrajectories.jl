module ReinforcementLearningTrajectories

const RLTrajectories = ReinforcementLearningTrajectories
export RLTrajectories
import StatsBase

include("patch.jl")
include("traces.jl")
include("common/common.jl")
include("episodes.jl")
include("samplers.jl")
include("controllers.jl")
include("trajectory.jl")
include("normalization.jl")
end
