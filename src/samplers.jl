export BatchSampler

using Random

struct BatchSampler
    batch_size::Int
    rng::Random.AbstractRNG
end

"""
    BatchSampler(batch_size; rng=Random.GLOBAL_RNG)

Uniformly sample a batch of examples for each trace.

See also [`sample`](@ref).
"""
BatchSampler(batch_size; rng=Random.GLOBAL_RNG) = BatchSampler(batch_size, rng)
