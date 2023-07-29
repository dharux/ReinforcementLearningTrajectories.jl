@testset "BatchSampler" begin
    sz = 32
    s = BatchSampler(sz)
    t = Traces(
        state=rand(3, 4, 5),
        action=rand(1:4, 5),
    )

    b = sample(s, t)

    @test keys(b) == (:state, :action)
    @test size(b.state) == (3, 4, sz)
    @test size(b.action) == (sz,)
    
    #In EpisodesBuffer
    eb = EpisodesBuffer(CircularArraySARTSTraces(capacity=10)) 
    push!(eb, (state = 1, action = 1))
    for i = 1:5
        push!(eb, (state = i+1, action =i+1, reward = i, terminal = false))
    end
    push!(eb, (state = 7, action = 7))
    for (j,i) = enumerate(8:11)
        push!(eb, (state = i, action =i, reward = i-1, terminal = false))
    end
    s = BatchSampler(1000)
    b = sample(s, eb)
    cm = counter(b[:state])
    @test !haskey(cm, 6)
    @test !haskey(cm, 11)
    @test all(in(keys(cm)), [1:5;7:10])
end

@testset "MetaSampler" begin
    t = Trajectory(
        container=Traces(
            a=Int[],
            b=Bool[]
        ),
        sampler=MetaSampler(policy=BatchSampler(3), critic=BatchSampler(5)),
    )
    push!(t, (a = 1,))
    for i in 1:10
        push!(t, (a=i, b=true))
    end

    batches = collect(t)

    @test length(batches) == 11
    @test length(batches[1][:policy][:a]) == 3 && length(batches[1][:critic][:b]) == 5
end

@testset "MultiBatchSampler" begin
    t = Trajectory(
        container=Traces(
            a=Int[],
            b=Bool[]
        ),
        sampler=MetaSampler(policy=BatchSampler(3), critic=MultiBatchSampler(BatchSampler(5), 2)),
    )

    push!(t, (a = 1,))
    for i in 1:10
        push!(t, (a=i, b=true))
    end

    batches = collect(t)

    @test length(batches) == 11
    @test length(batches[1][:policy][:a]) == 3
    @test length(batches[1][:critic]) == 2 # we sampled 2 batches for critic
    @test length(batches[1][:critic][1][:b]) == 5 #each batch is 5 samples 
end

#! format: off
@testset "NStepSampler" begin
    γ = 0.9
    n_stack = 2
    n_horizon = 3
    batch_size = 4

    t1 = MultiplexTraces{(:state, :next_state)}(1:10) +
        MultiplexTraces{(:action, :next_action)}(iseven.(1:10)) +
        Traces(
            reward=1:9,
            terminal=Bool[0, 0, 0, 1, 0, 0, 0, 0, 1],
        )

    s1 = NStepBatchSampler(n=n_horizon, γ=γ, stack_size=n_stack, batch_size=batch_size)

    xs = RLTrajectories.StatsBase.sample(s1, t1)

    @test size(xs.state) == (n_stack, batch_size)
    @test size(xs.next_state) == (n_stack, batch_size)
    @test size(xs.action) == (batch_size,)
    @test size(xs.reward) == (batch_size,)
    @test size(xs.terminal) == (batch_size,)

    
    state_size = (2,3)
    n_state = reduce(*, state_size)
    total_length = 10
    t2 = MultiplexTraces{(:state, :next_state)}(
            reshape(1:n_state * total_length, state_size..., total_length)
        ) +
        MultiplexTraces{(:action, :next_action)}(iseven.(1:total_length)) +
        Traces(
            reward=1:total_length-1,
            terminal=Bool[0, 0, 0, 1, 0, 0, 0, 0, 1],
        )

    xs2 = RLTrajectories.StatsBase.sample(s1, t2)

    @test size(xs2.state) == (state_size..., n_stack, batch_size)
    @test size(xs2.next_state) == (state_size..., n_stack, batch_size)
    @test size(xs2.action) == (batch_size,)
    @test size(xs2.reward) == (batch_size,)
    @test size(xs2.terminal) == (batch_size,)

    inds = [3, 5, 7]
    xs3 = RLTrajectories.StatsBase.sample(s1, t2, Val(SS′ART), inds)

    @test xs3.state == cat(
        (
            reshape(n_state * (i-n_stack)+1: n_state * i, state_size..., n_stack)
            for i in inds
        )...
        ;dims=length(state_size) + 2
    ) 

    @test xs3.next_state == xs3.state .+ (n_state * n_horizon)
    @test xs3.action == iseven.(inds)
    @test xs3.terminal == [any(t2[:terminal][i: i+n_horizon-1]) for i in inds]

    # manual calculation
    @test xs3.reward[1] ≈ 3 + γ * 4  # terminated at step 4
    @test xs3.reward[2] ≈ 5 + γ * (6 + γ * 7)
    @test xs3.reward[3] ≈ 7 + γ * (8 + γ * 9)
end
#! format: on

@testset "Trajectory with CircularPrioritizedTraces and NStepBatchSampler" begin
    n=1
    γ=0.99f0

    t = Trajectory(
        container=CircularPrioritizedTraces(
            CircularArraySARTSTraces(
                capacity=5,
                state=Float32 => (4,),
            );
            default_priority=100.0f0
        ),
        sampler=NStepBatchSampler{SS′ART}(
            n=n,
            γ=γ,
            batch_size=32,
        ),
        controller=InsertSampleRatioController(
            threshold=100,
            n_inserted=-1
        )
    )

    push!(t, (state = 1, action = true))
    for i = 1:9
        push!(t, (state = i+1, action = true, reward = i, terminal = false))
    end

    b = RLTrajectories.StatsBase.sample(t)
    @test haskey(b, :priority)
end


@testset "Trajectory with CircularArraySARTSTraces and NStepBatchSampler" begin
    n=1
    γ=0.99f0

    t = Trajectory(
        container=CircularArraySARTSTraces(
                capacity=5,
                state=Float32 => (4,),
        ),
        sampler=NStepBatchSampler{SS′ART}(
            n=n,
            γ=γ,
            batch_size=32,
        ),
        controller=InsertSampleRatioController(
            threshold=100,
            n_inserted=-1
        )
    )

    push!(t, (state = 1, action = true))
    for i = 1:9
        push!(t, (state = i+1, action = true, reward = i, terminal = false))
    end

    b = RLTrajectories.StatsBase.sample(t)
    @test haskey(b, :priority)
end
