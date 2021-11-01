using DrWatson, Flux, CSV, DataFrames, MLDataPattern, StatsBase
@quickactivate "Swarm"

include(srcdir("data.jl"))

dataframe = DataFrame(CSV.File("data/exp_raw/data2.csv"))

# 2 boids
# 3 dimentions (x, y ,z)
# 610 timesteps
# 10 batches
# 2 x 3 x 60 x 10
mat = Matrix(dataframe)

n = axes(mat)[1]
d = axes(mat)[2]

D = mat

w = 60
s = 1

function newNorm(M) 
    max = maximum(M)
    min = minimum(M)
    return (M .- min) ./ (max - min)
end

norm_d = newNorm(D)

Z = slidingwindow(norm_d', w, stride=s)


axes(Z[1])

# correct = reshape(mat, (610, 3,2))

# normalized_data = unitNormalise(correct)

 model = Chain(
         # input is meant to be 60x6
         Conv((10,10), 6 => 64, pad = 1, relu),
         MaxPool((2,2)),

         Conv((5,5), 30 => 32, pad = 1, relu),
         MaxPool((2,2)),

         Conv((5,5),  15 => 12, pad = 1, relu),
         MaxPool((3,3)),

         Flux.flatten,
         Dense(60, 60),

         Conv((5,5),  5 => 12, pad = 1, relu),
         Upsample(size = (3,3)),

         Conv((5,5),  15 => 32, pad = 1, relu),
         Upsample(size = (2,2)),

         Conv((10,10),  60 => 64, pad = 1, relu),
         Upsample(size = (2,2))
 )


 loss(x) = Flux.Losses.mse(model(x),x)

obs = map(getobs,Z)

# loss(getobs(Z[1]))
# Flux.normalize(correct

# normalized_data
epochs = 10

for epoch = 1:epochs
  for z in Z 
    gs = gradient(params(m)) do
      l = loss(z)
    end
    update!(opt, params(m), gs)
  end
end
