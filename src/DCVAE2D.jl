##
# A 2D convolution approach - I've done this before and lost the code I think :(
#
#
using DrWatson
quickactivate(@__DIR__)
using Flux.Data: DataLoader
using Flux, DataFrames, StatsBase, MLDataPattern, CUDA, PlotlyJS, LegolasFlux, CSV
using Wandb, Dates, Logging

include("training_utils.jl")

# Start a new run, tracking hyperparameters in config
# logger = WandbLogger(
#     project="Swarm-VAE",
#     name="swarm-vae-training-$(now())",
#     config=Dict(
#         "η" => 0.00001,
#         "batch_size" => 48,
#         "data_set" => "data_test"
#     ),
# )

# global_logger(logger)



function create_vae()

    #instead of 60x300x48 we will have 60x3x100x48
  
    encoder_features = Chain(

        #60x3x100xb
        Conv((10,3), 100 => 1000, relu; pad = SamePad()),
        #60x3x1000xb
        Conv((10,3), 1000 => 750, relu; pad = SamePad(), stride=(2,1)),
        #30x3x750xb
        Conv((10,3), 750 => 500, relu; pad = SamePad(), stride=(2,1)),
        #15x3x500xb
        Conv((5,1), 500 => 250, relu; pad = SamePad(), stride=(3,1)),
        #5x3x250xb
        Conv((5,1), 250 => 100, relu; pad = SamePad()),
        #5x3x100xb
        Conv((5,1), 100 => 10, relu; pad = SamePad()),
        #5x3x10xb

        Flux.flatten,
        Dense(150,100,relu),
        Dense(100,10,relu)
      )
      
    encoder_μ      = Chain(encoder_features, Dense(10,10)) 
      
    encoder_logvar = Chain(encoder_features, Dense(10,10)) 
    
    decoder = Chain(
        Dense(10,100,relu),
        Dense(100,150,relu),

        (x -> reshape(x, 5, 3,10,:)),
        #5x3x10xb
        ConvTranspose((5,1), 10 => 100, relu; pad = SamePad()),
        #5x3x100xb
        ConvTranspose((5,1), 100 => 250, relu; pad = SamePad()),
        #5x3x250xb
        ConvTranspose((5,1), 250 => 500, relu; pad = SamePad(), stride=(3,1)),
        #15x3x500xb
        ConvTranspose((10,3), 500 => 750, relu; pad = SamePad(), stride=(2,1)),
        #30x3x750xb
        ConvTranspose((10,3), 750 => 1000, relu; pad = SamePad(), stride=(2,1)),
        #60x3x1000xb
        ConvTranspose((10,3), 1000 => 100; pad = SamePad()),

      )
      return (encoder_μ, encoder_logvar, decoder)
      
  end


function train!(encoder_μ, encoder_logvar, decoder, train, validate, opt_enc_μ, opt_enc_logvar, opt_dec; num_epochs=100, dev=Flux.gpu)
    # The training loop for the model
    for e = 1:num_epochs
        for x in train
            x = x |> dev
            # pullback function returns the result (loss) and a pullback operator (back)
            loss, back = Flux.pullback(encoder_μ, encoder_logvar, decoder) do enc_μ, enc_logvar, dec
                vae_loss(enc_μ, enc_logvar, dec, x;dev=dev)
            end
            @info "metrics" train_loss = loss

            # Feed the back 1 to obtain the gradients and update the model parameters
            grad_enc_μ, grad_enc_logvar, grad_dec = back(1f0)

            Flux.update!(opt_enc_μ, encoder_μ, grad_enc_μ)
            Flux.update!(opt_enc_logvar, encoder_logvar, grad_enc_logvar)
            Flux.update!(opt_dec, decoder, grad_dec)

        end
        for y in validate
            y = y |> dev
            validate_loss = vae_loss(encoder_μ, encoder_logvar, decoder, y)
            @info "metrics" validate_loss = validate_loss
        end
    end
    save_model("2d-conv-100-10-vae-2", cpu(Chain(encoder_μ, encoder_logvar, decoder)), num_epochs, "")

end

function load_data(file_path, window_size)
    df = DataFrame(CSV.File(file_path; header=false, types=Float32))
    data = slidingwindow(Array(df)', window_size, stride=1)

    ts, vs = splitobs(shuffleobs(data), 0.7)
    ts_length = length(ts)
    vs_length = length(vs)

    train_set = permutedims(reshape(reduce(hcat, ts), (3, 100, window_size, ts_length)), (3, 1, 2, 4))
    validate_set = permutedims(reshape(reduce(hcat, vs), (3, 100, window_size, vs_length)), (3, 1, 2, 4))

    train_loader = DataLoader(mapslices(normalise, train_set; dims=3); batchsize=48, shuffle=true)
    validate_loader = DataLoader(mapslices(normalise, validate_set; dims=3); batchsize=48, shuffle=true)
    (train_loader, validate_loader)

end


window_size = 60
num_epochs = 250

(train_loader, validate_loader) = load_data("$(datadir())/exp_raw/data_large.csv", window_size)

encoder_μ, encoder_logvar, decoder = create_vae() |> gpu

# ADAM optimizer
opt_enc_μ = Flux.setup(Adam(0.0001), encoder_μ)
opt_enc_logvar = Flux.setup(Adam(0.0001), encoder_logvar)
opt_dec = Flux.setup(Adam(0.0001), decoder)


train!(encoder_μ, encoder_logvar, decoder, train_loader, validate_loader, opt_enc_μ,opt_enc_logvar,opt_dec, num_epochs=num_epochs)
# train!(encoder_μ, encoder_logvar, decoder, train_loader, validate_loader, Flux.Optimise.ADAM(get_config(logger, "η")), num_epochs=num_epochs)

# x = first(train_loader) |> gpu
#  μ = encoder_μ(x)
#  logσ = encoder_logvar(x)
# z = μ + gpu(randn(Float32, size(logσ))) .* exp.(logσ)
#  reshaped_input = permutedims(x, (4,1,2,3))
#  CUDA.@profile begin
#      NVTX.@range "PSI" EmergencePsi(reshaped_input, z')
#  end

# close(logger)
