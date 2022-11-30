using DrWatson
quickactivate(@__DIR__)
using Flux.Data: DataLoader
using Flux, DataFrames, StatsBase,MLDataPattern, CUDA, PlotlyJS, LegolasFlux, CSV
using Wandb, Dates,Logging

## Start a new run, tracking hyperparameters in config
logger = WandbLogger(
   project = "Swarm-VAE",
   name = "swarm-vae-training-$(now())",
   config = Dict(
      "η" => 0.00001,
      "batch_size" => 48,
      "data_set" => "data_test"
   ),
)

global_logger(logger)

function gaussian_nll(x̂, logσ, x)
    return 0.5 * (@. ( (x - x̂) / exp(logσ))^2 + logσ + 0.5 * log2(pi))
end

function softclip(input, min_val)
    return min_val .+ NNlib.softplus(input - min_val)
end

function reconstruction_loss(x̂, x)
    logσ = log(sqrt(mean((x - x̂).^2)))
    logσ = softclip(logσ, -6)
    rec  = sum(gaussian_nll(x̂, logσ, x))
    return rec
end

function vae_loss(encoder_μ, encoder_logvar, decoder, x)
    len = size(x)[end]
    @assert len != 0
    # Forward propagate through mean encoder and std encoders
    μ = encoder_μ(x)
    logσ = encoder_logvar(x)
    z = μ + gpu(randn(Float32, size(logσ))) .* exp.(logσ)
    # Reconstruct from latent sample
    x̂ = decoder(z)
    
    kl = -0.5 * sum(@. 1 + logσ - μ^2 - exp(logσ) )

    rec = reconstruction_loss(x̂, x)

    @info "metrics" reconstruction_loss=rec kl=kl
    
    return rec + kl
 
end

function create_vae()
    
    encoder_features = Chain(
        # 60x300xb
        Conv((9,), 300 => 3000, relu; pad = SamePad()),
        MaxPool((2,)),
        # 30x3000xb
        Conv((5,), 3000 => 1500, relu; pad = SamePad()),
        MaxPool((2,)),
        # 15x4500xb
        Conv((5,),1500 => 750, relu; pad = SamePad()),
        # 15x2250xb
        MaxPool((3,)),
        Conv((3,),750 => 250, relu; pad = SamePad()),
        Conv((3,),250 => 25, relu; pad = SamePad()),
        Conv((3,),25 => 10, relu; pad = SamePad()),
        # 5x25xb
        Flux.flatten,
        Dense(50,10,relu)
      )
      
    encoder_μ      = Chain(encoder_features, Dense(10,10)) 
      
    encoder_logvar = Chain(encoder_features, Dense(10,10)) 
    
    decoder = Chain(
        Dense(10,50,relu),
        (x -> reshape(x, 5,10,:)),
          # 5x100xb
        ConvTranspose((3,), 10  => 25, relu; pad = SamePad()),
        ConvTranspose((3,), 25  => 250, relu; pad = SamePad()),
        ConvTranspose((3,), 250 => 750, relu; pad = SamePad()),
        Upsample((3,)),
        # 15x2250xb
        ConvTranspose((5,), 750 => 1500 , relu; pad = SamePad()),
        Upsample((2,)),
        # 30x4500xb
        ConvTranspose((5,), 1500 => 3000, relu; pad = SamePad()),
        Upsample((2,)),
        # 60x9000xb
        ConvTranspose((9,), 3000 => 300; pad = SamePad()),
        # 60x900xb
      )
      return (encoder_μ, encoder_logvar, decoder)
      
  end

function save_model(m, epoch, loss)
    model_row = LegolasFlux.ModelV1(; weights = fetch_weights(cpu(m)),architecture_version=1, loss=loss)
    write_model_row("1d_100_model-vae-$epoch-$loss.arrow", model_row)
end

function rearrange_1D(x)
    permutedims(cat(x..., dims=3), [2,1,3])
end

function train!(encoder_μ, encoder_logvar, decoder, train, validate, optimiser; num_epochs=100, dev=Flux.gpu)
    # The training loop for the model
    trainable_params = Flux.params(encoder_μ, encoder_logvar, decoder)
    local train_loss_acc
    local validate_loss, validate_loss_acc
    local last_improvement = 0
    local prev_best_loss = 0.01
    local improvement_thresh = 5.0
    validate_losses = Vector{Float64}()
    for e = 1:num_epochs
        train_loss_acc = 0.0
        for x in train
            x = x |> gpu
            #x = rearrange_1D(x) |> dev
            # pullback function returns the result (loss) and a pullback operator (back)
            loss, back = Flux.pullback(trainable_params) do
                vae_loss(encoder_μ, encoder_logvar, decoder, x)
            end
            @info "metrics" train_loss=loss
            train_loss_acc += loss
            # Feed the pullback 1 to obtain the gradients and update the model parameters
            gradients = back(1f0)
            Flux.Optimise.update!(optimiser, trainable_params, gradients)
        end
        validate_loss_acc = 0.0
        for y in validate
            y = y |> gpu
            validate_loss = vae_loss(encoder_μ, encoder_logvar, decoder, y)
            @info "metrics" validate_loss=validate_loss
            validate_loss_acc += validate_loss
        end
        validate_loss_acc = round(validate_loss_acc / (length(validate)/validate.batchsize); digits=6)
        train_loss_acc    = round(train_loss_acc / (length(train)/train.batchsize) ;digits=6)
        push!(validate_losses, validate_loss_acc)
        println("Epoch $e/$num_epochs\t train loss: $train_loss_acc\t validate loss: $validate_loss_acc")
    end        
    @info "saving post training with validate_loss:  $validate_loss_acc"
    save_model(cpu(Chain(encoder_μ, encoder_logvar, decoder)), num_epochs, validate_loss_acc)

end

function normalise(M) 
    min_m = minimum(M)
    max_m = maximum(M)
    return (M .- min_m) ./ (max_m - min_m)
end


function load_data(file_path, window_size)
    
    df           = DataFrame(CSV.File(file_path; header=false, types=Float32));
    data         = slidingwindow(Array(df)',window_size,stride=1)

    ts, vs       = splitobs(shuffleobs(data), 0.7)
    ts_length    = length(ts)
    vs_length    = length(vs)

    train_set    = permutedims(reshape(reduce(hcat, ts), (300,window_size,ts_length)), (2,1,3))
    validate_set = permutedims(reshape(reduce(hcat, vs), (300,window_size,vs_length)), (2,1,3))

    train_loader    = DataLoader(mapslices(normalise,train_set; dims=3); batchsize=48,shuffle=true)
    validate_loader = DataLoader(mapslices(normalise,validate_set; dims=3); batchsize=48, shuffle=true)
    (train_loader, validate_loader)

end



window_size = 60

(train_loader, validate_loader) = load_data("$(datadir())/exp_raw/data_large.csv", window_size)

num_epochs  = 250

encoder_μ, encoder_logvar, decoder = create_vae() |> gpu

train!(encoder_μ, encoder_logvar, decoder, train_loader, validate_loader, Flux.Optimise.ADAM(get_config(logger, "η")), num_epochs=num_epochs)

close(logger)
