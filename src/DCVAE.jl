using DrWatson
quickactivate(@__DIR__)
using Flux.Data: DataLoader
using Flux, DataFrames, StatsBase,MLDataPattern, CUDA, PlotlyJS, LegolasFlux, CSV
using Wandb, Dates,Logging

# Start a new run, tracking hyperparameters in config
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
        # 60x900xb
        Conv((9,), 900 => 9000, relu; pad = SamePad()),
        MaxPool((2,)),
        # 30x9000xb
        Conv((5,), 9000 => 4500, relu; pad = SamePad()),
        MaxPool((2,)),
        # 15x4500xb
        Conv((5,),4500 => 2250, relu; pad = SamePad()),
        # 15x2250xb
        MaxPool((3,)),
        Conv((3,),2250 => 1000, relu; pad = SamePad()),
        Conv((3,),1000 => 100, relu; pad = SamePad()),
        # 5x100xb
        Flux.flatten,
        Dense(500,100,relu)
      )
      
    encoder_μ      = Chain(encoder_features, Dense(100, 100)) 
      
    encoder_logvar = Chain(encoder_features, Dense(100, 100)) 
    
    decoder = Chain(
        Dense(100,500,relu),
        (x -> reshape(x, 5,100,:)),
          # 5x100xb
        ConvTranspose((3,), 100  => 1000, relu; pad = SamePad()),
        ConvTranspose((3,), 1000 => 2250, relu; pad = SamePad()),
        Upsample((3,)),
        # 15x2250xb
        ConvTranspose((5,), 2250 => 4500, relu; pad = SamePad()),
        Upsample((2,)),
        # 30x4500xb
        ConvTranspose((5,), 4500 => 9000, relu; pad = SamePad()),
        Upsample((2,)),
        # 60x9000xb
        ConvTranspose((9,), 9000 => 900; pad = SamePad()),
        # 60x900xb
      )
      return (encoder_μ, encoder_logvar, decoder)
      
  end

  function save_model(m, epoch, loss)
    model_row = LegolasFlux.ModelRow(; weights = fetch_weights(cpu(m)),architecture_version=1, loss=loss)
    write_model_row("1d_300_model-vae-$epoch-$loss.arrow", model_row)
end

function rearrange_1D(x)
    permutedims(cat(x..., dims=3), [2,1,3])
end

function train!(encoder_μ, encoder_logvar, decoder, train, validate, optimiser; num_epochs=100, bs=16, dev=Flux.gpu)
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
        for x in eachbatch(train, size=bs)
            x = rearrange_1D(x) |> dev
            # pullback function returns the result (loss) and a pullback operator (back)
            loss, back = Flux.pullback(trainable_params) do
                vae_loss(encoder_μ, encoder_logvar, decoder, x)
            end
            x |> cpu
            @info "metrics" train_loss=loss
            train_loss_acc += loss
            # Feed the pullback 1 to obtain the gradients and update the model parameters
            gradients = back(1f0)
            Flux.Optimise.update!(optimiser, trainable_params, gradients)
            
        end
        validate_loss_acc = 0.0
        for y in eachbatch(validate, size=bs)
            y  = rearrange_1D(y) |> dev
            validate_loss = vae_loss(encoder_μ, encoder_logvar, decoder, y)
            y |> cpu
            @info "metrics" validate_loss=validate_loss
            validate_loss_acc += validate_loss
        end
        validate_loss_acc = round(validate_loss_acc / (length(validate)/bs); digits=6)
        train_loss_acc = round(train_loss_acc / (length(train)/bs) ;digits=6)
        if abs(validate_loss_acc) < 2e+6 
            if abs(validate_loss_acc) < prev_best_loss
                @info "new best accuracy $validate_loss_acc saving model..."
                save_model(cpu(Chain(encoder_μ, encoder_logvar, decoder)), e, validate_loss_acc)
                last_improvement = e
                prev_best_loss = validate_loss_acc
            elseif (e - last_improvement) >= improvement_thresh && optimiser.eta > 1e-5
                @info "Not improved in $improvement_thresh epochs. Dropping learning rate to $(opt.eta / 2.0)"
                opt.eta /= 2.0
                last_improvement = e # give it some time to improve
                improvement_thresh = improvement_thresh * 1.5
            elseif (e - last_improvement) >= 15
                @info "Not improved in 15 epochs. Converged I guess"
                break
            end
        end
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


window_size = 60

num_epochs = 100



#load the data
df = DataFrame(CSV.File("data_test.csv"; header=false, types=Float32));

normalised = Array(df) |> normalise

data = slidingwindow(normalised',window_size,stride=1)

train_set, validate_set = splitobs(shuffleobs(data), 0.7);

encoder_μ, encoder_logvar, decoder = create_vae() |> gpu



train!(encoder_μ, encoder_logvar, decoder, train_set, validate_set, Flux.Optimise.ADAM(get_config(logger, "η")), num_epochs=num_epochs, bs=get_config(logger, "batch_size"))

close(logger)