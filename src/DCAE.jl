using DrWatson
quickactivate(@__DIR__)

using Flux, CSV, DataFrames, MLDataPattern, StatsBase, CUDA

function normalise(M) 
    min = minimum(minimum(eachcol(M)))
    max = maximum(maximum(eachcol(M)))
    return (M .- min) ./ (max - min)
end

function load_and_normalise_data()
    df = DataFrame(CSV.File("$(datadir())/exp_raw/data_3.csv"; types=Float32))
    normalised = Array(df) |> normalise
    window_size = 60
    slidingwindow(normalised',window_size,stride=1)
end

data = load_and_normalise_data()

function create_ae_1d()
    # Define the encoder and decoder networks 
    encoder = Chain(
        # 60x900xb
        Conv((9,), 900 => 9000, relu; pad=SamePad()),
        MaxPool((2,)),
        # 30x9000xb
        Conv((5,), 9000 => 4500, relu; pad=SamePad()),
        MaxPool((2,)),
        # 15x4500xb
        Conv((5,), 4500 => 2250, relu; pad=SamePad()),
        # 15x2250xb
        MaxPool((3,)),
        Conv((3,), 2250 => 1000, relu; pad=SamePad()),
        Conv((3,), 1000 => 100, relu; pad=SamePad()),
        # 5x100xb
        Flux.flatten,
        Dense(500, 100)
    )
    decoder = Chain(
        Dense(100, 500),
        (x -> reshape(x, 5, 100, :)),
        # 5x100xb
        ConvTranspose((3,), 100 => 1000, relu; pad=SamePad()),
        ConvTranspose((3,), 1000 => 2250, relu; pad=SamePad()),
        Upsample((3,)),
        # 15x2250xb
        ConvTranspose((5,), 2250 => 4500, relu; pad=SamePad()),
        Upsample((2,)),
        # 30x4500xb
        ConvTranspose((5,), 4500 => 9000, relu; pad=SamePad()),
        Upsample((2,)),
        # 60x9000xb
        ConvTranspose((9,), 9000 => 900, relu; pad=SamePad()),
        # 60x900xb
    )
    return (encoder, decoder)
end

function save_model(m, epoch, loss)
    model_row = LegolasFlux.ModelRow(; weights = fetch_weights(cpu(m)),architecture_version=1, loss=0.0001)
    write_model_row("1d_300_model-$epoch-$loss.arrow", model_row)
end

function rearrange_1D(x)
    permutedims(cat(x..., dims=3), [2,1,3])
end

function train_model_1D!(model, train, validate, opt; epochs=20, bs=16, dev=Flux.gpu)
    ps = Flux.params(model)
    local train_loss, train_loss_acc
    local validate_loss, validate_loss_acc
    local last_improvement = 0
    local prev_best_loss = 0.01
    local improvement_thresh = 5.0
    validate_losses = Vector{Float64}()
    for e in 1:epochs
        train_loss_acc = 0.0
        for x in eachbatch(train, size=bs)
            x  = rearrange_1D(x) |> dev
            gs = Flux.gradient(ps) do
                train_loss = Flux.Losses.mse(model(x),x)
                return train_loss
            end
            train_loss_acc += train_loss
            Flux.update!(opt, ps, gs)
        end
        validate_loss_acc = 0.0
        for y in eachbatch(validate, size=bs)
            y  = rearrange_1D(y) |> dev
            validate_loss = Flux.Losses.mse(model(y), y)
            validate_loss_acc += validate_loss
        end
        validate_loss_acc = round(validate_loss_acc / (length(validate)/bs); digits=6)
        train_loss_acc = round(train_loss_acc / (length(train)/bs) ;digits=6)
        if validate_loss_acc < 0.001
            if validate_loss_acc < prev_best_loss
                @info "new best accuracy $validate_loss_acc saving model..."
                save_model(model, e, validate_loss_acc)
                last_improvement = e
                prev_best_loss = validate_loss_acc
            elseif (e - last_improvement) >= improvement_thresh && opt.eta > 1e-5
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
        println("Epoch $e/$epochs\t train loss: $train_loss_acc\t validate loss: $validate_loss_acc")
    end
    validate_losses
 end


 losses_0001       = train_model_1D!(model, train, validate, Flux.Optimise.ADAM(0.0001); epochs=200, bs=48);