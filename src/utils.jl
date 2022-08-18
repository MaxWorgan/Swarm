df = DataFrame(CSV.File(datadir("exp_raw/data_static.csv"); header=false, types=Float32))
delete!(df, 781)
df2 = DataFrame(CSV.File(datadir("exp_raw/data_wind.csv"); header=false, types=Float32));
df3 = DataFrame(CSV.File(datadir("exp_raw/data_forces.csv"); header=false, types=Float32));


function normalise(M) 
    min = minimum(minimum(eachcol(M)))
    max = maximum(maximum(eachcol(M)))
    return (M .- min) ./ (max - min)
end

window_size = 60


normalised = Array(df) |> normalise
data = slidingwindow(normalised',window_size,stride=1)
normalised = Array(df2) |> normalise
data2 = slidingwindow(normalised',window_size,stride=1)
normalised = Array(df3) |> normalise
data3 = slidingwindow(normalised',window_size,stride=1)

all_data = vcat([data, data2, data3]...)

train, test = splitobs(shuffleobs(all_data), 0.7);

# lets train PCA and save to file
train_encoded = Matrix{Float32}[]
for t ∈ train
    output = hcat(encoder(Flux.unsqueeze(t', 3))...)
    push!(train_encoded, output)
end

PCA_model = fit(PCA, vcat(train_encoded...)', maxoutdim=10)

save(datadir("PCA_data.jld"),"mean",PCA_model.mean,"proj",PCA_model.proj,"prinvars",PCA_model.prinvars,"tprinvar", PCA_model.tprinvar, "tvar",PCA_model.tvar)

pca_data = load(datadir("PCA_data.jld"))

function load_pca_model()
    pca_data = load(datadir("PCA_data.jld"))
    PCA{Float32}(pca_data["mean"],pca_data["proj"],pca_data["prinvars"],pca_data["tprinvar"],pca_data["tvar"])
end
