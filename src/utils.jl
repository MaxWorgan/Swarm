df = DataFrame(CSV.File(datadir("exp_raw/data_3.csv"); header=false, types=Float32))
normalised = Array(df) |> normalise

window_size = 60
data = slidingwindow(normalised',window_size,stride=1)
train, test = splitobs(data, 0.7);

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
