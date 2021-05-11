# Raman spectra pipeline for Brian Lerner and Ben Lawrie.
# Written by Nick Thompson, but follows Brian Lerner's python script.

using Plots
using PlotThemes
using HDF5
using Wavelets
using LsqFit
using BenchmarkTools
theme(:solarized)
#PLOTS_DEFAULTS = Dict(:dpi => 600)
Plots.GRBackend()

@. bell_curve(λ, p) = p[1] + p[2]*exp(-((λ-p[3])/p[4])^2)

# L(λ) = k + A(Γ/2)²/((λ-λ₀)² + (Γ/2)²)
@. lorentzian(λ, p) = p[1] + p[2]*(p[4]/2)^2/((λ-p[3])^2 + (p[4]/2)^2)

# f(λ) = k + A((Γ/2)²(1-q²) - qΓ(λ-λ₀))/((λ-λ₀)² + (Γ/2)²)
# k = p[1]
# A = p[2]
# λ₀ = p[3]
# Γ = p[4]
# q = p[5]
@. fano(λ, p) = p[1] + p[2]*(  (p[4]/2)^2*( 1 - (p[5])^2 )  - p[5]*p[4]*(λ-p[3]) )/( (λ-p[3])^2 + (p[4]/2)^2 )

# This model must have the following meaning:
# First parameter: Vertical shift
# Second parameter: Amplitude
# Third parameter: Horizontal shift
# Fourth parameter: Linewidth
# If it doesn't, the initial guesses will be completely wrong.
# Of course, it might converge anyway though!
# If least-squares fitting was really robust against initial guess,
# we wouldn't need to fit to the denoised data.
# But we need the initial guesses from the denoised data.
function fit_denoised_to_model(wavelengths, denoised)
    max_idx = argmax(denoised)
    max_reflectance = denoised[max_idx]
    
    max_wavelength = wavelengths[max_idx]
    max_reflectance = denoised[max_idx]
    idx = 0
    while denoised[max_idx + idx] > max_reflectance/2
        idx += 1
    end
    linewidth_guess = max_wavelength - wavelengths[idx]

    p0 = [0.0, max_reflectance, max_wavelength, linewidth_guess]

    idx1 = max(1, max_idx - idx)
    idx2 = min(max_idx + idx, length(wavelengths))
    fit = curve_fit(lorentzian, view(wavelengths, idx1:idx2), view(denoised, idx1:idx2), p0)

    if !fit.converged
        println("The Lorentzian fit did not converge. This bad.")
    end

    p0 = fit.param
    #println("Fit parameters for Lorentzian: $p0")
    # Now q = 0:
    push!(p0, 0.0)
    fit = curve_fit(fano, view(wavelengths, idx1:idx2), view(denoised, idx1:idx2), p0)

    if !fit.converged
        println("The Fano fit did not converge.")
    end

    p0 = fit.param
    #println("Fit parameters for Fano      : $p0")
    fit
end

function fit_denoised_brick_to_model(wavelengths, denoisedBrick)
    dims = size(denoisedBrick)
    fitParamsBrick = Array{Float64,3}(undef, dims[1], dims[2], 5)
    Threads.@threads for j in 1:dims[2]
        for i in 1:dims[1]
            fit = fit_denoised_to_model(wavelengths, denoisedBrick[i,j,:])
            fitParamsBrick[i,j,:] = fit.param
        end
    end
    fitParamsBrick
end

function load_dataset(filestring)
    if !HDF5.ishdf5(filestring)
        println("$filestring is not an HDF5 file.")
        exit(1)
    end

    h5 = HDF5.h5open(filestring, "r")
    # This just reads the metadata:
    dataset = h5["Acquisition2/ImageData/Image"]
    # This actually loads the data into RAM:
    dataset = read(dataset)
    # For unknown reasons (to Nick), this returns a 5d dataset, and two of the dimensions have length 1.
    # Use dropdims to truncate it into a 3d brick:
    dataset = dropdims(dataset, dims=(3,4))
    # Note that dropdims = Python's squeeze.
    # It's not clear to me that explicit specification of the dimensions is robust.
    # Another solution is to drop all the dimensions of length 1:
    #a = dropdims(dset, dims = tuple(findall(size(dset) .== 1)...))
    # However, caveat emptor:
    # https://stackoverflow.com/questions/52505760/dropping-singleton-dimensions-in-julia/52507859
    # Julia reads this as a 40x79x1024 image brick of UInt64's.
    # Python reads it as a 1024x79x40 brick.

    # I suspect this data is in meters; you want it in nanometers?
    # Note that these wavelengths *look* equally spaced at a glance-but they are not *quite* equispaced.
    wavelengths = convert(Array{Float64}, read(h5["Acquisition2/ImageData/DimensionScaleC"])*1e9)
    close(h5)
    # The data is stored in UInt16 format.
    # Obviously it's be preferable to keep it that way, but the denoiser and the fitting routine
    # expects floating point data.
    # We could also get away with Float32 here, but let's wait until there's a performance problem
    # as it could cause numerical problems using less precision.
    floatDataset = convert(Array{Float64,3}, dataset)
    return wavelengths, floatDataset
end

function denoise_brick(rawData)
    dims = size(rawData)
    denoisedData = Array{Float64, 3}(undef, (dims[1],dims[2],dims[3]))
    Threads.@threads for j in 1:dims[2]
        for i in 1:dims[1]
            # Translation invariance puts a little ring right on the peak (for the examples I tried!)
            # Hence the TI=false. I also tried the 6 and 10 vanishing moment symlet, which were both inferior
            # to the 8 vanishing moment symlet.
            # This is consistent with what has been repeatedly observed in the literature,
            # See S. Mallat, A Wavelet Tour of Signal Processing.
            denoisedData[i,j,:] = denoise(rawData[i,j,:], wavelet(WT.sym8, WT.Filter), TI=false)
            m = minimum(denoisedData[i,j,:])
            # We're subtracting off the minimum here because it will make our plots less janky over steps.
            # Brian Lerner also commented in the original python program that it was a poor man's background subtraction.
            rawData[i,j,:] .-= m
            denoisedData[i,j,:] .-= m
        end
    end
    denoisedData
end

function spot_check_workflow(i,j, wavelengths, denoisedDataset, rawDataset)
    println("Spot checking trace $i, $j")
    fit = fit_denoised_to_model(wavelengths, denoisedDataset[i,j,:])
    if !fit.converged
        println("The Lorentzian fit did not converge on trace $i,$j; this is weird.")
    end
    # Use: fieldnames(typeof(fit)) to explore this fit.
    # This gives, for me, fit.param, fit.resid, fit.jacobian, fit.converged, fit.wt
    fit_data = fano(wavelengths, fit.param)
    q = fit.param[5]
    plt = plot(wavelengths,
               [rawDataset[i,j,:],
               denoisedDataset[i,j,:],
               fit_data],
               label=["Raw data" "Denoised with 8 Vanishing Moment Symlet" "Fano fit (q=$q)"],
               xlabel="λ (nm)",
               ylabel="Reflectance (arbitrary units)",
               size=(1200,800))
    max_idx = argmax(denoisedDataset[i,j,:])
    lambda0 = wavelengths[max_idx]
    plot!([lambda0], seriestype="vline", label="argmax(denoised) = $lambda0")
    plot!(ylims = (0, maximum(rawDataset[i,j,:])))
    savefig("qc_trace_$i-$j.png")
end

function q_heatmap(fitBrick)
    dims = size(fitBrick)
    qMap = Array{Float64, 2}(undef, (dims[1],dims[2]))
    for j in 1:dims[2]
        for i in 1:dims[1]
            q = fitBrick[i,j,5]
            # clip:
            if q < -0.4
                q = -0.4
            end
            if q > 0.4
                q = 0.4
            end
            qMap[i,j] = q
        end
    end
    plt = heatmap(1:dims[1], 1:dims[2], qMap)
    savefig("q_heatmap.png")
end

function main()
    # An example file:
    # Assume that comments refer to this file.
    # This filename contains quite a bit of metadata.
    # For example, this was taken off oa CdSe/CdTe heterostructure.
    # Probably shot with a 10kV laser at 180 pico amps.
    # 1s - is it an integration type?
    filestring = "../data/CdSe-CdTe_10kV-180pA-hyper-1s-1B-1G-150nm-150g-200um-VISNIR-80K.h5"
    for arg in ARGS
        filestring = arg
    end

    wavelengths, rawDataset = load_dataset(filestring)
    denoisedDataset = denoise_brick(rawDataset)

    dims = size(denoisedDataset)
    #for j in 1:4
    #    for i in 1:4
    #        spot_check_workflow(i, j, wavelengths, denoisedDataset, rawDataset)
    #    end
    #end

    fitBrick = fit_denoised_brick_to_model(wavelengths, denoisedDataset)
    q_heatmap(fitBrick)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end