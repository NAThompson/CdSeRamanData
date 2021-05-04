# Raman spectra pipeline for Brian Lerner and Ben Lawrie.
# Written by Nick Thompson, but follows Brian Lerner's python script.

using Plots
using PlotThemes
using HDF5
using Wavelets
theme(:solarized)
PLOTS_DEFAULTS = Dict(:dpi => 600)
Plots.GRBackend()


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
    # NB: I'm still trying to figure out the layout!
    # Note that Julia is Fortran order, Python is C order.

    # Anyways, on to the background subtraction.
    # In Brian's original Python code, he finds 3160 = 79x40 minima,
    # which he then subtracts from each image.

    dims = size(dataset)
    # For the example file: dims = (40, 79, 1024)
    # Don't forget that Julia is 1-indexed!
    minima = Vector{UInt16}(undef, dims[1]*dims[2])

    for k in 1:dims[3]
        minimum = typemax(UInt16);
        for j in 1:dims[2]
            for i in 1:dims[1]
                if dataset[i,j,k] < minimum
                    minimum = dataset[i,j,k]
                end
            end
        end
        # Now we have the minimum; subtract it off:
        for j in 1:dims[2]
            for i in 1:dims[1]
                @assert dataset[i,j,k] >= minimum
                dataset[i,j,k] -= minimum
            end
        end
    end

    # I suspect this data is in meters; you want it in nanometers?
    # Note that these wavelengths *look* equally spaced at a glance-but they are not *quite* equispaced.
    wavelengths = read(h5["Acquisition2/ImageData/DimensionScaleC"])*1e9

    # The Wavelets Package operates on floating point data:
    i = 4
    j = 3
    x = convert(Array{Float32}, dataset[i,j,:])

    # Translation invariance puts a little ring right on the peak (for the examples I tried!)
    # Hence the TI=false. I also tried the 6 and 10 vanishing moment symlet, which were both inferior
    # to the 8 vanishing moment symlet.
    # This is consistent with what has been repeatedly observed in the literature,
    # See S. Mallat, A Wavelet Tour of Signal Processing.
    #y = denoise(x, wavelet(WT.sym8, WT.Filter), TI=false)
    #plt = plot(wavelengths, [x, y], label=["Raw data" "Denoised with 8 Vanishing Moment Symlet"], xlabel="λ (nm)", size=(1200,800))
    #max_idx = argmax(y)
    #plot!([wavelengths[max_idx]], seriestype="vline", label="argmax(denoised)")
    #savefig("denoised_qc_$i-$j.png")

    # The first denoise was a quality check.
    # Now let's spool up the turbo and denoise them all:
    denoisedData = Array{Float32, 3}(undef, (dims[1],dims[2],dims[3]))
    maxima = Array{Float32,2}(undef, (dims[1],dims[2]))
    for j in 1:dims[2]
        for i in 1:dims[1]
            x = convert(Array{Float32}, dataset[i,j,:])
            denoisedData[i,j,:] = denoise(x, wavelet(WT.sym8, WT.Filter), TI=false)
            plt = plot(wavelengths, [x, denoisedData[i,j,:]], label=["Raw data" "Denoised with 8 Vanishing Moment Symlet"], xlabel="λ (nm)", size=(1200,800))
            max_idx = argmax(denoisedData[i,j,:])
            plot!([wavelengths[max_idx]], seriestype="vline", label="argmax(denoised)")
            display(plt)
            sleep(1.5)
            maximum = 0
            for k in 1:dims[3]
                if denoisedData[i,j,k] > maximum
                    maximum = denoisedData[i,j,k]
                end
            end
            maxima[i,j] = maximum
        end
    end


    close(h5)
end

main()