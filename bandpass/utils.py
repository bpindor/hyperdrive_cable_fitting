import numpy as np

def get_unflagged_indices(clip_width=0):
    """
    Returns indices of unflagged frequency channels
    """

    n_bands = 24
    n_chan = 32
    edge_flags = 2 + clip_width
    centre_flag = 16

    channels = np.arange(n_chan)
    channels = np.delete(channels,centre_flag)
    channels = channels[2+clip_width:-(2+clip_width)]

    all_channels = []
    for n in range(n_bands):
        for c in channels:
            all_channels.append(c+(n*n_chan))
    return np.array(all_channels)


def fit_autos(autos,xvals=None,order=3,clip_width=0):
    """
    Fit polynomial to autocorrelations
    """
    
    from numpy.polynomial import Polynomial
    
    if(xvals is None):
        xvals = np.arange(np.shape(autos)[1])
    auto_models = np.zeros_like(autos)
    
    n_pols = 2
    
    for i in range(n_pols):
        fit = Polynomial.fit(xvals,autos[i],deg=order)
    
        auto_models[i] = fit(xvals)
    
    return auto_models

def fit_all_autos(all_autos,xvals,present_autos,n_ants=128,order=7):

    all_auto_poly_models = []

    for i in range(n_ants):
        if(i in present_autos):
            tile_xx, tile_yy = all_autos[i][0],all_autos[i][1]
            auto_models = fit_autos([tile_xx,tile_yy],xvals=xvals, order=order)
            all_auto_poly_models.append(auto_models)
        else:
            all_auto_poly_models.append([None,None])

    return all_auto_poly_models

def fit_hyperdrive_sols(sols,order=3,clip_width=0,return_coeffs=False):
    """
    Fits polynomial to real and imaginary part of hyperdrive solutions
    """

    from numpy.polynomial import Polynomial


    # Remove flagged channels
    # Hyperdrive solutions have dimensions (ant,freq,pols)

    n_ants,n_freqs,n_pols = np.shape(sols)
    
    xvals = np.arange(np.shape(sols)[1])
    models_out = np.zeros_like(sols,dtype=complex)
    coeffs_out = np.zeros((n_ants,n_pols,2,order+1))

    #clipped_x = get_unflagged_indices(n_freqs,clip_width=clip_width)
    clipped_x = get_unflagged_indices()
    
    # Remove flagged tiles which are nan at first unflagged frequency
    good_tiles = np.argwhere(~np.isnan(sols[:,2,0]))

    freq_array = np.arange(n_freqs)

    for ant in good_tiles:
        for pol in range(n_pols):
            z_r = Polynomial.fit(clipped_x,np.real(sols[ant,clipped_x,pol]),deg=order)
            z_i = Polynomial.fit(clipped_x,np.imag(sols[ant,clipped_x,pol]),deg=order)
            coeffs_r = z_r.convert().coef
            coeffs_i = z_i.convert().coef
            coeffs_out[ant,pol,0,:] = coeffs_r
            coeffs_out[ant,pol,1,:] = coeffs_i
            models_out[ant,:,pol] = z_r(freq_array) + z_i(freq_array) * 1j
            
    if(return_coeffs):
        return models_out, coeffs_out
    else:
        return models_out

def estimate_reflection_frequency(cable):
    """
    Estimates cable reflection frequency to narrow search range
    """
    
    # empicially, 90m cables have frequency ~ 0.03
    
    freq_90m = 0.03 * 2.0 * np.pi
    
    if(cable ==  'RG6_90'):
        return freq_90m
    elif(cable == 'RG6_150'):
        return freq_90m * (150.0/90.0)
    elif(cable == 'RG6_230'):
        return freq_90m * (230.0/90.0)
    elif(cable == 'LMR400_320'):
        return freq_90m * (320.0/90.0)
    elif(cable == 'LMR400_400'):
        return freq_90m * (400.0/90.0)
    elif(cable == 'LMR400_524'):
        return freq_90m * (524.0/90.0)
    elif(cable == 'Long_Wavelength_Reflection'):
        return 0.0033 * 2.0 * np.pi
    else:
        print('Unknown Cable Type {cable}')
        return 0.0

def get_lssa_frequencies(all_autos,all_auto_poly_models,antennas2cables,present_autos,n_ants=128):
    """ 
    Use Lomb-Scargle periodogram to find reflection frequencies 
    """

    import scipy.signal as signal

    # x = sampled channels
    # y = sample values
    # w = compute at these frequencies

    all_lssa_freqs = []
    all_pgrams = []

    x = get_unflagged_indices()
    nout = 10000 # frequency resolution

    for i in range(n_ants):
        fiducial_freq = estimate_reflection_frequency(antennas2cables[i])
        w = np.linspace(0.9*fiducial_freq,1.1*fiducial_freq,nout)
    
        lssa_freqs = []
        pgrams = []
        for j in range(2): # XX/YY
            if(i in present_autos):
                y = all_autos[i][j] - all_auto_poly_models[i][j]
                pgram = signal.lombscargle(x, y, w, normalize=False)
                lssa_freqs.append(w[np.argmax(pgram)] / (2.0 * np.pi))
                pgrams.append(pgram)
            else:
                lssa_freqs.append(0.0)
                pgrams.append(0.0)
        
        all_lssa_freqs.append(lssa_freqs)
        
    return np.array(all_lssa_freqs)

def get_interpolated_models(average_sols,poly_models,present_autos,n_ants=128):
    
    from scipy import interpolate

    all_interp_models = []

    all_freqs = np.arange(768)
    x_new = all_freqs[2:-2]

    xvals = get_unflagged_indices()
    interp_xvals = xvals - 2 # Interpolated models already clip first two channels

    flagged_models = np.zeros(len(xvals))

    for i in range(n_ants):
    
        interp_model = []

        for j in [0,3]: # XX,YY pols 

            if(i in present_autos):
            
                y = (average_sols[i,:,j])[xvals] - (poly_models[i,:,j])[xvals]
                f_real = interpolate.interp1d(xvals, np.real(y))
                f_imag = interpolate.interp1d(xvals, np.imag(y))
                interp_model_real = f_real(x_new)
                interp_model_imag = f_imag(x_new)
                interp_model.append(interp_model_real + 1.0j*interp_model_imag)
                        
            else:
                interp_model.append(np.zeros(len(x_new)))

        all_interp_models.append(interp_model)
    
    return  np.array(all_interp_models)

def cosine_model(amp,freq_in,phase_in,freqs):
    """ 
    Returns cosine function with given parameters at values (freqs)
    """

    model = amp * np.cos(2*np.pi*(freqs * freq_in) + phase_in)
    return model

# definition of fft
# sum (x * exp(-ikn 2pi/ N))

def my_fft(data,k_in):
    """ 
    Performs Direct Fourier Summation at given frequency.
    Used to super-resolve FFT frequencies
    """
    
    out = []
    N = float(len(data))
    for k in k_in:
        out.append(np.sum([x * np.exp(-np.pi * 2.0 * k * float(n) *1.0j / N) for x,n in zip(data,np.arange(len(data)))]))
        
    return np.array(out)  

def reflection_model_from_lssa(data,lssa_freq,verbose=False,return_amp_phase=False):
    """
    Returns model and parameters for Fourier mode corresponding to input 
    frequency
    """
    
    # First need to work out fftfrequency which corresponds to
    fft_df = 1.0/(float(len(data)))
    fft_k = lssa_freq / fft_df
    
    lssa_model = my_fft(data,[fft_k])
    
    amp = abs((lssa_model)) * 2 / len(data)
    phase = np.arctan2(np.imag(lssa_model) , np.real(lssa_model))
    if(verbose):
        print(amp,phase)
        
    if(return_amp_phase):    
        return amp,phase,cosine_model(amp,lssa_freq,phase,np.arange(len(data)))
    else:
        return cosine_model(amp,lssa_freq,phase,np.arange(len(data)))

def fit_all_reflection_models(interp_models,poly_models,antennas2cables,cables_to_fit,all_lssa_freqs,long_lssa_freq,present_autos,n_ants=128):

    double_reflection_models = []
    long_lssa_params = []
    short_lssa_params = []

    xvals = get_unflagged_indices()
    interp_xvals = get_unflagged_indices() - 2 

    for i in range(n_ants):
    
        lssa_models = []
        long_params = []
        short_params = []

        for j in range(2): # XX/YY   
            if(i in present_autos):
                        
                if(antennas2cables[i] in cables_to_fit):
                    
                    r_amp,r_phase,real_lssa_model = reflection_model_from_lssa(np.real(interp_models[i][j]),all_lssa_freqs[i][j],verbose=False,return_amp_phase=True)
                    i_amp,i_phase,imag_lssa_model = reflection_model_from_lssa(np.imag(interp_models[i][j]),all_lssa_freqs[i][j],verbose=False,return_amp_phase=True)
                    short_params.append([r_amp[0],r_phase[0],i_amp[0],i_phase[0]])
            
                else:
                    real_lssa_model = np.zeros_like((poly_models[i,:,j*3]))
                    imag_lssa_model = np.zeros_like((poly_models[i,:,j*3]))
                    short_params.append([0,0,0,0])
                
                r_amp,r_phase,real_long_model = reflection_model_from_lssa(np.real(interp_models[i][j]),long_lssa_freq,verbose=False,return_amp_phase=True)
                i_amp,i_phase,imag_long_model = reflection_model_from_lssa(np.imag(interp_models[i][j]),long_lssa_freq,verbose=False,return_amp_phase=True)    
                        
                lssa_models.append((poly_models[i,:,j*3])[xvals] + real_lssa_model[interp_xvals] \
                               + 1.0j * imag_lssa_model[interp_xvals] \
                              + real_long_model[interp_xvals] + 1.0j * imag_long_model[interp_xvals])
                long_params.append([r_amp[0],r_phase[0],i_amp[0],i_phase[0]])
            else:
                lssa_models.append((poly_models[i,:,j*3])[xvals])
                long_params.append([0,0,0,0])
                short_params.append([0,0,0,0])                                                           
            
        double_reflection_models.append(lssa_models)
        long_lssa_params.append(long_params)
        short_lssa_params.append(short_params)                                                                   

    return np.array(double_reflection_models,dtype=complex), np.array(long_lssa_params), np.array(short_lssa_params)     
        

