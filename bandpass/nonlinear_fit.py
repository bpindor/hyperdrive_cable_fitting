from scipy.optimize import minimize, least_squares
from numpy.polynomial import Polynomial
from .utils import get_unflagged_indices, cosine_model
import numpy as np

freqs = get_unflagged_indices() - 2

def reflection_and_polynomial_model(params):
    
    tau = params[0]
    amp = params[1]
    phase = params[2]
    reflection = cosine_model(amp,tau,phase,freqs)
    p = Polynomial(coef=params[3:])
    poly = p(freqs)
    
    return reflection + poly

def fit_lst_squares_reflection_and_poly(r0,xvals,data,verbose=0):
    
    def residuals(params):
        return reflection_and_polynomial_model(params) - data
            
    res = least_squares(residuals,r0,verbose=verbose)
    
    return res, reflection_and_polynomial_model(res.x)

def double_reflection_and_polynomial_model(params):
    
    amp = params[1]
    tau = params[0]
    phase = params[2]
    reflection1 = cosine_model(amp,tau,phase,freqs)
    
    amp = params[4]
    tau = params[3]
    phase = params[5]
    reflection2 = cosine_model(amp,tau,phase,freqs)
    p = Polynomial(coef=params[6:])
    poly = p(freqs)
    
    return reflection1 + reflection2 + poly


def fit_lst_squares_double_reflection_and_poly(r0,xvals,data):
    
    def residuals(params):
        return double_reflection_and_polynomial_model(params) - data

    res = least_squares(residuals,r0)
    
    return res, double_reflection_and_polynomial_model(res.x)

def fit_all_lsq_models(average_sols,poly_coeffs,poly_models,short_reflection_params,long_reflection_params,short_lssa_freqs,long_lssa_freq,cables_to_fit,antennas2cables,present_autos,present_chs,n_ants=128):
    double_reflection_lsq_models = []

    for i in range(n_ants):
    
        lsq_models = []

        for j in range(2):
            pol = j*3
            if(i in present_autos):

                if(antennas2cables[i] in cables_to_fit):
                    # Real Model
                    r0 = np.concatenate([short_reflection_params[i,j,0:2],poly_coeffs[i,pol,0,:]])
                    r0 = np.insert(r0,0,short_lssa_freqs[i][j])
                    r0 = np.concatenate([long_reflection_params[i,j,0:2],r0])
                    r0 = np.insert(r0,0,long_lssa_freq)

                    r_fit, r_model = fit_lst_squares_double_reflection_and_poly(r0,freqs,np.real(average_sols[i,:,pol])[present_chs])
                
                    # Imag Model
                
                    r0 = np.concatenate([short_reflection_params[i,j,2:],poly_coeffs[i,pol,1,:]])
                    r0 = np.insert(r0,0,short_lssa_freqs[i][j])
                    r0 = np.concatenate([long_reflection_params[i,j,2:],r0])
                    r0 = np.insert(r0,0,long_lssa_freq)
                    i_fit, i_model = fit_lst_squares_double_reflection_and_poly(r0,freqs,np.imag(average_sols[i,:,pol])[present_chs])
                                                                               
                else:
                
                    r0 = np.concatenate([long_reflection_params[i,j,0:2],poly_coeffs[i,pol,0,:]])
                    r0 = np.insert(r0,0,long_lssa_freq)
                    r_fit, r_model = fit_lst_squares_reflection_and_poly(r0,freqs,np.real(average_sols[i,:,pol])[present_chs])
                
                    r0 = np.concatenate([long_reflection_params[i,j,2:],poly_coeffs[i,pol,1,:]])
                    r0 = np.insert(r0,0,long_lssa_freq)
                    i_fit, i_model = fit_lst_squares_reflection_and_poly(r0,freqs,np.imag(average_sols[i,:,pol])[present_chs])
                
                lsq_models.append(r_model + 1.0j*i_model)
            
            else:
            
                lsq_models.append((poly_models[i,:,j*3])[present_chs])
                
        double_reflection_lsq_models.append(lsq_models)
    
    return np.array(double_reflection_lsq_models)

                                                                                

    
