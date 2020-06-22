#!/usr/bin/env python
# coding: utf-8

import sys
import time
import soundfile as sf
import numpy as np
from scipy import signal as sg
from scipy import special as sc
import matplotlib.pyplot as plt
from museval.metrics import bss_eval_images, bss_eval_sources

### Function for audio pre-processing ###
def pre_processing(data, Fs, down_sample):
    
    #Transform stereo into monoral
    if data.ndim == 2:
        wavdata = 0.5*data[:, 0] + 0.5*data[:, 1]
    else:
        wavdata = data
    
    #Down sampling and normalization of the wave
    if down_sample is not None:
        wavdata = sg.resample_poly(wavdata, down_sample, Fs)
        Fs = down_sample
    
    return wavdata, Fs

### Function for getting STFT ###
def get_STFT(wav, Fs, frame_length, frame_shift):
    
    #Calculate the index of window size and overlap
    FL = round(frame_length * Fs)
    FS = round(frame_shift * Fs)
    OL = FL - FS
    
    #Execute STFT
    freqs, times, Y = sg.stft(wav, fs=Fs, window='hamm', nperseg=FL, noverlap=OL)
    
    #Display the size of input
    print("Spectrogram size (freq, time) = " + str(Y.shape))
    
    return Y, Fs, freqs, times

### Function for getting inverse STFT ###
def get_invSTFT(Y, Fs, frame_length, frame_shift):
    
    #Get the inverse STFT
    FL = round(frame_length * Fs)
    FS = round(frame_shift * Fs)
    OL = FL - FS
    _, rec_wav = sg.istft(Y, fs=Fs, window='hamm', nperseg=FL, noverlap=OL)
    
    return rec_wav, Fs

### Function for computing an expected value under the generalized inverse Gaussian (GIG) ###
def get_GIG_expectation(gamma, rho, tau):
    
    #Initialize variables
    if np.isscalar(gamma) == True:
        gamma = gamma * np.ones_like(rho) #Extend into the same size
    E_x = np.zeros_like(rho)
    E_invx = np.zeros_like(rho)
    
    #Avoid "invalid multiplication" error caused by small tau
    tau[tau < 1e-100] = 1e-100
    
    #Compute the scaled bessel function "kve" using the scipy.special
    rt_rho = np.sqrt(rho)
    rt_tau = np.sqrt(tau)
    bessel_gamma = sc.kve(gamma, 2*rt_rho*rt_tau)
    bessel_gamma_1minus = sc.kve(gamma-1, 2*rt_rho*rt_tau)
    bessel_gamma_1plus = sc.kve(gamma+1, 2*rt_rho*rt_tau)
    
    #Compute expectations of x and inverse x under the GIG
    E_x = (bessel_gamma_1plus * rt_tau) / (rt_rho * bessel_gamma)
    E_invx = (bessel_gamma_1minus * rt_rho) / (rt_tau * bessel_gamma)
    
    return E_x, E_invx

### Function for getting the number of valid basements ###
def get_valid_M(Y, E_w, E_h, E_u):
    
    #Define power and cutoff for truncation (original paper)
    #power = E_w
    #cut_off = 1e-6 * np.sum(power)
    
    #More stable way
    power = E_w * np.amax(E_h, axis=0) * np.amax(E_u, axis=1)
    cut_off = 1e-10 * np.sum(power)
    
    #Get the sorted maximum power in descending order
    sort_power = np.flipud(np.argsort(power))
    idx = np.where(power[sort_power] > cut_off)[0][-1]
    
    #Truncate components under the cut_off
    valid_M = sort_power[: idx+1]
    
    return valid_M

### Function for updating basement matrix ###
def update_h(Y, E_w, E_h, E_u, E_invw, E_invh, E_invu, rho_h, tau_h, valid_M, a):
    
    ### Update the only "valid_M" indices ###
    #Update auxiliary variables
    omega = (E_h[:, valid_M] @ (E_w[valid_M, np.newaxis] * E_u[valid_M, :]))**(-1)
    phi_w = 1/E_invw
    phi_h = 1/E_invh
    phi_u = 1/E_invu
    sum_phi = Y * (phi_h[:, valid_M] @ (phi_w[valid_M, np.newaxis] * phi_u[valid_M, :]))**(-2)
    
    #Update the parameters for GIG
    rho_h[:, valid_M] = a + omega @ (E_w[np.newaxis, valid_M] * E_u[valid_M, :].T)
    tau_h[:, valid_M] = phi_h[:, valid_M]**2 * (sum_phi @ (phi_w[np.newaxis, valid_M] * phi_u[valid_M, :].T))
    
    #Update basements
    E_h[:, valid_M], E_invh[:, valid_M] = get_GIG_expectation(a, rho_h[:, valid_M], tau_h[:, valid_M])
    
    return E_h, E_invh, rho_h, tau_h

### Function for updating activation matrix ###
def update_u(Y, E_w, E_h, E_u, E_invw, E_invh, E_invu, rho_u, tau_u, valid_M, b):
    
    ### Update only "valid_M" indices ###
    #Update auxiliary variables
    omega = (E_h[:, valid_M] @ (E_w[valid_M, np.newaxis] * E_u[valid_M, :]))**(-1)
    phi_w = 1/E_invw
    phi_h = 1/E_invh
    phi_u = 1/E_invu
    sum_phi = Y * (phi_h[:, valid_M] @ (phi_w[valid_M, np.newaxis] * phi_u[valid_M, :]))**(-2)
    
    #Update the parameters for GIG
    rho_u[valid_M, :] = b + (E_w[valid_M, np.newaxis] * E_h[:, valid_M].T) @ omega
    tau_u[valid_M, :] = phi_u[valid_M, :]**2 * ((phi_w[valid_M, np.newaxis] * phi_h[:, valid_M].T) @ sum_phi)
    
    #Update activation
    E_u[valid_M, :], E_invu[valid_M, :] = get_GIG_expectation(b, rho_u[valid_M, :], tau_u[valid_M, :])
    
    return E_u, E_invu, rho_u, tau_u

### Function for updating weight vector ###
def update_w(Y, E_w, E_h, E_u, E_invw, E_invh, E_invu, rho_w, tau_w, valid_M, alpha):
    
    ### Update only "valid_M" indices ###
    #Update auxiliary variables
    omega = (E_h[:, valid_M] @ (E_w[valid_M, np.newaxis] * E_u[valid_M, :]))**(-1)
    phi_w = 1/E_invw
    phi_h = 1/E_invh
    phi_u = 1/E_invu
    sum_phi = Y * (phi_h[:, valid_M] @ (phi_w[valid_M, np.newaxis] * phi_u[valid_M, :]))**(-2)
    
    #Update the parameters for GIG
    c = np.mean(Y)
    rho_w[valid_M] = alpha*c + np.sum((E_h[:, valid_M].T @ omega) * E_u[valid_M, :], axis=1)
    tau_w[valid_M] = phi_w[valid_M]**2 * np.sum((phi_h[:, valid_M].T @ sum_phi) * phi_u[valid_M, :], axis=1)
    
    #Update weights
    E_w[valid_M], E_invw[valid_M] = get_GIG_expectation(alpha/M, rho_w[valid_M], tau_w[valid_M])
    
    return E_w, E_invw, rho_w, tau_w

### Function for computing the logarithm of Gamma distribution and GIG ###
def get_prior(E_x, E_invx, k, lamb, rho, tau):
    
    #The underlying probability distribution
    #P(x)=Gamma(x | k, lambda)
    #Q(x)=GIG(x | k, rho, tau)
    #The log(x) term is canceled out between logP(x) and logQ(x)
    
    #Initialization
    prior = 0
    
    #Avoid "invalid multiplication" error caused by small tau
    tau[tau < 1e-100] = 1e-100
    
    #The logP(x) term other than the canceled log(x)
    prior = prior + (k * np.log(lamb) - sc.gammaln(k)) * E_x.size
    prior = prior - np.sum(lamb * E_x)
    
    #The -logQ(x) term
    #log2
    prior = prior + np.log(2) * E_x.size
    
    #-k/2*(log(rho)-log(tau))
    prior = prior - (k/2) * np.sum(np.log(rho) - np.log(tau))
    
    #rho*x+tau/x
    prior = prior + np.sum(rho * E_x + tau * E_invx)
    
    #log(bessel(2*sqrt(rho*tau)))
    prior = prior + np.sum(np.log(sc.kve(k, 2*np.sqrt(rho*tau))))
    
    return prior

### Function for executing the GaP-NMF algorithm ###
def Gamma_process_NMF(Y, max_iter, bound_ratio, M, a, b, alpha):
    
    #Define the size of spectrogram
    K, N = Y.shape[0], Y.shape[1]
    
    #Initialize parameters for GIG randomly
    rho_h = 1e3 * np.random.gamma(100, 1./1000, size=(K, M))
    tau_h = 1e3 * np.random.gamma(100, 1./1000, size=(K, M))
    rho_u = 1e3 * np.random.gamma(100, 1./1000, size=(M, N))
    tau_u = 1e3 * np.random.gamma(100, 1./1000, size=(M, N))
    rho_w = M * 1e3 * np.random.gamma(100, 1./1000, size=(M, ))
    tau_w = 1./M * 1e3 * np.random.gamma(100, 1./1000, size=(M, ))
    bound_list = []
    M_list = []
    
    #Initialize the expected value of w, H, and U under the GIG
    E_h, E_invh = get_GIG_expectation(a, rho_h, tau_h)
    E_u, E_invu = get_GIG_expectation(b, rho_u, tau_u)
    E_w, E_invw = get_GIG_expectation(alpha/M, rho_w, tau_w)
    
    #Repeat for each iteration
    for i in range(max_iter):
        
        #Initialization
        start = time.time()
        
        ### Update all expected values ###
        #Update the expected value of activation matrix
        valid_M = get_valid_M(Y, E_w, E_h, E_u)
        E_u, E_invu, rho_u, tau_u = update_u(Y, E_w, E_h, E_u, E_invw, E_invh, E_invu, rho_u, tau_u, valid_M, b)
        
        #Update the expected value of basement matrix
        valid_M = get_valid_M(Y, E_w, E_h, E_u)
        E_h, E_invh, rho_h, tau_h = update_h(Y, E_w, E_h, E_u, E_invw, E_invh, E_invu, rho_h, tau_h, valid_M, a)
        
        #Update the expected value of weight vector
        valid_M = get_valid_M(Y, E_w, E_h, E_u)
        E_w, E_invw, rho_w, tau_w = update_w(Y, E_w, E_h, E_u, E_invw, E_invh, E_invu, rho_w, tau_w, valid_M, alpha)
        
        #Reset the parameters corresponding to invalid basements
        valid_M = get_valid_M(Y, E_w, E_h, E_u)
        invalid_M = np.setdiff1d(np.arange(M), valid_M)
        rho_h[:, invalid_M], tau_h[:, invalid_M] = a, 0
        rho_u[invalid_M, :], tau_u[invalid_M, :] = b, 0
        
        #Compute the expected values again
        E_h[:, valid_M], E_invh[:, valid_M] = get_GIG_expectation(a, rho_h[:, valid_M], tau_h[:, valid_M])
        E_u[valid_M, :], E_invu[valid_M, :] = get_GIG_expectation(b, rho_u[valid_M, :], tau_u[valid_M, :])
        E_w[valid_M], E_invw[valid_M] = get_GIG_expectation(alpha/M, rho_w[valid_M], tau_w[valid_M])
        
        ### Compute the variational lower bound ###
        bound = 0
        
        #The likelihood term (the formula 13 in the original paper)
        omega = E_h[:, valid_M] @ (E_w[valid_M, np.newaxis] * E_u[valid_M, :])
        phi_w = 1/E_invw
        phi_h = 1/E_invh
        phi_u = 1/E_invu
        sum_phi = phi_h[:, valid_M] @ (phi_w[valid_M, np.newaxis] * phi_u[valid_M, :])
        bound = bound - np.sum(Y / sum_phi + np.log(omega))
        
        #The prior terms (the 2nd to 4th lines of formula 8 in the original paper)
        bound = bound + get_prior(E_h, E_invh, a, a, rho_h, tau_h)
        bound = bound + get_prior(E_u, E_invu, b, b, rho_u, tau_u)
        c = np.mean(Y)
        bound = bound + get_prior(E_w, E_invw, alpha/M, alpha*c, rho_w, tau_w)
        
        #Compare the bound with the last one
        if i == 0:
            diff = 0
        else:
            diff = (bound - bound_list[-1]) / np.abs(bound_list[-1]) #diffrence from the former iteration
        finish = time.time() - start
        print("Iter{}, Bound={:.2f}, Diff={:.5f}, Process_time={:.1f}sec".format(i+1, bound, diff, finish))
        
        #Add to lists
        bound_list.append(bound)
        M_list.append(valid_M.shape[0])
        
        #Condition of convergence
        if i != 0 and np.abs(diff) < bound_ratio:
            break
    
    return E_w, E_h, E_u, valid_M, bound_list, M_list

### Function for plotting Spectrogram and loss curve ###
def display_graph(Y, X, times, freqs, bound, M_list):
    
    #Plot the loss curve
    plt.rcParams["font.size"] = 16
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.title('An original spectrogram')
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.pcolormesh(times, freqs, 10*np.log10(np.abs(Y)), cmap='jet')
    plt.colorbar(orientation='horizontal').set_label('Power')
    
    plt.subplot(1, 2, 2)
    plt.title('The approximation by GaP-NMF')
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.pcolormesh(times, freqs, 10*np.log10(np.abs(X)), cmap='jet')
    plt.colorbar(orientation='horizontal').set_label('Power')
    
    #Plot the bound curve
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(bound)+1), bound[:], marker='.')
    plt.title('Variational bound curve')
    plt.xlabel('Iteration')
    plt.ylabel('Variational lower bound')
    
    #Plot the change in number of basements
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(M_list)+1), M_list[:], marker='.')
    plt.title('The change in the number of basements')
    plt.xlabel('Iteration')
    plt.ylabel('The number of basements')
    plt.show()
    
    return

### Main ###
if __name__ == "__main__":
    
    #Setup
    down_sample = 16000   #Downsampling rate (Hz) [Default]None or 16000
    frame_length = 0.064  #STFT window width (second) [Default]0.064
    frame_shift = 0.032   #STFT window shift (second) [Default]0.032
    max_iter = 1000       #The maximum number of iteration [Default]1000
    bound_ratio = 1e-4    #Thresholds to stop iteration [Default]1e-4
    M = 100               #Initial number of basements [Default]100
    a = 0.1               #Shape parameter for P(H) [Default]0.1 
    b = 0.1               #Shape parameter for P(U) [Default]0.1
    alpha = 1.0           #Shape parameter for P(w) [Default]1.0
    spec_type = "pow"     #Select the type of spectrum ("amp" or "pow") [Default]pow
    
    #Define random seed
    np.random.seed(seed=128)
    
    #Read a sound file
    source = "./data/piano.wav"
    data, Fs = sf.read(source)
    
    #Call my function for audio pre-processing
    data, Fs = pre_processing(data, Fs, down_sample)
    
    #Call my function for getting STFT (complex STFT amplitude)
    Y, Fs, freqs, times = get_STFT(data, Fs, frame_length, frame_shift)
    arg = np.angle(Y)
    Y = np.abs(Y)
    
    #Normalization
    Y = Y / np.amax(Y) #maximum = 1
    Y[Y < 1e-8] = 1e-8 #minimum = 1e-8
    
    #In the case of power spectrogram
    if spec_type == "pow":
        Y = Y**2
    
    #Call my function for executing the GaP-NMF
    E_w, E_h, E_u, valid_M, bound, M_list = Gamma_process_NMF(Y, max_iter, bound_ratio, M, a, b, alpha)
    
    #Spectrogram approximated by the GaP-NMF
    print("The number of valid basements: {}".format(valid_M.shape[0]))
    #print("The weight vector: {}".format(np.sort(E_w)[::-1]))
    X = E_h[:, valid_M] @ (E_w[valid_M, np.newaxis] * E_u[valid_M, :])
    
    #In the case of power spectrogram
    if spec_type == "pow":
        Y = np.sqrt(Y)
        X = np.sqrt(X)
    
    #Phase recovery
    Y = Y * np.exp(1j*arg)
    X = X * np.exp(1j*arg)
    
    #Call my function for getting inverse STFT
    original_wav, Fs = get_invSTFT(Y, Fs, frame_length, frame_shift)
    original_wav = original_wav[: int(data.shape[0])] #inverse stft includes residual part due to zero padding
    sf.write("./log/original.wav", original_wav, Fs)
    
    #Call my function for getting inverse STFT
    rec_wav, Fs = get_invSTFT(X, Fs, frame_length, frame_shift)
    rec_wav = rec_wav[: int(data.shape[0])] #inverse stft includes residual part due to zero padding
    sf.write("./log/approximated.wav", rec_wav, Fs)
    
    #Compute the SDR by bss_eval from museval library ver.4
    original_wav = original_wav[np.newaxis, :, np.newaxis]
    rec_wav = rec_wav[np.newaxis, :, np.newaxis]
    sdr, isr, sir, sar, perm = bss_eval_images(original_wav, rec_wav)
    #sdr, sir, sar, perm = bss_eval_sources(truth, data) #Not recommended by documentation
    print("SDR: {:.3f} [dB]".format(sdr[0, 0]))
    
    #Call my function for displaying graph
    display_graph(Y, X, times, freqs, bound, M_list)