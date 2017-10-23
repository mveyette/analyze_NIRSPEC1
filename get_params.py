"""
File Name: get_params.py
Author: Mark Veyette
Date: Oct 22, 2017
Python version: 3.5

Purpose:
    Given a 1d NIRSPEC-1 spectrum, returns Teff, [Fe/H],
    and [Ti/Fe] by applying empircal corrections to
    measured EWs and matching to a BT-Settl Grid.
"""

import numpy as np
import os
import pickle
from scipy.interpolate import griddata, LinearNDInterpolator
from scipy.optimize import minimize
from scipy.ndimage.filters import maximum_filter
from scipy.signal import savgol_filter
from numpy.polynomial.chebyshev import chebval, chebfit

def getCont(wave, flam, ss=21.):
    """Fits and returns continuum"""
    cont = maximum_filter(savgol_filter(flam, 5, 2, mode='interp'), ss)
    chebx = wave-wave.min()
    chebx *= 2./chebx.max()
    chebx -= 1.
    contfit = chebfit(chebx, cont, 6)
    cont = chebval(chebx, contfit)
    return cont    

def transformEWs(p, fehind, feEWs, tiEWs):
    """Apply empirical transformation to observed EWS"""
    ifehind = p[0] + p[1]*fehind
    ifeEWs = np.zeros_like(feEWs)
    itiEWs = np.zeros_like(tiEWs)
    i0 = 2
    for i in range(len(feEWs)):
        ifeEWs[i] = p[i0+3*i] + p[i0+1+3*i]*feEWs[i] + p[i0+2+3*i]*fehind
    i0 = len(feEWs)*3+2
    for i in range(len(tiEWs)):
        itiEWs[i] = p[i0+3*i] + p[i0+1+3*i]*tiEWs[i] + p[i0+2+3*i]*fehind
    return ifehind, ifeEWs, itiEWs

def fitEWs(p, fehind, feEWs, tiEWs, weights, interp_fehind, interp_feEWs, interp_tiEWs):
    """Fitting function to minmize difference in EWs between obs and models"""
    tscl, ifeh, itife = p
    iteff = 3500 * (1+tscl)
    ifehind = interp_fehind(iteff, ifeh, itife)
    ifeEWs  = [interp_feEWs[feline](iteff, ifeh, itife) for feline in range(len(feEWs))]
    itiEWs  = [interp_tiEWs[tiline](iteff, ifeh, itife) for tiline in range(len(tiEWs))]
    chi2ish = np.mean(np.hstack(( ((ifehind-fehind)*weights[0])**2,
                                  ((ifeEWs-feEWs)*weights[1])**2,
                                  ((itiEWs-tiEWs)*weights[2])**2 )))
    if np.isnan(chi2ish):
        chi2ish = np.inf
    return chi2ish
       
def get_params(wave, flam, BTpkl='Models_ews.pkl'):
    """
    Main function
       
    Inputs:
        wave  - 1D wavelength array in microns
        flam  - 1D flux array, arbitrary units
        BTpkl - path to BT-Settl grid EWs pickle file

    Returns:
        Teff, [Fe/H], [Ti/Fe]
    """

    ## FeH index, Fe I lines, and Ti I lines from Veyette+2017
    fehindlam1 = [0.984, 0.989]
    fehindlam2 = [0.990, 0.995]
    FeLines = [[1.01475, 1.01506],
               [1.02183, 1.02200],
               [1.03980, 1.03990],
               [1.04253, 1.04273],
               [1.04719, 1.04733],
               [1.05343, 1.05360],
               [1.07854, 1.07867]]
    TiLines = [[1.00001, 1.00013],
               [1.00367, 1.00378],
               [1.00597, 1.00609],
               [1.03990, 1.04009],
               [1.04979, 1.05000],
               [1.05866, 1.05886],
               [1.06100, 1.06111],
               [1.06793, 1.06806],
               [1.07285, 1.07300],
               [1.07768, 1.07787]]

    ## Get continuum
    cont = getCont(wave, flam)

    ## Oversample spectrum
    samp = 100.0
    npix = len(wave)
    px = np.arange(npix)
    ipx = np.arange(samp*npix)/samp
    iwave = np.interp(ipx, px, wave)
    dwave = np.array([iwave[1]-iwave[0]] + ((iwave[2:]-iwave[0:-2])/2.0).tolist() + [iwave[-1]-iwave[-2]])
   
    ## Find locations of Fe Lines
    wFes = []
    for line in FeLines:
        wFes.append((iwave > line[0]) & (iwave < line[1]))
    
    ## Find locations of Ti Lines
    wTis = []
    for line in TiLines:
        wTis.append((iwave > line[0]) & (iwave < line[1]))
    
    ## Measure EWs and FeH index
    iflam = np.interp(iwave, wave, flam)
    icont = np.interp(iwave, wave, cont)
    w1 = (iwave > fehindlam1[0]) & (iwave < fehindlam1[1])
    w2 = (iwave > fehindlam2[0]) & (iwave < fehindlam2[1])
    fehind  = np.mean(iflam[w1])/np.mean(iflam[w2])
    tiEWs = [1e4*np.sum((1.0 - iflam[w]/icont[w]) * dwave[w]) for w in wTis]
    feEWs = [1e4*np.sum((1.0 - iflam[w]/icont[w]) * dwave[w]) for w in wFes]

    ## Load model EW grid
    with open(BTpkl, 'rb') as pklfile:
        modparams, modtiEWs, modfeEWs, modfehinds = pickle.load(pklfile)

    ## Create model interpolators
    interp_fehind =  LinearNDInterpolator((modparams.teff, modparams.mh, modparams.am), modfehinds)
    interp_feEWs  = [LinearNDInterpolator((modparams.teff, modparams.mh, modparams.am), 
                     modfeEWs[:,feline]) for feline in range(np.shape(modfeEWs)[1])]
    interp_tiEWs  = [LinearNDInterpolator((modparams.teff, modparams.mh, modparams.am),
                     modtiEWs[:,tiline]) for tiline in range(np.shape(modtiEWs)[1])]

    ## EW transformation parameters and RMSES from Veyette+2017
    p0 = np.array([-0.05739705,  1.06522736, -1.48494078,  0.68645177,  1.48588568,
                   -1.23607087,  0.58248966,  1.22010439, -0.69807841,  0.72890748,
                    0.69526113, -1.57189377,  0.67014627,  1.56287216, -0.16653091,
                    0.82326467,  0.16415348,  0.0800287 ,  0.89026991, -0.09245657,
                   -0.17189639,  1.04532376,  0.15551587, -0.77289903,  0.57542744,
                    0.7774392 , -0.75074802,  0.63286895,  0.75471231, -0.28748536,
                    0.75025907,  0.29402524, -1.55605448,  0.75454103,  1.55681389,
                   -0.34003218,  1.03738346,  0.27770767, -1.47602946,  0.66230678,
                    1.47199162, -0.45985325,  0.56774103,  0.46178838, -0.79949147,
                    0.60445859,  0.80564563, -0.95500996,  0.81713573,  0.95386383,
                   -0.87751444,  0.35289312,  0.90288599])
    rmses = [0.0035474296574772701,
             np.array([0.01430803, 0.00778985, 0.00498426, 0.0127829, 0.00702018, 0.00646545, 0.00775817]),
             np.array([0.00777423, 0.00556433, 0.00531975, 0.01430448, 0.01287295,
                       0.01539784,  0.00470782,  0.0062955 ,  0.00909083,  0.00783688])]
    weights = [1./rmses[0]**2, 1./rmses[1]**2, 1./rmses[2]**2]

    ## Transform EWs
    tfehind, tfeEWs, ttiEWs = transformEWs(p0, fehind, feEWs, tiEWs)

    ## Find best fit parameters
    fit = minimize(fitEWs, [0.01, 0.01, 0.01],
          args=(tfehind, tfeEWs, ttiEWs, weights, interp_fehind, interp_feEWs, interp_tiEWs))
    
    return 3500.*(1.+fit['x'][0]), fit['x'][1], fit['x'][2]





