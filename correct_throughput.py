"""
File Name: correct_throughput.py
Author: Mark Veyette
Date: Oct 22, 2017
Python version: 3.5

Purpose:
    Correct a NIRSPEC-1 spectrum for throughout by
    matching to a grid of BT-Settl models. Also improves
    wavelength solution and shifts to v=0 frame.
"""

import numpy as np
import os, sys
import astropy.io.fits as fits
from numpy.polynomial.chebyshev import chebval, chebfit
from scipy.optimize import minimize
from scipy.ndimage.filters import maximum_filter
from scipy.signal import savgol_filter

def readGrid(file):
    """Reads in BT-Settl grid"""
    try:
        hdus = fits.open(file)
        spgrid = hdus[0].data
        spwave = hdus[1].data
        spaxes = [hdus[2].data, hdus[3].data]
        hdus.close()
    except IOError:
        sys.exit("Could not open spgrid file: {}".format(file))
    spgrid /= np.median(spgrid,0)
    spwave *= 1e-4
    return spgrid, spwave, spaxes

def parseSpec(inspec, spwave):
    waves, flams, fvars = [], [], []
    for sp in inspec:
        waves.append(sp[0])
        flams.append(sp[1]/np.median(sp[1]))
        fvars.append((np.sqrt(sp[2])/np.median(sp[1]))**2)
    if np.min(waves) < np.min(spwave) or np.max(waves) > np.max(spwave):
        sys.exit('Spectrum outside of model range')
    return [np.array(item) for item in [waves, flams, fvars]]

def getrv(objwave, objflam, refwave, refflam, maxrv=[-200.,200.], waverange=[-np.inf,np.inf]):
    """
    Calculates the rv shift of an object relative to some reference spectrum.
    
    Inputs:
        objwave - obj wavelengths, 1d array
        objflam - obj flux, 1d array
        refwave - ref wavelengths, 1d array
        refflam - ref flux, 1d array
        maxrv   - min and max rv shift in km/s, 2-element array
        
    Output:
        rv shift in km/s
    """
    
    ow =   (objwave >= np.nanmin(refwave)) & (objwave <= np.nanmax(refwave)) \
         & (objwave >= waverange[0]) & (objwave <= waverange[1])
    rw =   (refwave >= np.nanmin(objwave)) & (refwave <= np.nanmax(objwave)) \
         & (refwave >= waverange[0]) & (refwave <= waverange[1])

    oscl = 1.0/np.nanmedian(objflam[ow])
    rscl = 1.0/np.nanmedian(refflam[rw])

    iflam = np.interp(refwave[rw], objwave[np.isfinite(objflam)], objflam[np.isfinite(objflam)], left=np.nan, right=np.nan)

    drv = np.nanmedian(2.*(refwave[rw][1:]-refwave[rw][:-1])/(refwave[rw][1:]+refwave[rw][:-1]))
    maxshift = maxrv / (drv*3e5)
    ssd   = []
    ss = np.arange(int(maxshift[0]),int(maxshift[1]+1))
    for s in ss:
        if s > 0:
            shiftflam = np.append(np.repeat(np.nan, s), iflam[0:-s])
        elif s < 0:
            shiftflam = np.append(iflam[-s:], np.repeat(np.nan, -s))
        else:
            shiftflam = iflam
        ssd.append(np.nansum((rscl*refflam[rw] - oscl*shiftflam)**2)/np.sum(~np.isnan(shiftflam)))
        
    ssd = np.array(ssd)
    s = ss[np.nanargmin(ssd)]
    
    return s*drv*3e5

def fitFun(p, iwave, iflam, ifvar, nbpoly, chebx, spwave, spgrid, k, l):
    bpoly, wpoly = np.split(p, [nbpoly+1])
    
    bcorr = chebval(chebx, bpoly)
    wcorr = chebval(chebx, wpoly)
    
    owave = iwave * wcorr
    oflam = iflam * bcorr
    ofvar = (np.sqrt(ifvar) * bcorr)**2.
    
    mflam = np.interp(owave, spwave, spgrid[:,k,l], left=99., right=99.)
    
    chi2 = np.mean((oflam-mflam)**2. / ofvar)
    
    return chi2

def correct_throughput(inspec, spFile='BT-Settl_Asplund2009.fits', quiet=False):
    """
    Main function

    Inputs:
        inspec - list of input spectra, each list item should
                 be a 3xN array of wavelenghts (in microns),
                 flux, and variance. One list item for each
                 order for orders 71-77
        spFile - (optional) path to fits file containing
                 BT-Setll grid, default: BT-Settl_Asplund2009.fits
        quiet  - set True to turn off all printed output
    
    Returns:
        wave - wavelength array of final combined spectrum
        flam - flux array
        fvar - variance array
    """

    ## Read in synthetic spectrum grid
    spgrid, spwave, spaxes = readGrid(spFile)

    ## Parse input spectrum
    waves, flams, fvars = parseSpec(inspec, spwave)

    ## Define cheby grid
    norder, npix = waves.shape
    chebx = np.linspace(-1,1,npix)

    ## Initial guesses
    ## Polynomial to correct for blaze function 
    nbpoly = 3
    bpolys = np.zeros((norder, nbpoly+1))
    ## Polynomial to correct wavelength
    nwpoly = 1
    wpolys = np.zeros((norder, nwpoly+1))
    wpolys[:,0] = 1.0
    for i in range(norder):
        bpolys[i] = chebfit(chebx, 1./flams[i], nbpoly)
        rv = getrv(waves[i], flams[i]*chebval(chebx,bpolys[i]), spwave, spgrid[:,9,2])
        wpolys[i,0] = (1.+rv/3e5)
    ## Model parameters
    teff = 3500
    mh   = 0.0
    ips = np.array([np.hstack((bpolys[i],wpolys[i])) for i in range(norder)])
    
    ## Loop over entire model grid and fit for each order
    chi2s = np.zeros([norder,spgrid.shape[1],spgrid.shape[2]])
    chi2s [:] = 9e9
    ps = np.tile(np.zeros_like(ips[0]), [norder,spgrid.shape[1],spgrid.shape[2],1])
    for k in range(0, spgrid.shape[1]):
        for l in range(spgrid.shape[2]):
            if not quiet:
                print('Teff = {0}, [M/H] = {1}'.format(spaxes[0][k],spaxes[1][l]))
            for i in range(norder):
                flam = flams[i]
                fvar = fvars[i]
                wave = waves[i]
                fit = minimize(fitFun, ips[i], args=(wave,flam,fvar,nbpoly,chebx,spwave,spgrid,k,l))
                chi2s[i,k,l] = fit['fun']
                ps[i,k,l] = fit['x']
                #if not quiet:
                #    print('    '+fit['message'])
                #    print('    '+str(fit['x']))
                #    print('    '+str(fit['fun']))
                #    print()
            if not quiet:
                print(np.mean(chi2s[:,k,l]))
    mink, minl = np.unravel_index(np.argmin(np.sum(chi2s,0)),[len(spaxes[0]),len(spaxes[1])])
    bpolys, wpolys = np.split(ps[:,mink,minl], [nbpoly+1], axis=1)
    teff  = spaxes[0][mink]
    mh    = spaxes[1][minl]
    
    ## Correct everything
    corrwaves = np.zeros_like(waves)
    corrflams = np.zeros_like(flams)
    corrfvars = np.zeros_like(fvars)
    for i in range(norder):
        corrwaves[i] = waves[i] * chebval(chebx, wpolys[i])
        corrflams[i] = flams[i] * chebval(chebx, bpolys[i])
        corrfvars[i] = (np.sqrt(fvars[i]) * chebval(chebx, bpolys[i]))**2.

    ## Flatten and sort
    wave = corrwaves.flatten()
    srt = np.argsort(wave)
    wave = wave[srt]
    flam = corrflams.flatten()[srt]
    fvar = corrfvars.flatten()[srt]

    return wave, flam, fvar

