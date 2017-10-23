## Example script to get Teff, [Fe/H], and [Ti/Fe] from a NIRSPEC-1 M dwarf spectrum

import pickle
from correct_throughput import correct_throughput
from get_params import get_params

## Read in example spectrum
with open('PM_I18007+2933.pkl', 'rb') as file:
    inspec = pickle.load(file)
    
## Relative-flux-calibrate and shift to v=0
## This step can take a few minutes, unset quiet=True
## to see proof that it is making progress.
wave, flam, fvar = correct_throughput(inspec, quiet=True)

## Get parameters
teff, feh, tife = get_params(wave, flam)

print('The "true" parameters for this star are: \n'
      'Teff = 3510 K, [Fe/H] = -0.080, [Ti/Fe] = +0.050 \n \n'
      'The inferred parameters are: \n'
      'Teff = {:4.0f} K, [Fe/H] = {:+6.3f}, [Ti/Fe] = {:+6.3f}'.format(
      teff, feh, tife))