'''Features based on Electron-IonInteraction profile, proposed by Han et al. 
(2018)'''

import numpy as np
from scipy.fft import fft
from lncrnapy import utils


class EIIPPhysicoChemical:
    '''EIIP-derived physico-chemical features, as proposed by LNCFinder. Every 
    sequence is converted into an EIIP representation, of which the power
    spectrum is calculated with a Fast Fourier Transform. Several properties 
    are derived from this power spectrum. 

    Attributes
    ----------
    `name`: `str`
        Names of the EIIP-derived physico-chemical features.
    `eiip_map`: `dict[str:float]`
        Mapping to convert nucleotides into EIIP values.

    References
    ----------
    LNCFinder: Han et al. (2018) https://doi.org/10.1093/bib/bby065'''

    def __init__(self, eiip_map={'A':0.126, 'C':0.134, 'G':0.0806, 'T':0.1335}):
        '''Initializes `EIIPPhysicoChemical` object.
        
        Arguments
        ---------
        `eiip_map`: `dict[str:float]`
            Mapping to convert nucleotides into EIIP values.'''
        
        self.name = ['EIIP 1_3', 'EIIP SNR', 'EIIP Q1', 'EIIP Q2', 'EIIP min', 
                     'EIIP max']
        self.eiip_map = eiip_map
        
    def calculate(self, data):
        '''Calculate EIIP physico-chemical features for every row in `data`.'''
        print("Calculating EIIP-derived physico-chemical features...")
        results = []
        for _, row in utils.progress(data.df.iterrows()):
            results.append(self.calculate_per_sequence(row['sequence']))
        return results

    def calculate_per_sequence(self, sequence):
        '''Calculate EIIP physico-chemical features of given `sequence`.'''
        spectrum = self.calculate_power_spectrum(sequence)

        N = len(spectrum)
        EIIP_onethird = spectrum[int(N/3)] # pcRNA often has peak at 1/3
        EIIP_SNR = EIIP_onethird / np.mean(spectrum) # Signal to noise ratio

        # Quantile statistics of top 10%
        sorted = np.sort(spectrum)[::-1][:int(N/10)] # Top 10% (desc.)
        EIIP_Q1, EIIP_Q2 = np.quantile(sorted, [0.25, 0.5]) 
        EIIP_min, EIIP_max = np.min(sorted), np.max(sorted)

        return EIIP_onethird, EIIP_SNR, EIIP_Q1, EIIP_Q2, EIIP_min, EIIP_max
    
    def calculate_power_spectrum(self, sequence):
        '''Given an RNA `sequence`, convert it to EIIP values and calculate 
        its power spectrum.'''

        # Conversion to EIIP values
        EIIP_values = []
        for base in sequence:
            try: 
                EIIP_values.append(self.eiip_map[base])
            except KeyError:
                EIIP_values.append(np.mean(list(self.eiip_map.values())))

        # Fast fourier transform, obtain power spectrum
        N = int(len(sequence)/3)*3 # Cut off at mod 3
        EIIP_values = EIIP_values[:N]
        return np.abs(fft(EIIP_values)) 