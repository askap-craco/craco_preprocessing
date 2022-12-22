#from socket import VM_SOCKETS_INVALID_VERSION
from hashlib import blake2s
import numpy as np
from craco_fits_modifier import bl2ant, ants2bl

from iqrm import iqrm_mask

def get_slice_along_axes(arr, axi_indices):
    '''
    Adapted from: https://stackoverflow.com/questions/24398708/slicing-a-numpy-array-along-a-dynamically-specified-axis
    Slices the ndarray with indices along axis
    arr: numpy.ndarray
        The data array to slice
    axi_index: list 
        A list of tuples containing axis as the first element and a list of indices as the second element
        E.g. [(0, [1, 2, 3]), (1, [12, 13]), (2, [100, 200])]
        This will return the [1, 2, 3] elements from the 0th axes, [12, 13] elements from the 1st axis and [100, 200] elements from the 2nd axis
        negative numbers will work in both (axis and index) values with usual meaning
    '''
    sl = [slice(None)] * arr.ndim
    for pair in axi_indices:
        sl[pair[0]] = pair[1]
        return tuple(sl)

def get_median_and_mad(y):
    '''Returns the median and MAD of a 1-D numpy array `y`
    '''
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    return med, mad

def get_globally_bad_idxs_along_axis(block, axis):
    '''
    Test fx
    '''
    print(f"Obtaining bad idxs along axis = {axis}")
    axes_tuple = [i for i in range(block.ndim)].remove(axis)
    axis_tseries = block.mean(axis = axes_tuple)
    rmed, rmad = get_median_and_mad(axis_tseries.real)
    imed, imad = get_median_and_mad(axis_tseries.imag)

    r_rms = rmad * 1.4826
    i_rms = imad * 1.4826

    r_normed_axis_tseries = (axis_tseries.real - rmed) / r_rms
    i_normed_axis_tseries = (axis_tseries.imag - imed) / i_rms

    bad_idxs = (r_normed_axis_tseries > 5) | (i_normed_axis_tseries > 5)
    return bad_idxs

def get_freq_mask(baseline_data, threshold=3.0):
    #Take the mean along the pol axis and rms along the time axis
    #print("Shape of data recieved = ", baseline_data.shape)
    
    bl_d = baseline_data[..., 0] * baseline_data[..., 1]        #apply the weights before getting the mask
    #bl_d = baseline_data[..., 0]

    #print("shape of data after using masks = ", bl_d.shape)
    rms = bl_d.mean(axis=1).std(axis=-1)
    #print("Shape of rms = ", rms.shape)
    mask, votes = iqrm_mask(rms, radius = len(rms) / 10, threshold=threshold)
    return mask

def get_time_mask(baseline_data, threshold = 3.0):
    #Take the mean along the pol axis and rms along freq axis
    bl_d = baseline_data[..., 0] * baseline_data[..., 1]
    #bl_d = baseline_data[..., 0]
    rms = bl_d.mean(axis=1).std(axis=0)
    mask, votes = iqrm_mask(rms, radius = len(rms)/ 10, threshold=threshold)
    return mask

def get_cas_time_mask(cas_data, threshold=3.0):
    #cd = cas_data
    cd = cas_data[..., 0] * cas_data[..., 1]
    rms = cd.mean(axis=1).std(axis=0)
    mask, votes = iqrm_mask(rms, radius=len(rms)/10, threshold=threshold)
    return mask

def convert_acm_to_complx_bl(acm, block):
    data_shape = block[next(iter(block))].shape
    acm_bl = {}
    for ibl in block.keys():
        a1, a2 = bl2ant(ibl)
        chanmask = acm[a1] | acm[a2]
        acm_bl[ibl] = np.ones(data_shape, dtype=np.complex64) * np.nan
        acm_bl[ibl][chanmask, :, :] = 1 + 1j
        
    return acm_bl

def convert_ccm_to_complx_bl(ccm, block):
    data_shape = block[next(iter(block))].shape
    ccm_bl = {}
    for ibl in block.keys():
        ccm_bl[ibl] = np.ones(data_shape, dtype=np.complex64) * np.nan
        
        if ibl in ccm.keys():    
            chanmask = ccm[ibl]
            ccm_bl[ibl][chanmask, :, :] = 1 + 1j
    
    return ccm_bl

def convert_casm_to_complx_bl(casm, block):
    data_shape = block[next(iter(block))].shape
    casm_bl = {}
    for ibl in block.keys():
        casm_bl[ibl] = np.ones(data_shape, dtype=np.complex64) * np.nan
        casm_bl[ibl][casm[0], :, :] = 1 + 1j
        
    return casm_bl
                



class RFI_cleaner(object):
    def __init__(self, fitsHDU):
        '''
        FitsHDU: object
            Object of CracoFitsReader class
        '''
        self.hdu = fitsHDU
        self.freq_axis = 1
        self.pol_axis = 2
        self.bl_axis = 0
        self.time_axis = 3


    #@property
    #def freq_axis(self):
    #    for item in self.hdu.header.items():
    #        if item[1] == "FREQ":
    #            return int(item[0].strip("CTYPE"))
    #        else:
    #            raise RuntimeError("Could not figure out which axis is the frequency axis")    
            

    #@property
    #def pol_axis(self):
    #    for item in self.hdu.header.items():
    #        if item[1] == "STOKES":
    #            return int(item[0].strip("CTYPE"))
    #        else:
    #            raise RuntimeError("Could not figure out which axis is the polarisation axis")

    #def get_statistics(self, block):


    def zap_channel_range(self, chanlist):
        '''
        chanlist: list of ints
            A list of channel indices to zap
        '''
        slicer = get_slice_along_axes(self.hdu.vis.data, (self.freq_axis, chanlist))
        self.hdu.vis.data[slicer] = 0.0


    def zap_samp_range(self, samplist):
        '''
        samplist: list of ints
            A list of samp indices to zap
        '''
        slicer = get_slice_along_axes(self.hdu.vis.data, (self.time_axis, samplist))
        self.hdu.vis.data[slicer] = 0.0

    def clean_block(self, block):
        '''
        Applies various RFI mitigation strategies on the block
        Returns a cleaned block

        block: numpy.ndarray
            The input time block created by the CracoFitsReader class
            (nbl, nf, npol, nt)
        '''
        #block_abs = np.abs(block)       #We only need the absolute values to work out the RFI cells
        zap_mask = np.zeros_like(block.real, dtype='int8')
        '''
        bad_baselines = get_globally_bad_idxs_along_axis(block, axis=0)
        bad_chans = get_globally_bad_idxs_along_axis(block, axis=1)
        bad_pols = get_globally_bad_idxs_along_axis(block, axis=2)
        bad_samps = get_globally_bad_idxs_along_axis(block, axis=3)
        

        zap_mask[:, bad_chans, :, :] = 1
        zap_mask[:, :, :, bad_samps] = 1
        zap_mask[bad_baselines, :, :, :] = 1
        zap_mask[:, :, bad_pols, :] = 1

        print("Zapping the bad idxs now")
        #block[zap_mask] = 0       #Will do it for both real and imag parts
        '''
        block[:] = 0
        ncells_zapped = np.sum(zap_mask)
        print(f"Zapped {ncells_zapped} cells")

        print("Returning the cleaned block")
        return block, ncells_zapped


    def get_IQRM_autocorr_masks(self, block_dict):
        autocorr_masks = {}
        for ibl, baseline_data in block_dict.items():
            a1, a2 = bl2ant(ibl)
            if a1 != a2:
                continue
            baseline_data = np.abs(block_dict[ibl])
            #print("Shape of baseline_data = ", baseline_data.shape)
            autocorr_mask = get_freq_mask(baseline_data)
            autocorr_masks[a1] = autocorr_mask

        return autocorr_masks

    def get_cas_sum(self, block_dict):
        cas_sum = np.zeros(block_dict[next(iter(block_dict))].shape)
        cas_sum[..., 1] = 1     #set all weights to one
        #print("Cas_sum.shape = ", cas_sum.shape)
        for ibl, baseline_data in block_dict.items():
            a1, a2 = bl2ant(ibl)
            if a1!=a2:
                bl_d = np.abs(baseline_data[..., 0])
                #bl_d = baseline_data[..., 0] * baseline_data[..., 1]
                #print("bl_d.shape = ", bl_d.shape)
                cas_sum[..., 0] += bl_d
        return cas_sum

    def clean_bl_using_autocorr_mask(self, ibl, bldata, autocorr_masks):
        ant1, ant2 = bl2ant(ibl)
        autocorr_mask = autocorr_masks[ant1] | autocorr_masks[ant2]
        bldata[autocorr_mask, :, :, 1] = 0


    def get_IQRM_crosscorr_mask(self, ibl, bldata):
        bl_mask = get_freq_mask(bldata, threshold = 5)
        return bl_mask

    def get_IQRM_time_mask(self, ibl, bldata):
        bl_mask = get_time_mask(bldata, threshold=3)
        return bl_mask

    def IQRM_filter(self, block_dict, mask_autos = True, mask_corrs = True, mask_cas = True, mask_time = False):
        '''
        Does the IQRM magic
        Returns
        -------
        autocorr_masks: dict
            Dictionary containing autocorr_masks keyed by antid (1 - 36), valued by a 1-D numpy array of len nf
        crosscorr_masks: dict
            Dictionary containing crosscorr_masks keyed by blid (256*a1 + a2), valued by 1-D numpy array of len nf
        cas_masks: dict
            Single element dictionary keyed by 0, valued by a 1-D numpy array of len nf
        '''
        autocorr_masks = {}
        if mask_autos:
            autocorr_masks = self.get_IQRM_autocorr_masks(block_dict)
        crosscorr_masks = {}
        cas_masks = {}
        time_masks = {}

        if mask_time or mask_autos or mask_corrs:
            for ibl, baseline_data in block_dict.items():
                if mask_time:
                    time_mask = self.get_IQRM_time_mask(ibl, np.abs(baseline_data))
                    #print("Shape of time_mask = ", time_mask.shape)
                    baseline_data[:, :, time_mask, 1] = 0
                    time_masks[ibl] = time_mask
                if mask_autos:
                    self.clean_bl_using_autocorr_mask(ibl, np.abs(baseline_data), autocorr_masks)

                if mask_corrs:
                    ant1, ant2 = bl2ant(ibl)

                    if ant1 == ant2:
                        continue
            
                    crosscorr_bl_mask = self.get_IQRM_crosscorr_mask(ibl, np.abs(baseline_data))
                    baseline_data[crosscorr_bl_mask, :, :, 1] = 0
                    crosscorr_masks[ibl] = crosscorr_bl_mask

        if mask_cas:
            cas_sum = self.get_cas_sum(block_dict)
            #Finally find bad samples in the CAS
            cas_masks[0] = get_freq_mask(cas_sum)
            cas_sum[cas_masks[0], :, :, 1] = 0
            for ibl, baseline_data in block_dict.items():
                baseline_data[cas_masks[0], :, :, 1] = 0
            cas_time_masks = get_cas_time_mask(cas_sum)
            for ibl, baseline_data in block_dict.items():
                baseline_data[:, :, cas_time_masks, 1] = 0

        return autocorr_masks, crosscorr_masks, cas_masks, time_masks
        
        



            












        


