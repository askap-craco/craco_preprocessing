import numpy as np
import matplotlib.pyplot as plt

from craco_fits_modifier import CracoFitsReader, bl2ant, bl2array
from rfi_strategy import RFI_cleaner

from astropy.io import fits

def convert_acm_to_complx_bl(acm, block):
    data_shape = block[next(iter(block))].shape[:-1]
    acm_bl = {}
    for ibl in block.keys():
        a1, a2 = bl2ant(ibl)
        chanmask = acm[a1] | acm[a2]
        acm_bl[ibl] = np.ones(data_shape, dtype=np.complex64) * np.nan
        acm_bl[ibl][chanmask, :, :] = 1 + 1j
        
    return acm_bl

def convert_ccm_to_complx_bl(ccm, block):
    data_shape = block[next(iter(block))].shape[:-1]
    ccm_bl = {}
    for ibl in block.keys():
        ccm_bl[ibl] = np.ones(data_shape, dtype=np.complex64) * np.nan
        
        if ibl in ccm.keys():    
            chanmask = ccm[ibl]
            ccm_bl[ibl][chanmask, :, :] = 1 + 1j
    
    return ccm_bl

def convert_casm_to_complx_bl(casm, block):
    data_shape = block[next(iter(block))].shape[:-1]
    casm_bl = {}
    for ibl in block.keys():
        casm_bl[ibl] = np.ones(data_shape, dtype=np.complex64) * np.nan
        casm_bl[ibl][casm[0], :, :] = 1 + 1j
        
    return casm_bl

def convert_timem_to_complx_bl(timem, block):
    data_shape = block[next(iter(block))].shape[:-1]
    timem_bl = {}
    for ibl in block.keys():
        timem_bl[ibl] = np.ones(data_shape, dtype=np.complex64) * np.nan
        
        if ibl in timem.keys():    
            timemask = timem[ibl]
            timem_bl[ibl][:, :, timemask] = 1 + 1j
    
    return timem_bl

def main():
    fname = "/home/gup037/tmp/SB43128_run3.uvfits"
    f = fits.open(fname)
    hdulist = CracoFitsReader(f)

    rfi_st = RFI_cleaner(hdulist)

    nt = 32
    ipol = 0
    blocker = hdulist.time_blocks(nt)

    for iblock, block in enumerate(blocker):
        block_arr = bl2array(block, sort = False)
        '''
        
        nbl_to_plot = len(block) // 40
        nrows = nbl_to_plot // 5 + 1
        ncols = 5
        fig1, axes1 = plt.subplots(nrows = nrows, ncols = ncols)
        axes1 = axes1.flatten()
        
        for bl_idx in range(len(block_arr)):
            which_bl = bl_idx + 20 * 22
            bl = block_arr[which_bl]
            if bl_idx > nbl_to_plot:
                break
            axes1[bl_idx].imshow(bl.real[:, ipol, :, 0] , aspect='auto', interpolation='None')
            blid = list(block.keys())[which_bl]
            ant1, ant2 = bl2ant(blid)
            axes1[bl_idx].set_title(str(ant1) + "," +str(ant2))
        
        plt.title("Block = " + str(iblock))
        plt.show(block=False)

        cas_sum = rfi_st.get_cas_sum(block)

        plt.figure()
        plt.imshow(cas_sum[:, 0, :, 0], aspect='auto', interpolation="None")
        plt.title("Cross Amp sum of all baselines, block = " + str(iblock))
        plt.show(block=False)

        
        '''
        print("Cleaning block ", iblock)
        acm, ccm, casm, timem = rfi_st.IQRM_filter(block, mask_time = True)
        '''
        acm_arr = bl2array(convert_acm_to_complx_bl(acm, block))
        ccm_arr = bl2array(convert_ccm_to_complx_bl(ccm, block))
        casm_arr = bl2array(convert_casm_to_complx_bl(casm, block))
        timem_arr = bl2array(convert_timem_to_complx_bl(timem, block))

        fig2, axes2 = plt.subplots(nrows = nrows, ncols = ncols)
        axes2 = axes2.flatten()
        for bl_idx in range(len(block_arr)):
            which_bl = bl_idx + 20 * 22
            if bl_idx > nbl_to_plot:
                break
            axes2[bl_idx].imshow(block_arr[which_bl].real[:, ipol, :, 0], aspect='auto', interpolation='None')
            axes2[bl_idx].imshow(acm_arr[which_bl][:, ipol, :].real, aspect='auto', interpolation='None', cmap='autumn')
            axes2[bl_idx].imshow(ccm_arr[which_bl][:, ipol, :].real, aspect='auto', interpolation='None', cmap='Wistia')
            axes2[bl_idx].imshow(casm_arr[which_bl][:, ipol, :].real, aspect='auto', interpolation='None', cmap='winter')
            axes2[bl_idx].imshow(timem_arr[which_bl][:, ipol, :].real, aspect='auto', interpolation='None', cmap='Greys_r')
            blid = list(block.keys()) [which_bl]
            ant1, ant2 = bl2ant(blid)
            axes2[bl_idx].set_title(str(ant1) + "," +str(ant2))

        print(np.nansum(acm_arr.real) / block_arr.size, np.nansum(ccm_arr.real)/block_arr.size, np.nansum(casm_arr).real/block_arr.size)
        plt.title("Block = " + str(iblock))
        plt.show(block = False)
        _ = input()
        plt.close('all')
        '''
        print("Editing vis for block = ", iblock)
        hdulist.edit_vis(iblock, block, nt)

    hdulist.write_vis_out(outname="/home/gup037/tmp/modified_new.uvfits", overwrite=True)


if __name__ == '__main__':
    main()
