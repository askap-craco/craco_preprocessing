import numpy as np
import matplotlib.pyplot as plt
import argparse

from craco_fits_modifier import CracoFitsReader, bl2ant, bl2array
from rfi_strategy import RFI_cleaner

from astropy.io import fits

def main(args):
    fname = args.inputfile
    f = fits.open(fname)
    hdulist = CracoFitsReader(f)

    rfi_st = RFI_cleaner(hdulist)

    nt = 16
    ipol = 0
    blocker = hdulist.time_blocks(nt )

    masking_depth = {'mask_autos': False,
                     'mask_time': False,
                     'mask_corrs': False,
                     'mask_cas': False}
    for ii, key in enumerate(masking_depth.keys()):
        if ii < args.cleaning_depth:
            masking_depth[key] = True
    
    print("Masking depth = {}".format(masking_depth))
    outname = args.outname
    if args.outname is None:
        outname = args.inputfile + ".cleaned"
    for iblock, block in enumerate(blocker):
        print("Cleaning block ", iblock)
        acm, ccm, casm, timem = rfi_st.IQRM_filter(block, **masking_depth)
        print("Editing vis for block = ", iblock)
        hdulist.edit_vis(iblock, block, nt)
        print("Done")

    hdulist.write_vis_out(outname=outname, overwrite=True)


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("inputfile", type=str, help="Path to the input UV fits file to clean")
    a.add_argument("-cleaning_depth", type=int, help="Depth level to which to clean the data (options= 1, 2, 3, 4 = Autos, Autos + Time, Autos + Time + Crosses, Autos + Time + Crosses + CAS) [def = 4]", default=4)
    a.add_argument("-outname", type=str, help="Path to the output file (will be overwritten if it exists) [def = with a '.cleaned' extension]", default=None)
    args = a.parse_args()
    main(args)
