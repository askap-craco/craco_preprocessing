import numpy as np
import matplotlib.pyplot as plt
import argparse

from craco_fits_modifier import CracoFitsReader, bl2ant, bl2array

from astropy.io import fits

def main(args):
    fname = args.inputfile
    f = fits.open(fname)
    hdulist = CracoFitsReader(f)

    nt = args.nt
    blocker = hdulist.time_blocks(nt, return_arr=True, read_weights=args.apply_masks)

    for iblock, block in enumerate(blocker):
        print("Plotting block ", iblock, block.shape)
        #print(block[:, :, :, :, 1])
        #continue
        hdulist.plot_block(iblock, block, apply_weights=args.apply_masks)

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("inputfile", type=str, help="Path to the input UV fits file to clean")
    a.add_argument("-apply_masks", action='store_true', help="Apply masks? (def = False)", default=False)
    a.add_argument("-nt", type=int, help="nt per block to plot (def: 256)", default=256)
    args = a.parse_args()
    main(args)
