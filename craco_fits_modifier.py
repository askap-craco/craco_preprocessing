from configparser import Interpolation
import numpy as np 
#from astropy.io import fits
import logging, warnings
from astropy import units as u
from astropy.time import Time
import matplotlib.pyplot as plt


def ants2bl(a1, a2):
    return a1*256 + a2
    
def bl2ant(bl):
    '''
    Convert baseline to antena numbers according to UV fits convention
    Antenna numbers start at 1 and:

    baseline = 256*ant1 + ant2

    :see: http://parac.eu/AIPSMEM117.pdf

    :returns: (ant1, ant2) as integers

    >>> bl2ant(256*1 + 2)
    (1, 2)

    >> bl2ant(256*7 + 12)
    (7, 12)
    '''
    ibl = int(bl)
    a1 = ibl // 256
    a2 = ibl % 256

    assert a1 >= 1
    assert a2 >= 1

    return (a1, a2)


def bl2array(baselines, sort=True):
    '''
    Converts baseline dictionary into an array sorted by baseline id
    :returns: np array of shape [nbl, nf, nt]
    '''
    blids = baselines.keys()
    if sorted:
        blids = sorted(baselines.keys())
    nbl = len(blids)
    tfshape = baselines[blids[0]].shape
    fullshape = [nbl]
    fullshape.extend(tfshape)

    d = np.zeros(fullshape, dtype=np.complex64)
    for idx, blid in enumerate(blids):

        d[idx, ...] = baselines[blid]

    return d

def get_bl_length(ant_table, bl):
    a1, a2 = bl2ant(bl)
    bl_length = ((ant_table[a1]['STABXYZ'] - ant_table[a2]['STABXYZ'])**2).sum()**0.5
    return bl_length

def time_blocks(vis, nt, flagant=[], flag_autos=True, return_arr = False, read_weights = False):
    '''
    Generator that returns nt time blocks of the given visiblity table


    :returns: Dictionary, keyed by baseline ID, value = np.array size (nchan, npol, nt) dtype=complex64
    :see: bl2ant()
    '''

    nrows = vis.size
    inshape = vis[0].data.shape
    nchan = inshape[-3]
    npol = inshape[-2]
    nweight = 2
    #logging.info('returning blocks for nrows=%s rows nt=%s visshape=%s', nrows, nt, vis[0].data.shape)
    d = {}
    t = 0
    d0 = vis[0]['DATE']
    first_blid = vis[0]['BASELINE']
    #for irow in xrange(nrows):
    for irow in range(nrows):
        row = vis[irow]
        blid = row['BASELINE']
        a1,a2 = bl2ant(blid)

        #if a1 in flagant or a2 in flagant or (flag_autos and a1 == a2):
        #    continue

        #logging.(irow, blid, bl2ant(blid), row['DATE'], d0, t)
        if row['DATE'] > d0 or (blid == first_blid and irow != 0): # integration finifhsed when we see first blid again. date doesnt have enough resolution
            t += 1
            tdiff = row['DATE'] - d0
            d0 = row['DATE']
            #logging.debug('Time change or baseline change irow=%d, len(d)=%d t=%d d0=%s rowdate=%s tdiff=%0.2f millisec', irow, len(d), t, d0, row['DATE'], tdiff*86400*1e3)

            if t == nt:
                #logging.debug('Returning block irow=%d, len(d)=%d t=%d d0=%s rowdate=%s tdiff=%0.2f millisec', irow, len(d), t, d0, row['DATE'], tdiff*86400*1e3)
                if return_arr:
                    yield bl2array(d)
                else:
                    yield d
                d = {}
                t = 0


        if blid not in list(d.keys()):
            d[blid] = np.zeros((nchan, npol, nt, nweight), dtype=np.complex64)
            d[blid][:, :, :, 1].real = 1       #Set the weight to one for all samples

        d[blid][:, :, t, 0].real = row.data[...,0]
        d[blid][:, :, t, 0].imag = row.data[...,1]
        if read_weights:
            d[blid][:, :, t, 1] = row.data[..., 2]

    if len(d) > 0:
        if t < nt - 1:
            warnings.warn('Final integration only contained t={} of nt={} samples len(d)={} nrows={}'.format(t, nt, len(d), nrows))
        if return_arr:
            print("Yielding an array")
            yield bl2array(d)
        else:
            print("Yielding a dict")
            yield d


def edit_vis_with_time_blocks(vis, d, nt, startrow = 0):
    nrows = vis.size

    #logging.info('returning blocks for nrows=%s rows nt=%s visshape=%s', nrows, nt, vis[0].data.shape)
    t = 0
    d0 = vis[0]['DATE']
    first_blid = vis[0]['BASELINE']
    #for irow in xrange(nrows):
    for irow in range(nrows):
        irow = startrow + irow
        if irow > nrows:
            break
        row = vis[irow]
        blid = row['BASELINE']
        
        if row['DATE'] > d0 or (blid == first_blid and irow != 0): # integration finifhsed when we see first blid again. date doesnt have enough resolution
            t += 1
            d0 = row['DATE']
            
            if t == nt:
                break
    
        row.data[..., 0] = d[blid][:, :, t, 0].real
        row.data[..., 1] = d[blid][:, :, t, 0].imag
        row.data[..., 2] = d[blid][:, :, t, 1].real
    
    return irow

def get_freqs(hdul):
    '''
    Returns a numpy array of channel frequencies in Hz from a UVFITS HDU list

    :returns: Np array length NCHAN
    '''
    
    hdr = hdul[0].header
    fch1 = hdr['CRVAL4']
    foff = hdr['CDELT4']
    ch1 = hdr['CRPIX4']
    #assert ch1 == 1.0, f'Unexpected initial frequency: {ch1}'
    assert foff > 0, 'cant handle negative frequencies anywhere athe moment foff=%f' % foff
    vis = hdul[0].data
    nchan = vis[0].data.shape[-3]
    freqs = (np.arange(nchan, dtype=np.float) - ch1)*foff + fch1 # Hz

    return freqs


class CracoFitsReader:

    def __init__(self, hdulist, max_nbl=None):
        self.hdulist = hdulist
        self.max_nbl = max_nbl
        self.flagant = []
        self.ignore_autos = False
        self.in_data_shape = self.vis[0].data.squeeze().shape
        self.in_group_shape = self.vis[0].data.shape

    def set_flagants(self, flagant):
        '''
        set list of 1-based antennas to flag
        '''
        self.flagant = flagant
        return self

    @property
    def nf(self):
        for item in self.header.items():
            if item[1] == "FREQ":
                key_idx =  int(item[0].strip("CTYPE"))
                key = f"NAXIS{key_idx}"
                return self.header[key]
        raise RuntimeError("Could not figure out which axis is the frequency axis")    
            

    @property
    def npol(self):
        for item in self.header.items():
            if item[1] == "STOKES":
                key_idx = int(item[0].strip("CTYPE"))
                key = f"NAXIS{key_idx}"
                return self.header[key]
        raise RuntimeError("Could not figure out which axis is the polarisation axis")

    @property
    def header(self):
        return self.hdulist[0].header

    @property
    def vis(self):
        '''Returns visbility table
        '''
        return self.hdulist[0].data

    @property
    def ant_table(self):
        '''Returns the antenna table
        '''
        return self.hdulist[2].data

    @property
    def start_date(self):
        row = self.vis[0]
        d0 = row['DATE']
        try:
            d0 += row['_DATE'] # FITS standard says add these two columns together
        except KeyError:
            pass

        return d0

    @property
    def channel_frequencies(self):
        return get_freqs(self.hdulist)

    @property
    def baselines(self):
        '''
        Returns all data from first integration
        Doesn't include baselines containing flagant
        
        :returns: dictionary, keyed by baseline ID of all basesline data with a timestamp
        equal to the first timestamp in the file
        '''
            
        d0 = self.start_date
        baselines = {}
        vis = self.vis
        for i in range(self.vis.size):
            row = vis[i]
            blid = row['BASELINE']
            a1, a2 = bl2ant(blid)
            if a1 in self.flagant or a2 in self.flagant:
                continue

            if self.ignore_autos and a1 == a2:
                continue
            
            baselines[blid] = row
            if row['DATE'] != d0 or (self.max_nbl is not None and i > self.max_nbl):
                break

        return baselines

    @property
    def nbl(self):
        '''
        Reads the first integration and counts how many baselines are there
        '''
        nbl = 0
        blids_seen = []
        first_blid = self.vis[0]['BASELINE']
        for irow in range(self.vis.size):
            row = self.vis[irow]
            blid = row['BASELINE']
            if blid not in blids_seen:
                blids_seen.append(blid)
                nbl += 1
            else:
                #logging.debug(f"Hit the first baseline again, breaking now. nbl is {nbl}")
                break

        return nbl

    def overwrite_data(self, data):
        self.hdulist[0].data = data.copy()

    def writenew(self, outname, ignoreexisting=True):
        self.hdulist.writeto(outname, overwrite=ignoreexisting)

    def get_max_uv(self):
        ''' 
        Return the largest absolute values of UU and VV in lambdas
        '''
        fmax = self.channel_frequencies.max()
        baselines = self.baselines
        ulam_max = max([abs(bldata['UU'])*fmax for bldata in list(baselines.values())])
        vlam_max = max([abs(bldata['VV'])*fmax for bldata in list(baselines.values())])
        return (ulam_max, vlam_max)

    def edit_vis(self, iblock, block, nt):
        startrow = iblock * nt * self.nbl
        edit_vis_with_time_blocks(self.vis, block, nt, startrow)

    def time_blocks(self, nt, return_arr = False, read_weights = False):
        '''
        Returns a sequence of baseline data in blocks of nt
        '''
        # WARNING TODO: ONLY RETURN BASELINES THAT HAVE BEEN RETURNED in .baselines
        # IF max_nbl has been set
        return time_blocks(self.vis, nt, self.flagant, self.ignore_autos, return_arr=return_arr, read_weights=read_weights)


    def plot_block(self, iblock, block, pol='I', apply_weights = True):
        if apply_weights:
            cblock = block[..., 0] * block[..., 1].real
        else:
            cblock = block[..., 0].copy()

        if pol.upper() == 'I':
            bdata = cblock.mean(axis = 2)
            bmask = block[..., 1].real.mean(axis=2)
        elif pol.upper() == 'XX':
            bdata = cblock[:, :, 0, :]
            bmask = block[..., 1].real[:, :, 0, :]
        elif pol.upper() == 'YY':
            bdata = cblock[:, :, 1, :]
            bmask = block[..., 1].real[:, :, 1, :]
        else:
            raise ValueError("Unknown pol type requested -> valied types are [I / XX / YY]")
        
        #data = np.abs(bdata).mean(axis=0)        #Sum across baselines
        fig = plt.figure()
        ax1 = fig.add_subplot(321)
        ax1.imshow(bdata.real.mean(axis=0), aspect='auto', interpolation='None')
        if apply_weights:
            ax1.imshow((1 - bmask).mean(axis=0), aspect='auto', interpolation='None', cmap='Greys_r')
        ax1.set_title("CAS")
        ax1.set_xlabel("Time [samp]")
        ax1.set_ylabel("Freq [chan]")


        ax1i = fig.add_subplot(322)
        ax1i.imshow(bdata.imag.mean(axis=0), aspect='auto', interpolation='None')
        if apply_weights:
            ax1i.imshow((1 - bmask).mean(axis=0), aspect='auto', interpolation='None', cmap='Greys_r')
        ax1i.set_title("CAS")
        ax1i.set_xlabel("Time [samp]")
        ax1i.set_ylabel("Freq [chan]")
        
        ax2 = fig.add_subplot(323)
        ax2.imshow(np.abs(bdata).mean(axis=-1), aspect='auto', interpolation='None')
        if apply_weights:
            ax2.imshow((1 - bmask).mean(axis=-1), aspect='auto', interpolation='None', cmap='Greys_r')
        ax2.set_title("BL amps")
        ax2.set_xlabel("Freq [chan]")
        ax2.set_ylabel("BL index")

        ax3 = fig.add_subplot(325)
        ax3.imshow(np.abs(bdata).mean(axis=1), aspect='auto', interpolation='None')
        if apply_weights:
            ax3.imshow((1 - bmask).mean(axis=1), aspect='auto', interpolation='None', cmap='Greys_r')
        ax3.set_title("BL vs Freq")
        ax3.set_xlabel("Time [samp]")
        ax3.set_ylabel("BL index")

        fig.suptitle(f"Block = {iblock}")

        f2 = plt.figure()
        np.random.seed(777)
        bls_to_plot = (np.random.uniform(0, 1, 9) * len(bdata)).astype('int')
        for ii in range(9):
            axi = f2.add_subplot(3, 3, ii+1)
            axi.imshow(np.abs(bdata[bls_to_plot[ii], :, :]), aspect='auto', interpolation='None')
            axi.imshow((1 - bmask[bls_to_plot[ii], :, :]), aspect='auto', interpolation='None', cmap='Greys_r')
            axi.set_title(f"BL {ii}")

        plt.show()

    def get_tstart(self):
        '''
        return tstart as astropy Time from header otherwise first row of fits table
        '''
        f = self
        if 'DATE-OBS' in f.header:
            tstart = Time(f.header['DATE-OBS'], format='isot', scale='utc')
        else:
            jdfloat = f.start_date
            tstart = Time(jdfloat, format='jd', scale='utc')
        
        return tstart
        
    def get_target_position(self, targidx=0):
        '''
        return (ra,dec) degrees from header if available, otherwise source table
        '''
        f = self
        if 'OBSRA' in f.header:
            ra = f.header['OBSRA'] * u.degree
            dec = f.header['OBSDEC'] * u.degree
            #log.info('Got radec=(%s/%s) from OBSRA header', ra, dec)
        else:
            source_table = f.hdulist[3].data
            assert len(source_table)==1, f'Dont yet support multiple source files: {len(source_table)}'
            row = source_table[targidx]
            src = row['SOURCE']
            ra = row['RAEPO']*u.degree
            dec = row['DECEPO']*u.degree
            #log.info('Got radec=(%s/%s) from source table for %s', ra, dec, src)

        return (ra, dec)
    
    def baseline_length_order(self, baseline_list):
        baseline_lengths = []
        for ibl in baseline_list:
            bl_length = get_bl_length(self.ant_table, ibl)
            baseline_lengths.append(bl_length)
       
        print(bl_length, baseline_lengths, np.argsort(baseline_lengths), type(np.argsort(baseline_lengths)))
        sorted_baselines = np.array(baseline_list)[np.argsort(baseline_lengths)]
        return sorted_baselines

    def write_vis_out(self, outname, overwrite=False):
        '''
        Writes out the vis data (potentially modified) back as a fits file
        '''
        print(f"Writing to the file = {outname}")
        self.hdulist.writeto(outname, overwrite = overwrite)

    def close(self):
        return self.hdulist.close()
