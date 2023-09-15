from astropy.io import fits
import numpy as np
import os

def write_hyperdrive_sols(sols,templatefile,outfile='fit_hyperdrive_solutions.fits'):
    """ 
    Writes new (fitted) solutions into hyperdrive solutions fits format
    in style of template
    """
    template = fits.open(templatefile)
    # Only consider first timeblock
    template[1].data[0,:,:,::2] = np.real(sols)
    template[1].data[0,:,:,1::2] = np.imag(sols)

    template.writeto(outfile,overwrite="True")

def read_hyperdrive_sols(solsfile):
    """
    Reads hyperdrive solutions from fits file into numpy array
    """

    f = fits.open(solsfile)
    data = f[1].data
    # Only looking at the first timeblock.
    i_timeblock = 0
    sols = data[i_timeblock, :, :, ::2] + data[i_timeblock, :, :, 1::2] * 1j
    
    return sols

def average_hyperdrive_solutions(cal_list,flagged_tiles,verbose=False):
    """
    Reads a list of hyperdrive solutions from a text file and return a single
    average set of solutions
    """

    infile1 = open(cal_list) # text file with path to hyperdrive solution fits files 
    sols_sum1 = None
    n_sols = 0
    sols_list = []

    for line in infile1:
        if(verbose):
            print(line)
        sols = read_hyperdrive_sols(line.split()[0])
        for i in range(np.shape(sols)[0]):
            if(np.all(np.isnan(sols[i,:,0]))):
                if(i not in flagged_tiles):
                    flagged_tiles.append(i)
        n_sols += 1
        if(sols_sum1 is None):
            sols_sum1 = np.copy(sols)
        else:
            sols_sum1 += sols
    sols_list.append(sols)    

    sols_sum1 /= n_sols

    return sols_sum1

def get_cables_from_metafits(metafits):
    """
    Returns dictionary for mapping antenna numbers to cable flavours
    """
    
    meta_file = fits.open(metafits)
    cables = meta_file[1].data['Flavors']
    antennas = meta_file[1].data['Antenna']

    return dict(zip(antennas,cables))
    
# Load previously written auto data from text files

def load_autos_data(obsid,tile_n,auto_dir):
    """
    Loads XX/YY autocorrelations from previously written text files
    """
    
    xx_autos = []
    yy_autos = []
    
    with open(f'{auto_dir}{obsid}/{obsid}_{tile_n}_XX_auto.txt') as xx_file:
        for line in xx_file:
            xx_autos.append(line.split())
        
    with open(f'{auto_dir}{obsid}/{obsid}_{tile_n}_YY_auto.txt') as xx_file:
        for line in xx_file:
            yy_autos.append(line.split())
        
    return np.array(np.ravel(xx_autos),dtype=float), np.array(np.ravel(yy_autos),dtype=float)

def get_present_autos(obsid,auto_dir,n_ants=128):
    """
    Returns list of autos present in auto_dir (ie unflagged tiles)
    """
    
    present_autos = []
    
    for i in range(n_ants):
        if(os.path.exists(f'{auto_dir}{obsid}/{obsid}_{i}_XX_auto.txt')):
            present_autos.append(i)
    
    return present_autos

def load_all_autos(obsid,auto_dir,present_autos,xvals,n_ants=128):
    """
    Load XX/YY auto-correlations from text files
    """

    all_autos = []

    for i in range(n_ants):
        if(i in present_autos):
            tile_xx, tile_yy = load_autos_data(obsid,i,auto_dir=auto_dir)
            all_autos.append([tile_xx,tile_yy])
        else:
            all_autos.append([None,None])

    return all_autos    

        

