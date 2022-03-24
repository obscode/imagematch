'''Some FITS-related routines that are used again and again. These are for
simple CCD frames (one PrimaryHDU in and HDUList)'''

from astropy.io import fits
import gc

def qdump(filename, data, header=None, extras=[]):
   if header is None:
      header = fits.PrimaryHDU(data)
      for (key,val) in extras:
         header.header[key] = val
      header.writeto(filename, overwrite=True)
   else:
      if type(header) is str:
         hfts = fits.open(header, memmap=False)
         header = hfts[0].header
         for (key,val) in extras:
            header[key] = val
      fits.writeto(filename, data, header, overwrite=True)

def qload(infile, hdu=0):
   '''Safely load the header and data from a FITS file.'''
   with fits.open(infile, memmap=False) as hdul:
      data = hdul[hdu].data.copy()
      header = hdul[hdu].header.copy()
      del hdul
      gc.collect()
   return header,data


def copyFits(inp):
   '''Copy the FITS header and data and return the copy.'''
   newhdr = inp[0].header.copy()
   newdata = inp[0].data.copy()
   newphdu = fits.PrimaryHDU(newdata, header=newhdr)
   newhdul = fits.HDUList([newphdu])
   return newhdul


