''' imageMatching.py

   Software to optimally match the PSF of one image (a template) to another
   (the image) and then subtract them.

   AUTHOR:  Dan Kelson
   
   10/10/2005:  Added a bit of code to let the user specify masks in the
                header, allowing them to throw out bad stars, ghosts, etc.
   02/21/2006:  We can now input a noise map (using -sig) so that the same
                convolution can be done to an image's noise map.
   03/02/2010:  Added WCS code to allow for easier use.  Now, if both the image
                and template have WCS in their headers, we can take out a lot of
                the logic that had a tendency to fail with different instruments
                and relative rotations.
  02/13/2020:  Ack, 10 years later!  Trying to port this to more common
                modules and python 3
'''
import gc
import time,os
from .fitsutils import qdump, qload
from .VTKHelperFunctions import *  # all start with VTK, so this is okay :)
from .ReadSex import readsex
from astropy.io import fits as FITS
from astropy.io import ascii
from astropy.stats import gaussian_fwhm_to_sigma as f2s
import numpy.fft as fft
import numpy as np
from .GaussianBG import GaussianBG
from .basis import abasis,mbasis,svdfit
from .npextras import bwt, divz, between
from .fitsutils import qdump
from numpy import linalg
from scipy.ndimage import map_coordinates
from astropy.convolution import convolve_fft, Gaussian2DKernel
import sys
debug = 1
try:
   from . import Picker
   from matplotlib import pyplot as plt
   from matplotlib import rcParams
   from astropy.visualization import simple_norm
except:
   Picker = None
   plt = None
try:
   from astropy import wcs
except:
   wcs = None
from astropy.coordinates import SkyCoord
from astropy import units as u

try:
   import fit_psf
except:
   fit_psf = None

def singular_value_decomposition(A, full_matrices=0):
   return linalg.svd(A, full_matrices)

NA = np.newaxis
def QuickShift(i,x,y):
   '''Shift an image, i, by amount x and y.  Note 'x' is in the sense of the
   2nd index (image coordinates)'''
   s = VTKImageShift(i,x,y,interp=0,numret=1)
   return s

def ndf(nord):
   '''Returns the number of degrees of freedom for nord-order coordinate 
   transformation.'''
   if nord == -1:
      Ncoeff = 2
   elif nord == 0:
      # rotation:  xoff, yoff, scale, rot
      Ncoeff = 4
   else:
      Ncoeff = len(np.ravel(abasis(nord, np.array([0]), np.array([0]))))
   return Ncoeff

def headerVal(val, hdr, convert=float):
   '''If val is a header keyword, return it from the header.'''
   if type(val) is type(""):
      retval = hdr.getval(val)
      if retval == "N/A":
         raise ValueError("You gave me a {} header keyword, but I can't "
                          "find it in the header".format(val))
   else:
      retval = val
   if convert is not None:
      return convert(retval)
   else:
      return retval

class Point:
   ''' A class that corresponds to a point on the image.  It can either be in
   pixel or world coordinates, and there are member functions to transform them
   when a WCS is present.'''

   def __init__(self, x, y, image=None):
      self.units = None
      if x is not None:
         if type(x) is type(""):
            if x[-1] == 'd':
               self.x = float(x[:-1])
               self.units = 'd'
            else:
               self.x = float(x)
               self.units = 'p'
         else:
            self.x = float(x)
            self.units = 'p'
      else:
         self.x = None

      if y is not None:
         if type(y) is type(""):
            if y[-1] == 'd':
               if self.units == "p":
                  raise ValueError("Can't mix units for X and Y")
               self.y = float(y[:-1])
               self.units = 'd'
            else:
               if self.units == 'd':
                  raise ValueError("Can't mix units for X and Y")
               self.y = float(y)
               self.units = 'p'
         else:
            if self.units == 'd':
               raise ValueError("Can't mix units for X and Y")
            self.y = float(y)
            self.units = 'p'
      if self.x is None and self.y is None:
         raise ValueError("Error, one of x or y must be specified")
      self.image = image

   def topixel(self):
      if self.units == 'p':
         return (self.x, self.y)
      else:
         if self.image is None or self.image.wcs is None:
            raise AttributeError("Must have A WCS before I can transform to "
                                 "pixels")
         if self.x is None:
            x = self.image.naxis1/2
         else:
            x = self.x
         if self.y is None:
            y = self.image.naxis2/2
         else:
            y = self.y
         i,j = self.image.wcs.wcs_world2pix(x, y, 0)
         if self.x is None:  i = None
         if self.y is None:  j = None
         return(i,j)

   def toworld(self):
      if self.units == 'd':
         return (self.x, self.y)
      else:
         if self.image is None or self.image.wcs is None:
            raise AttributeError("Must have A WCS before I can transform to "
                                 "Ra/DEC")
         if self.x is None:
            x = self.image.wcs.wcs.crval[0]
         else:
            x = self.x
         if self.y is None:
            y = self.image.wcs.wcs.crval[1]
         else:
            y = self.y
         x,y = self.image.wcs.wcs_pix2world(x, y, 0)
         if self.x is None:  x = None
         if self.y is None:  y = None

         return(x,y)

class Mask:
   '''An object that defines a mask for an image.'''

   def __init__(self, image):
      '''[image] is an Observation class.'''
      self.image = image
      self.llcs = []    # lower-left corners... point classes
      self.urcs = []    # upper-right corners... point classes
      self.sides = []   # 0 --> mask out interior  1 --> mask out exterior

   def __repr__(self):
      st = "Masks:\n"
      for i in range(len(self.llcs)):
         i0,j0 = self.llcs[i].topixel()
         i1,j1 = self.urcs[i].topixel()
         if self.sides[i] == 0:
            st += "\t[%.1f:%.1f,%.1f:%.1f] = False\n" % (i0,i1,j0,j1)
         else:
            st += "\t[%.1f:%.1f,%.1f:%.1f] = True\n" % (i0,i1,j0,j1)
      return st

   def add_mask(self, x0, y0, x1, y1, sense='inside'):
      '''Add a box mask.  If [sense] is 'inside' mask the interior pixels, 
      otherwise, mask the outside pixels.'''
      if x1 < x0:  x0,x1 = x1,x0
      if y1 < y0:  y0,y1 = y1,y0
      try:
         p0 = Point(x0,y0,self.image)
         p1 = Point(x1,y1,self.image)
      except:
         raise ValueError("Problem with mask %s" % (str(x0,y0,x1,y1)))
      self.llcs.append(p0)
      self.urcs.append(p1)
      if sense.lower() == 'inside':
         self.sides.append(0)
      else:
         self.sides.append(1)

   def get_mask(self):
      '''Based on the current masks, return the end result.'''
      ox = getattr(self.image, 'ox', None)
      oy = getattr(self.image, 'oy', None)
      if ox is None or oy is None:
         oy,ox = np.indices((self.image.naxis2,self.image.naxis1),np.float32)
      mask = np.ones((self.image.naxis2,self.image.naxis1), np.float32)
      for i in range(len(self.llcs)):
         i0,j0 = self.llcs[i].topixel()
         i1,j1 = self.urcs[i].topixel()
         if i0 > i1:  i0,i1 = i1,i0
         if j0 > j1:  j0,j1 = j1,j0
         if self.sides[i] == 1:
            if i0 and i1:
               mask = VTKMultiply(mask, between(ox,i0,i1).astype(np.float32))
            if j0 and j1:
               mask = VTKMultiply(mask, between(oy,j0,j1).astype(np.float32))
         else:
            outsidex = VTKOr(np.less(ox,i0).astype(np.float32),
                             np.greater(ox,i1).astype(np.float32))
            outsidey = VTKOr(np.less(oy,j0).astype(np.float32),
                             np.greater(oy,j1).astype(np.float32))
            mask = VTKMultiply(mask, VTKOr(outsidex,outsidey))
      return mask

   def flag_pixels(self, x, y):
      ''' Given x,y points, determined if they are masked (False) or not masked
      (True).'''
      gids = np.ones(x.shape, bool)
      for i in range(len(self.llcs)):
         i0,j0 = self.llcs[i].topixel()
         i1,j1 = self.urcs[i].topixel()
         if i0 > i1:  i0,i1 = i1,i0
         if j0 > j1:  j0,j1 = j1,j0
         if self.sides[i] == 1:
            if i0 and i1:
               gids = gids*between(x, i0, i1)
            if j0 and j1:
               gids = gids*between(y, j0, j1)
         else:
            gids1 = np.less(x, i0) + np.greater(x, i1)
            gids2 = np.less(y, j0) + np.greater(y, j1)
            gids = gids*(gids1+gids2)
      return gids

   def dump_mask(self, file):
      '''Put it in a FITS file.'''
      qdump(file, self.get_mask())


class Observation:
   def __init__(self,image,wt=None,scale='scale',pakey="rotangle",
         saturate=30000, skylev=None, sigsuf=None, wtype="MAP_RMS", 
         mask_file=None, reject=0, snx=None, sny=None, snmaskr=10.0, 
         extra_sufs=[], hdu=0, magmin=None, magmax=None, store_bg=True):

      self.image = image   # The image name
      self.hdu = hdu
      self.magmin = magmin
      self.magmax = magmax
      self.store_bg = store_bg

      # Keep a logfile
      #self.log_stream = open(self.image.replace('.fits','')+'.log', 'w')
      self.log_stream = None

      #base = os.path.basename(image)
      base = image
      self.catalog = base.replace(".fits",".cat")  # Output catalog of objects
      # Error maps
      if sigsuf: 
         if sigsuf[0] == "@":
            # FITS extension, so use that, but sextractor needs a file.
            self.sigimage = self.image.replace('.fits','_sigma.fits')
            try:
               sigheader,sigdata = qload(self.image, hdu=sigsuf[1:])
               qdump(self.sigimage, sigdata, self.image)
            except:
               self.sigimage = None
         else:
            self.sigimage = self.image.replace(".fits",sigsuf+".fits")
      else: 
         self.sigimage = None
      # Any extra stuff
      if extra_sufs:
         self.extra_maps = [self.image.replace(".fits", suf+".fits") \
               for suf in extra_sufs]
      else:
         self.extra_maps = None

      # Type of weight map (sextractor)
      self.wtype=wtype
      # Name of host galaxy template image
      self.template = base.replace(".fits","temp.fits")
      # Name of rectified (but not convolved) template
      self.rectemp = base.replace(".fits","master.fits")
      # name of rectified (but not convolved) template error map
      self.recsigma = base.replace(".fits","master_sigma.fits")
      # name of SN image. Note that if -R, this may be convolved with kernel
      self.SN = base.replace(".fits","SN.fits")
      # Name of image for kernel basis functions
      self.imbases = base.replace(".fits","bases.fits")
      # Name of output image with kernel solutions over the image plane
      self.psfgrid = base.replace(".fits","grid.fits")
      # Name of output image with the final mask used
      self.masked = base.replace(".fits","mask.fits")
      # Name of weight FITS image
      self.weight = base.replace(".fits","weight.fits")
      # Name of output difference image
      self.difference = base.replace(".fits","diff.fits") # difference image
      # Name of difference image with positive weight
      self.resids = base.replace(".fits","resids.fits")
      # Name of optinal (400x400) output cutout around SN
      self.SNpostage = base.replace(".fits","SNpost.fits")
      # Name of optinal (400x400) output cutout around difference
      self.diffpostage = base.replace(".fits","diffpost.fits")
      # Output database of object matches
      self.db = base.replace(".fits", "db.txt")
      # Output of relative eigenvalues
      self.rdv = base.replace(".fits","rdv.fits")
      # Output segmentation map from source extractor
      self.segmap=base.replace(".fits","seg.fits")


      self.wt = wt                      # Weights
      self.bg = 0.0                     # Background
      self.saturate = saturate          # saturation value in image

      self.reject = reject              # completely reject saturated objects?

      # Try to figure some stuff out from the header
      #f = FITS.open(self.image, memmap=False)
      header,d = qload(self.image, hdu)
      self.epoch = header.get("epoch", "N/A")
      try:
         self.exptime = float(header.get("exptime", 1.0))
      except:
         self.exptime = 1.0
      if self.epoch == "N/A": self.epoch = -99
      if type(scale) is type(""):
         self.scale = header.get(scale, "N/A")
         if self.scale == 'N/A':
            raise AttributeError("Scale keyword not found: {}".format(scale))
      else:
         self.scale = scale
      if skylev is not None:
         self.skylev = header.get(skylev, "N/A")
         if self.skylev == 'N/A':
            raise AttributeError("Sky level keyword not found: {}".format(\
               scale))
         self.saturate = self.saturate - self.skylev
      else:
         self.skylev = 0

      self.starlist = None

      # SN position
      if snx is not None and sny is not None:
         try:
            self.snpos = Point(snx, sny, self)
         except:
            snx = header.get(snx, "N/A")
            sny = header.get(sny, "N/A")
            if snx == 'N/A' or sny == 'N/A':
               raise AttributeError('SN position keywords not found')
            self.snpos = Point(snx,sny, self)
      else:
         self.snpos = None
      self.snmaskr = snmaskr

      self.crowd = False

      self.mask = Mask(self)
      # Added by CRB:  Allow the user to mask out data.
      if mask_file is not None:
         mf = open(mask_file,'r')
         for line in mf.readlines():
            x0,y0,x1,y1 = line.split()
            self.mask.add_mask(x0,y0,x1,y1, sense='inside')
            self.log('Adding mask: %s %s %s %s' % (x0,y0,x1,y1))
         mf.close()
      self.pa = header.get(pakey, "N/A")
      if self.pa == "N/A": self.pa = 0.0
      #f.close()
      self.data = None
      self.nans = None   # where the data are NaN
      self.master = None

   def log(self, message):
      '''Quick function to print log information to the screen and also keep 
      a copy in a specified log file.'''
      print(message)
      if self.log_stream is not None:
         self.log_stream.write(str(message)+'\n')

   def __repr__(self):
      '''Give the user a nice representation of the object.'''
      return self.image

   def sex(self):
      '''This function runs sextractor on the image, finding images to be used
      for rectification and solving for the kernel.'''

      if self.sexdir[-1] != "/": self.sexdir += "/"
      self.sexcom = [self.sexcmd,self.image]
      self.sexcom += ["-c "+self.sexdir+"default.sex"]
      self.sexcom += ["-PARAMETERS_NAME "+self.sexdir+"default.param"]
      if self.sigimage: self.sexcom += ["-WEIGHT_IMAGE "+self.sigimage]
      if self.sigimage: self.sexcom += ["-WEIGHT_TYPE %s" % self.wtype]
      self.sexcom += ["-DETECT_MINAREA 5"]
      self.sexcom += ["-DETECT_THRESH "+str(self.thresh)]
      self.sexcom += ["-ANALYSIS_THRESH "+str(self.thresh)]
      self.sexcom += ["-PIXEL_SCALE "+str(self.scale)]
      self.sexcom += ["-SEEING_FWHM "+str(0.8/self.scale)]
      self.sexcom += ["-CATALOG_NAME",self.catalog]
      self.sexcom += ["-CHECKIMAGE_TYPE SEGMENTATION"]
      self.sexcom += ["-CHECKIMAGE_NAME "+self.segmap]
      self.sexcom += ["-FILTER_NAME "+self.sexdir+"gauss_3.0_5x5.conv"]
      self.sexcom += ["-STARNNW_NAME "+self.sexdir+"default.nnw"]
      self.sexcom += ["-SATUR_LEVEL "+str(self.saturate)]
      self.sexcom += ["-DEBLEND_MINCONT 1"]
      self.sexcom += ["-VERBOSE_TYPE QUIET"]
      if self.zp is not None:
         self.sexcom += ["-MAG_ZEROPOINT " + str(self.zp)]
      self.sexcom = ' '.join(self.sexcom)
      if self.verb:
         self.log("Running {}".format(self.sexcom))
      os.system(self.sexcom)

   def readcat(self, magcat=None, racol='RA', deccol='DEC', magcol='rmag'):
      '''Read the SExtractor output.'''
      if self.do_sex:  self.sex()
      self.sexdata = readsex(self.catalog)[0]
      self.sexdata['CLASS_STAR'] = np.array(self.sexdata['CLASS_STAR'])
      self.sexdata['NUMBER'] = np.array(self.sexdata['NUMBER'])
      #f = FITS.open(self.segmap, memmap=False)
      #self.seg = f[self.hdu].data
      h,self.seg = qload(self.segmap, self.hdu)
      #f.close()
      # read magnitude data
      if magcat is not None:
         if self.wcs is None:
            raise AttributeError("To use magcat, you need a WCS in the image")
         magtab = ascii.read(magcat, fill_values=[('...',0)])
         ii,jj = self.wcs.wcs_world2pix(magtab[racol], magtab[deccol], 0)
         self.segmags = np.zeros((self.seg.max()+1,))-999
         for i in range(len(ii)):
            if not 0 < jj[i] < self.data.shape[0] or \
               not 0 < ii[i] < self.data.shape[1]:
               continue
            obj_id = self.seg[int(jj[i]),int(ii[i])]
            if obj_id > 0:
               self.segmags[obj_id] = magtab[i][magcol]
      else:
         self.segmags = self.sexdata['MAG_APER']

   def compute(self,nmax=40, min_sep=None):
      '''Based on the data output by SExtractor, build arrays of the distances
      and angles between all the objects in the catalog.  Used later to match
      up the objects to figure out the coordinate transformation from image
      to template.'''
      self.nmax = nmax
      self.min_sep = min_sep
      x = self.sexdata["X_IMAGE"]
      y = self.sexdata["Y_IMAGE"]
      if "MAG_BEST" in self.sexdata: m = self.sexdata["MAG_BEST"]
      elif "MAG_AUTO" in self.sexdata: m = self.sexdata["MAG_AUTO"]

      # FLAGS can be used to filter out bad data
      q = self.sexdata["FLAGS"]
      q = np.array([int(qq) for qq in q])

      # Should make this an option...
      cids = np.less_equal(q,4)
      if not self.crowd:
         uncrowd = np.equal(np.bitwise_and(q,1), 0)
         cids = cids*uncrowd
      x,y,m,q = np.compress(cids,[x,y,m,q],1)
      x = x - 1   # SEXtractor convention is to index from 1
      y = y - 1
      # If the FITS header had masks in the header, then we mask that out too
      #  This is opposite of xwin,ywin:  it's want we want to keep OUT.
      #if self.mask:
      ids = self.mask.flag_pixels(x,y)
      x = x[ids];  y = y[ids];  m = m[ids];  q = q[ids]
      if self.min_sep is not None:
         dx = x[NA,::] - x[::,NA]
         dy = y[NA,::] - y[::,NA]
         ds = np.sqrt(np.power(dx,2) + np.power(dy,2))
         # number of neighbors within min_sep of each point
         num_nn = np.sum(np.less(ds, self.min_sep/self.scale), axis=1) - 1
         ids = np.nonzero(np.equal(num_nn, 0))
         x,y,m,q = np.take([x,y,m,q], ids, 1)

      a = self.pa*np.pi/180
      o = np.argsort(m)[:self.nmax]     # the tne nmax brightest objects
      x,y,m = np.take([x,y,m],o,1)
      dx = x[NA,::] - x[::,NA]
      dy = y[NA,::] - y[::,NA]
      # ds[i,j] = distance between object i and object j in the same frame
      ds = np.sqrt(np.power(dx,2) + np.power(dy,2))
      # da[i,j] = angle of vector connecting object i and j in same frame
      da = np.arctan2(dy,dx)*180/np.pi
      self.x = x
      self.y = y
      self.m = m
      self.dx = dx*self.scale
      self.dy = dy*self.scale
      self.ds = ds*self.scale
      self.da = da
    
   def IDBadObjs(self):
      # reject full objects if requested
      sat_pixels = np.greater(self.data, self.saturate)
      sat_pixels = np.logical_or(sat_pixels, self.nans)
      sat_ids = np.nonzero(np.ravel(sat_pixels))
      sat_objects = np.ravel(self.seg)[sat_ids]
      if self.magmax or self.magmin:
         gids = np.greater(self.segmags, -900)
         if self.magmax: 
            gids = gids*np.less(self.segmags, self.magmax)
         if self.magmin: 
            gids = gids*np.greater(self.segmags, self.magmin)
         gids[0] = False   # zero is no object
         sat_objects = np.concatenate([sat_objects,np.nonzero(~gids)[0]])
      # Now a bit of trickery to find non-repeating set of object numbers:
      # and segmentation numbers from 1, so remove 1 as index
      objects = [obj - 1 for obj in list(set(sat_objects))]
      return objects

   def objmatch(self,nord=0,perr=2,aerr=1.0, angles=[0.0],interactive=0,
         use_db=0, iter=3):
      '''Figure out which objects match up with which other objects in the
      image and template (master) frames.'''
      self.log("Matching "+self.master.image+" to "+self.image)
      if "sexdata" not in dir(self.master):
         self.master.master = self.master
         self.master.readcat()
         self.master.compute(nmax=self.nmax, min_sep=self.min_sep)

      # Now we match objets to objects. There are several ways to do this...
      if use_db:
         # we use a previous run's results
         if os.path.isfile(self.db) and os.path.isfile(self.master.db):
            x0,y0 = np.loadtxt(self.db, unpack=True)
            x1,y1 = np.loadtxt(self.master.db, unpack=True)
         else:
            self.log('Error:  to use -db, the database files must exist')
            sys.exit(1)
         Niter = 1
      elif interactive:
         # We allow the user to pick corresponding points (for difficult cases)
         if Picker is None:
            raise RuntimeError("Sorry, to work interactively, you need "
                               "matplotlib.")
         picker = Picker.FITSpicker(self.image, self.master.image,
               scale1=self.scale, scale2=self.master.scale, equal_scale=True,
               recenter='max', box=10)
               #x1s=self.x, y1s=self.y, x2s=self.master.x, y2s=self.master.y)
         plt.show()
         x0,y0,x1,y1 = picker.get_points()
         Niter = 1
      elif self.starlist is not None:
         # User has provided a list of stars to use. We need WCS to get proper
         # coordinates
         if self.wcs is None or self.master.wcs is None:
            raise RuntimeError("In order to use a star list, you need WCS"
                               "information in both frames")
         if fit_psf is None:
            raise RuntimeError("In order to use a star list, you need the "
                               "fit_psf module.")
         x0,x1,y0,y1 = [],[],[],[]
         for xi,yi in self.starlist:
            [[i0,j0]] = self.wcs.wcs_world2pix([[xi,yi]],1)
            [[i1,j1]] = self.master.wcs.wcs_world2pix([[xi,yi]],1)
            if not (10 < i0 < self.data.shape[1]-10 and \
                    10 < j0 < self.data.shape[0]-10 and \
                    10 < i1 < self.master.data.shape[1]-10 and \
                    10 < j1 < self.master.data.shape[0]-10):
               continue
            pars = fit_psf.fit_psf(self.data, i0, j0, 1.0/self.scale, 2.0)
            pars = fit_psf.fit_psf(self.master.data, i1, j1,
                                   1.0/self.master.scale,2.0)
            x0.append(i0); y0.append(j0)
            x1.append(i1); y1.append(j1)
         x0,x1,y0,y1 = map(np.array, [x0,x1,y0,y1])

         Niter = 1
      else:
         # No guidance, just go with the stars we can find.

         if self.wcs is not None and self.master.wcs is not None:
            # Woo-hoo!  We've got a WCS.  This will make matching much
            #  easier.
            uv0 = self.wcs.wcs_pix2world(np.transpose([self.x,self.y]),1)
            uv1 = self.master.wcs.wcs_pix2world(
                  np.transpose([self.master.x,self.master.y]),1)
            #np.savetxt('uv0', uv0)
            #np.savetxt('uv1', uv1)
            c0 = SkyCoord(uv0[:,0], uv0[:,1], unit=(u.degree, u.degree))
            c1 = SkyCoord(uv1[:,0], uv1[:,1], unit=(u.degree, u.degree))
            idx,sep,dist = c0.match_to_catalog_sky(c1)
            # Those with a proper match
            gids = np.less(sep.value*3600/self.scale, perr)
            if np.sum(gids) < 3:
               self.log("Found fewer than 3 matches, try increasing match "
                        "tolerance (-s)")
               return -1
            x0 = self.x[gids];  y0 = self.y[gids]
            x1 = self.master.x[idx][gids];  y1 = self.master.y[idx][gids]
         else:
            # Do it the hard way...  Look for pairs of ojbects that have
            # the same distances and vector directions. There's probably a
            # way to do this that's invariant under relative rotation of the
            # frames. The vector angles are all w.r.t. each pixel frame, so
            # are NOT rotation invariant. For now, just cycle through a set of
            # angles.
            best_N = -1
            best_a = 0
            for aoffset in angles:
               da = self.da[::,::,NA,NA] - self.master.da[NA,NA,::,::]+aoffset
               da = np.where(np.less(da, -180.0), da+180.0, da)
               da = np.where(np.greater(da, 180.0), da-180.0, da)
               ds = self.ds[::,::,NA,NA] - self.master.ds[NA,NA,::,::]
               use = np.less(abs(ds),perr)*np.less(abs(da),aerr)
               suse = np.add.reduce(np.add.reduce(use,3),1)
               if max(np.ravel(suse)) < 4:  continue
               guse = np.greater(suse,max(np.ravel(suse))/2)
               self.log("angle {:.2f} gives {} matches".format(
                  aoffset, np.sum(np.ravel(guse))))
               if np.sum(np.ravel(guse)) > best_N:
                  # better match, save the angle
                  best_a = aoffset
                  best_N = np.sum(np.ravel(guse))
            # Now apply best angle and move forward.
            da = self.da[::,::,NA,NA] - self.master.da[NA,NA,::,::] + best_a
            ds = self.ds[::,::,NA,NA] - self.master.ds[NA,NA,::,::]
            use = np.less(abs(ds),perr)*np.less(abs(da),aerr)
            suse = np.add.reduce(np.add.reduce(use,3),1)
            if self.verb > 1:
               for suse1 in suse: self.log(suse1)
            guse = np.greater(suse,max(np.ravel(suse))/2)
            if self.verb:
               self.log('found %d matches' % np.sum(np.ravel(guse)))
               if best_a != 0:
                  self.log('best angular offset = %.2f' % best_a)
            i = [j for j in range(self.x.shape[0]) if np.sum(guse[j])]
            m = [np.argmax(guse[j]) for j in range(self.x.shape[0]) \
                  if np.sum(guse[j])]
            x0,y0 = np.take([self.x,self.y],i,1)
            x1,y1 = np.take([self.master.x,self.master.y],m,1)
            rscale = self.master.scale/self.scale
            best_a = best_a*np.pi/180.0
            xt1 = rscale*(np.cos(best_a)*x1 + np.sin(best_a)*y1)
            yt1 = rscale*(-np.sin(best_a)*x1 + np.cos(best_a)*y1)
            # Use bi-weight estimators for average and dispersion
            xshift,xscat = bwt(x0-xt1)
            xscat = max([1.0,xscat])
            yshift,yscat = bwt(y0-yt1)
            yscat = max([1.0,yscat])
            # Trow out deviant points
            keep = np.less(abs(x0-xt1-xshift),3*xscat)*\
                  np.less(abs(y0-yt1-yshift),3*yscat)
            x0,y0,x1,y1 = np.compress( keep, [x0,y0,x1,y1], 1)
         Niter = iter

      # If specified, remove distant points
      if self.maxdist > 0 and self.snpos is not None:
         # throw out distant points for coordinate transformation
         sni,snj = self.snpos.topixel()
         dists = np.sqrt((x0-sni)**2 + (y0 - snj)**2)
         close = np.less(dists, self.maxdist)
         x0,y0,x1,y1 = np.compress(close, [x0,y0,x1,y1], 1)

      # At this point, we should have a set of matching x,y pairs in the
      # image and template frames
      Nmatches = len(x0)
      if Nmatches < 3:
         self.log("Found fewer than 3 matches.  Aborting.")
         return -1

      # Check to see if we have enough matches to constrain the transformation
      # If not, reduce the order.
      Ncoeff = ndf(nord)
      nord0 = nord   # in case we have to decrease
      if 2*Nmatches < Ncoeff:
         self.log("You have fewer than %d matches, trying a lower " % \
               (Ncoeff/2) + "order transformation")
         nord -= 1
         while 2*Nmatches < ndf(nord):
            nord -= 1
         self.log("Using order %d transformation" % (nord))

      # Start off with equal weight to all pixels
      wt = np.ones(x0.shape,np.float32)

      if self.reject:
         # Reject objects that are saturated or otherwise have bad data
         bobjs0 = self.IDBadObjs()
         bobjs1 = self.master.IDBadObjs()
      else:
         bobjs0 = []
         bobjs1 = []

      allx0 = np.array(self.sexdata["X_IMAGE"])
      ally0 = np.array(self.sexdata["Y_IMAGE"])
      star0 = np.array(self.sexdata["CLASS_STAR"])
      if "MAG_BEST" in self.sexdata: 
         allm0 = np.array(self.sexdata["MAG_BEST"])
      elif "MAG_AUTO" in self.sexdata: 
         allm0 = np.array(self.sexdata["MAG_AUTO"])
      allq0 = np.array(self.sexdata["FLAGS"])
      allx1 = np.array(self.master.sexdata["X_IMAGE"])
      ally1 = np.array(self.master.sexdata["Y_IMAGE"])
      if "MAG_BEST" in self.master.sexdata:
         allm1 = np.array(self.master.sexdata["MAG_BEST"])
      elif "MAG_AUTO" in self.master.sexdata: 
         allm1 = np.array(self.master.sexdata["MAG_AUTO"])
      allq1 = np.array(self.master.sexdata["FLAGS"])

      if self.maxdist > 0 and self.snpos is not None:
         # throw out distant points for coordinate transformation
         sni,snj = self.snpos.topixel()
         dists = np.sqrt((allx0-sni)**2 + (ally0-snj)**2)
         close = np.less(dists, self.maxdist)
         allx0,ally0,star0,allm0,allq0 = np.compress(close,
               [allx0,ally0,star0,allm0,allq0], 1)

      #np.savetxt('pass0_matches.np',[x0,y0,x1,y1])
      #np.savetxt('allx0s.np', [allx0,ally0])
      #np.savetxt('allx1s.np', [allx1,ally1])
      # now we iterate on the solution and throw out bad points
      for iter in range(Niter):
         if iter:
            if nord == -1:
               ix = allx0 + xshift
               iy = ally0 + yshift
            else:
               basis = abasis(nord,allx0,ally0,rot=[0,1][nord==0])
               ixy = np.add.reduce(sol[NA,::]*basis,1)
               ix,iy = ixy[:len(np.ravel(allx0))], ixy[len(np.ravel(allx0)):]
            delx = ix[::,NA] - allx1[NA,::]
            dely = iy[::,NA] - ally1[NA,::]
            dels = np.sqrt(np.power(delx,2) + np.power(dely,2))
            delq = np.less_equal(allq0,4)[::,NA]*np.less_equal(allq1,4)[NA,::]
            if getattr(self,'starmin', -1) > 0:
               delq = delq * np.less_equal(star0,self.starmin)[::,NA]
            dels = np.where(delq,dels,9999)
            ui0 = [];  ui1 = []
            for j in range(delx.shape[0]):
               if min(dels[j]) < perr and j not in bobjs0:
               #if min(dels[j]) < tol and j not in bobjs0:
                  idx = np.argmin(dels[j])
                  if idx not in bobjs1:
                     ui0.append(j)
                     ui1.append(idx)

            #ui0 = [j for j in range(delx.shape[0]) if min(dels[j]) < perr and \
            #      j not in bobjs0]
            #ui1 = [np.argmin(dels[j]) for j in range(delx.shape[0]) \
            #      if min(dels[j]) < perr and np.argmin(dels[j]) not in bobjs1]
            if len(ui0) == 0:
               self.log("Error:  Residuals of coordinate tranformation are all"
                        "greater")
               self.log("than one pixel.  Try with smaller order.")
               return -1
            x0,y0,m0 = np.take([allx0,ally0,allm0],ui0,1)
            x1,y1,m1 = np.take([allx1,ally1,allm1],ui1,1)
            #np.savetxt('pass{}_matches.np'.format(iter),[x0,y0,x1,y1])

            f0 = np.power(10,-0.4*(m0-min(m0)))
            f1 = np.power(10,-0.4*(m1-min(m1)))
            # not sure about the point here...
            wt = np.sqrt(f0+f1)# * 0.0 + 1
            #if self.maxdist is not None and self.snpos is not None:
            #   sni,snj = self.snpos.topixel()
            #   dists = np.sqrt((x0 - sni)**2 + (y0 - snj)**2)
            #   wt = wt*np.less(dists, self.maxdist)

            Nmatches = len(x0)
            if nord < nord0:
               # Try to re-increase the order to what was asked for originally
               while 2*Nmatches > ndf(nord) and nord <= nord0:
                  nord += 1
         if nord == -1:
            # simple offset
            xshift = np.sum(wt*(x1-x0))/np.sum(wt)
            yshift = np.sum(wt*(y1-y0))/np.sum(wt)
            sol = np.array([xshift,yshift,1.0,0.0])
            ix = x0 + xshift
            iy = y0 + yshift
         else:
            # higher-order tranformation, solve for it
            wt = np.concatenate([wt,wt])
            basis = abasis(nord,x0,y0,rot=[0,1][nord==0])
            sol = svdfit(basis*wt[::,NA],np.concatenate([x1,y1])*wt)
            ixy = np.add.reduce(sol[NA,::]*basis,1)
            ix,iy = ixy[:len(np.ravel(x0))], ixy[len(np.ravel(x0)):]

         Nmatches = len(x0)
         self.log( "Pass %d, with %d objects." % (iter+1,Nmatches))
         if nord==-1:
            self.log("  xshift: %.3f   yshift:  %.3f" % tuple(sol[0:2]))
         elif nord == 0:
            theta = np.arctan2(sol[3],sol[2])
            scale = sol[2]/np.cos(theta)
            self.log(" xshift: %.3f yshift: %.3f scale: %.3f rot:  %.3f" %\
                  (sol[0],sol[1],scale,theta*180.0/np.pi))
         else:
            self.log( str(sol))
         delx = ix-x1
         dely = iy-y1
         dels = np.sqrt(np.power(delx,2) + np.power(dely,2))
         scx = bwt(delx)[1]
         scy = bwt(dely)[1]
         self.log( "Biweight estimate for scatter in coordinate trans. (x,y) ="
                   "(%.5f,%.5f)" % (scx,scy))
         tol = 3.0*bwt(dels)[1]

      if not use_db:
         # Save matches for later use
         f = open(self.db, 'w')
         for i in range(len(x0)):
            #print >>f, x0[i],y0[i]
            f.write("{} {}\n".format(x0[i],y0[i]))
         f.close()
         f = open(self.master.db, 'w')
         for i in range(len(x0)):
            #print >>f, x1[i],y1[i]
            f.write("{} {}\n".format(x1[i],y1[i]))
         f.close()

      self.sol = sol
      self.nord = nord
      return 0

   def imread(self, bs=False, blur=0):
      '''Read in the FITS data and assign to the member variables.  
      Optionall background subtract if bs=True'''
      if self.data is None:
         self.log( "Now reading frames for %s" % (self))
         #f = FITS.open(self.image, memmap=False)
         #self.data = f[self.hdu].data.astype(np.float32)
         h,self.data = qload(self.image, self.hdu)
         self.data = self.data*1.0
         #h = FITS.getheader(self.image, self.hdu)
         if 'ZP' in h:
            if type(h['ZP']) is type(1.0):
               self.zp = h['ZP']
            else:
               print("Warning: FITS header ZP is present, but not a float")
               self.zp = None
         else:
            self.zp = None

         self.fwhm = h.get('FWHM', None)

         if self.master: 
            self.master.imread(bs=bs)

         # Check for previously determind BG and SD
         if self.store_bg and 'IMMATBG' in h:
            self._bg = h['IMMATBG']
            self._r = h['IMMATSD']

         # If we blur the image first:
         if blur is not None and blur != 0:
            if blur > 0:
               k = Gaussian2DKernel(blur)
               self.log("Blurring by Gaussian with sigma={}".format(blur))
            else:
               # figure it out automagically. Assume FWHM heders are in arcsec
               if self.fwhm is not None and self.master.fwhm is not None and \
                     self.fwhm < self.master.fwhm:
                  blur = np.sqrt(self.master.fwhm**2-self.fwhm**2)*f2s
                  blur = blur/self.scale
                  if blur > 0.5:
                     self.log("Blurring by Gaussian with sigma={}".format(blur))
                     k = Gaussian2DKernel(blur)
                  else:
                     k = None
               else:
                  k = None
            if k is not None:
               self.data = convolve_fft(self.data, k)
               # Save blurred imae for use with sextractor
               newimage = self.image.replace('.fits','_blur.fits')
               qdump(newimage, self.data, self.image)
               self.image = newimage

         # Check for NaNs
         self.nans = np.isnan(self.data)
         if np.sometrue(self.nans):
            # Keep track of where they were
            self.data = np.where(self.nans, 0.0, self.data)
            # make a new file, since sextractor is messed up by NaNs
            newimage = self.image.replace('.fits','_nonan.fits')
            qdump(newimage, self.data, self.image)
            self.image = newimage

         # see if there is WCS info in the header
         if wcs is not None:
            self.wcs = wcs.WCS(h)
            if not self.wcs.has_celestial:
               self.log('WCS for %s has no celestial coordinates' % self.image)
               self.wcs = None
            self.log('Loading WCS info for %s' % self.image)
            #if self.snra is not None and self.sndec is not None:
            #   self.snx,self.sny = self.wcs.topixel((self.snra,self.sndec))
         else:
            self.wcs = None
         if bs:
            self.data = self.data - GaussianBG(np.ravel(self.data).astype(
                                               np.float32),101)
         self.naxis1,self.naxis2 = h['NAXIS1'],\
                                   h['NAXIS2']
         if self.sigimage is not None:
            self.log( "Reading in sigma map %s" % self.sigimage)
            #f = FITS.open(self.sigimage, memmap=False)
            #self.sigma = f[self.hdu].data.astype(np.float32)
            #f.close()
            h,self.sigma = qload(self.sigimage, self.hdu)
         else:
            self.sigma = None

         if self.extra_maps is not None:
            self.extras = []
            for xmap in self.extra_maps:
               self.log("Reading in extra map %s" % xmap)
               #f = FITS.open(xmap, memmap=False)
               #self.extras.append(f[self.hdu].data.astype(np.float32))
               #f.close()
               self.extras.append(qload(xmap, self.hdu)[1])
         else:
            self.extras = None


   def blurImage(self, blur):
     if blur > 0:
        k = Gaussian2DKernel(blur)
        self.log("Blurring by Gaussian with sigma={}".format(blur))
     else:
        # figure it out automagically. Assume FWHM heders are in arcsec
        if self.fwhm is not None and self.master.fwhm is not None and \
              self.fwhm < self.master.fwhm:
           blur = np.sqrt(self.master.fwhm**2-self.fwhm**2)*f2s
           blur = blur/self.scale
           if blur > 0.5:
              self.log("Blurring by Gaussian with sigma={}".format(blur))
              k = Gaussian2DKernel(blur)
           else:
              k = None
        else:
           k = None
     if k is not None:
        self.data = convolve_fft(self.data, k)
        # Save blurred imae for use with sextractor
        newimage = self.image.replace('.fits','_blur.fits')
        qdump(newimage, self.data, self.image)
        self.image = newimage

   def mktemplate(self,registered,usewcs=False,sol=None,mimage=None,
                  blur=None):
      '''Transform the template (master) to the coordinates of the image.'''
      self.oy,self.ox = np.indices((self.naxis2,self.naxis1),np.float32)

      # first, reject master segmented objects before we transform
      if self.reject:
         # reject full objects if requested
         sat_pixels_m = np.greater(self.master.data, self.master.saturate)
         sat_pixels_m = np.logical_or(sat_pixels_m, self.master.nans)
         sat_ids_m = np.nonzero(np.ravel(sat_pixels_m))
         sat_objects_m = np.ravel(self.master.seg)[sat_ids_m]
         if self.master.magmax or self.master.magmin:
            gids = np.greater(self.master.segmags, -900)
            if self.master.magmax: 
               gids = gids*np.less(self.master.segmags, self.master.magmax)
            if self.master.magmin: 
               gids = gids*np.greater(self.master.segmags, self.master.magmin)
            gids[0] = False   # zero is no object
            sat_objects_m = np.concatenate([sat_objects_m,np.nonzero(~gids)[0]])
         # Now a bit of trickery to find non-repeating set of object numbers:
         objects_m = list(set(sat_objects_m))
         for obj in objects_m:
            self.master.seg[np.equal(self.master.seg,obj)] = 0

      if registered:
         # Well, then, not muc to do in that case!
         self.timage = self.master.data
         if self.master.sigimage is not None:
            self.tsigma = self.master.sigma
         else:
            self.tsigma = None
         if self.master.extras is not None:
            self.textras = [xmap for xmap in self.master.extras]
         else:
            self.textras = None
         self.mseg = np.greater(self.master.seg, 0.0).astype(np.float32)
      else:
         if not mimage: mimage = self.master.data
         if usewcs and self.wcs is not None and self.master.wcs is not None:
            # Here we don't bother with our own coordinate solution,
            #  we use the WCS in each image to do the coordinate
            #  transformation
            ux,uy = self.wcs.wcs_pix2world(self.ox,self.oy,0)
            ix,iy = self.master.wcs.wcs_world2pix(ux,uy, 1)
            self.tx = ix - self.ox
            self.ty = iy - self.oy
         else:
            if not sol: sol = self.sol
            self.log( "Constructing transformation...")
            ix,iy = mbasis(sol,self.ox,self.oy,rot=[0,1][self.nord<=0])
            self.tx = ix - self.ox
            self.ty = iy - self.oy
         self.log( "Transforming...")
         # Let's try geomap like thing
         self.timage = map_coordinates(mimage, [iy, ix], order=3,
               mode='constant', cval=0)
         #self.timage = VTKImageTransform(mimage,self.tx,self.ty,numret=1,
         #                                cubic=0,interp=1,constant=0)
         if self.verb:
            qdump(self.rectemp, self.timage, self.master.image)
         if self.master.sigimage is not None:
            self.log( "Transforming sigma map...")
            bpm = np.less(self.master.sigma,0)*1.0
            self.tsigma = map_coordinates(self.master.sigma, [iy,ix], 
                  order=3, mode='constant', cval=0)
            #self.tsigma = VTKImageTransform(self.master.sigma, self.tx, 
            tbpm = map_coordinates(bpm, [iy,ix], 
                  order=3, mode='constant', cval=0)
            #tbpm = VTKImageTransform(bpm, self.tx, 
            #      self.ty, numret=1, cubic=0, interp=1, constant=-1)
            self.tsigma = np.where(tbpm > 0.1, -1, self.tsigma)
         else:
            self.tsigma = None

         if self.master.extras is not None:
            self.log( "Transforming extra maps...")
            self.textras = []
            for xmap in self.master.extras:
               self.textras.append(map_coordinates(xmap, [iy,ix], 
                     order=3, mode='constant', cval=0))
               #self.textras.append(VTKImageTransform(xmap, self.tx,
               #     self.ty, numret=1, cubic=0, interp=1, constant=-1))
         else:
            self.textras = None

         mseg = np.greater(self.master.seg, 0.0).astype(np.float32)
         if self.master.mask:
            mseg = VTKMultiply(mseg, self.master.mask.get_mask())
         self.mseg = map_coordinates(mseg, [iy, ix], order=3, mode='constant',
               cval=0)
         #self.mseg = VTKImageTransform(mseg,self.tx,self.ty,numret=1,
         #                              cubic=0,interp=1,constant=-1)

   def estimate_bg(self, Niter=5):
      '''Estimate the background level.'''
      if getattr(self, '_bg', None) is not None:
         self.log("Using stored: BG=%.3f with sigma=%.3f" % (self._bg,self._r))
         return self._bg,self._r
      for uk in range(Niter):
         if uk:
            ukeep = 1.0*between(self.data, bg-4*r, bg+2*r)
            ukeep = 1.0*np.equal(VTKConvolve(ukeep, k=5, numret=1),1.0)
            udata = np.compress(ukeep.ravel(), self.data.ravel())
         else:
            udata = self.data.ravel()
         
         bg = GaussianBG(udata,99)
         #d = VTKSubtract(udata, bg)
         d = udata - bg
         r = 1.49*np.median(abs(d))
         if self.verb > 0:
            self.log(" BG iter %d: BG=%.3f, sigma=%.3f" % (uk,bg, r))
      self.log("Using %d: BG=%.3f with sigma=%.3f" % (len(udata),bg,r))

      # Save values into header, since they are expensive to compute
      if self.store_bg:
         FITS.setval(self.image, 'IMMATBG', value=bg)
         FITS.setval(self.image, 'IMMATSD', value=r)

      # Save for multiple calls (good for templates!)
      self._bg = bg
      self._r = r
      return bg,r

   def mkweight(self):
      '''This function works out the weight of the image, based on the noise
      in the background and the segmentation found by sextractor.  This is
      where you want to fiddle with which parts of the image make the most
      contribution to the final kernel solution.'''

      self.log( "Working on mask and weights...")
      # These are maps of where sextractor found objects, seg for this image
      #  and mseg for the master (template)

      if self.reject:
         # First, we figure out which objects have saturated pixels in them:
         sat_pixels = np.greater(self.data, self.saturate)
         # augment this with NaNs
         sat_pixels = np.logical_or(sat_pixels, self.nans)
         sat_ids = np.nonzero(np.ravel(sat_pixels))
         sat_objects = np.ravel(self.seg)[sat_ids]
         if self.magmax or self.magmin:
            gids = np.greater(self.segmags, -900)
            if self.magmax: gids = gids*np.less(self.segmags, self.magmax)
            if self.magmin: gids = gids*np.greater(self.segmags, self.magmin)
            gids[0] = False   # zero is no object
            sat_objects = np.concatenate([sat_objects, np.nonzero(~gids)[0]])

         # Retain only stellar objects
         if getattr(self,'starmin', -1) > 0:
            sat_objects = np.concatenate([sat_objects, self.sexdata['NUMBER'][\
                  self.sexdata['CLASS_STAR'] < self.starmin]])

         # Now a bit of trickery to find non-repeating set of object numbers:
         objects = list(set(sat_objects))
         if self.verb > 1:
            with open('bad_objects','w') as fout:
               [fout.write("{}\n".format(obj)) for obj in objects]

         # Now cut out the objects with saturated pixels.
         for obj in objects:
            self.seg[self.seg == obj] = 0

      # Throug out any pixeles that don't have an object
      seg = np.greater(self.seg,0.0).astype(np.float32)
      # put limit at 0.5, because mseg was transformed
      mseg = np.greater(self.mseg,0.5).astype(np.float32)
      if self.verb > 1:
         qdump('mseg.fits', mseg)

      gwid = max([2,1*self.pwid])
      self.bg,r = self.estimate_bg()
      self.tbg,r2 = self.master.estimate_bg()
      self.log( "Image:" )
      self.log( "Using BG=%.3f with sigma=%.3f" % (self.bg,r))
      self.log( "Cutting below 5-sigma: %.3f" % (self.bg-5*r))
      self.log( "Template:" )
      self.log( "Using BG=%.3f with sigma=%.3f" % (self.tbg,r2))
      self.log( "Cutting below 5-sigma: %.3f" % (self.tbg-5*r2))
      # Lower-cut. Reject data less than lcut

      # Take out data that is explicitly set to zero. Hmmmmmm....
      zids = np.equal(self.data, 0.0).astype(np.float32)
      zids = VTKGauss(zids, gwid, numret=1)
      # note by using np.less(zids, 0.02), we're effectively doing a fuzzy 
      #  'not'
      wt = VTKMultiply(seg, VTKMultiply(mseg, np.less(zids, 0.02)))
      if self.verb > 1:
         qdump('wt0.fits', wt)
      #print(np.sum(np.greater(wt, 0)))

      # throw out data that have very low values
      lowids = np.less(self.data, self.bg-5*r).astype(np.float32)
      lowids = VTKGauss(lowids, gwid, numret=1)
      swt = np.less(lowids, 0.02).astype(np.float32)

      lowids = np.less(self.timage, self.tbg-5*r2).astype(np.float32)
      lowids = VTKGauss(lowids, gwid, numret=1)
      twt = np.less(lowids, 0.02).astype(np.float32)
      wt = VTKMultiply(np.greater(VTKGauss(wt,gwid,numret=1),
                               0.02).astype(np.float32),swt*twt)
      if self.verb > 1:
         qdump('wt1.fits', wt)

      # Throw out data on the boundaries twice as wide as the kernel
      insidex = between(self.ox,2*gwid,self.naxis1-2*gwid).astype(np.float32)
      insidey = between(self.oy,2*gwid,self.naxis2-2*gwid).astype(np.float32)
      wt = VTKMultiply(wt,VTKAnd(insidex, insidey))
      if self.verb > 1:
         qdump('wt2.fits', wt)

      # If the FITS header had masks in the header, then we mask that out too
      #  This is opposite of xwin,ywin:  it's want we want to keep OUT.
      # Note that xwin and ywin are now included in get_mask
      wt = VTKMultiply(wt, self.mask.get_mask())
      if self.verb > 1:
         qdump('wt3.fits', wt)

      if self.snpos is not None:
         self.log("Applying super nova mask at x=%f, y=%f" % \
               (self.snpos.topixel()))
         snx,sny = self.snpos.topixel()
         dists = np.power(self.ox-snx, 2) + np.power(self.oy-sny, 2)
         cond = np.greater(dists, (self.snmaskr/self.scale)**2)
         if self.maxdist > 0:
            cond = cond*np.less(dists, self.maxdist**2)
         wt = VTKMultiply(wt, cond.astype(np.float32))
      if self.verb > 1:
         qdump('wt4.fits', wt)

      # Get rid of any saturated pixels

      swt = VTKGreaterEqual(self.data,self.saturate)
      twt = VTKGreaterEqual(self.timage,self.master.saturate)
      swt = VTKDilate(VTKOr(swt,twt),5,5,numret=1)
      swt = VTKGauss(swt,gwid,numret=1)
      wt = VTKMultiply(wt, VTKLessEqual(swt,0.02))
      if self.verb > 1:
         qdump('wt5.fits', wt)

      if self.sigimage:
         self.noise = self.sigma.astype(np.float64)
      else:
         n2 = VTKSubtract(self.data,self.bg)
         self.noise = VTKSqrt(VTKAdd(n2,pow(r,2)))
      self.invnoise = VTKInvert(self.noise)
      if self.master.sigimage:
         self.master.noise = self.master.sigma
      else:
         n2 = VTKSubtract(self.master.data, self.tbg)
         self.master.noise = VTKSqrt(VTKAdd(n2, pow(r2,2)))
      self.tinvnoise = VTKInvert(self.master.noise)
      wt = VTKMultiply(wt, self.invnoise)

      # Get rid of little "islands" of data that can't constrain the kernel
      # and just add to the noise
      if self.pwid > -1: 
         wt = VTKIslandRemoval(wt,1.0,0.0,max([3,self.pwid])**2,numret=1)
      self.wt = wt.astype(np.float32)

   def mkmatch(self,preserve,quick_convolve=0, Niter=1):
      '''Here's where the kernel is solved.  Need to work on comments here.
      Right now, it's pretty black-box.'''
      data = self.data - self.bg

      # Try to get counts on same scale
      timage = (self.timage - self.tbg) * (self.exptime/self.master.exptime)
      for i in range(Niter):
         if i == 0:
            # First time through
            wtflat = np.ravel(self.wt)
         else:
            # after first time throug:  throw out outliers.
            # index numbers where we have non-zero weights
            ids = np.nonzero(np.greater(wtflat, 0))[0]
            # expected noise at those locations
            sig = np.power(n0, -1)
            # where the residuals exceed 5*noise
            bids = np.greater(np.absolute(resid), 5*sig)
            # set them to zero in the original weight-map
            wtflat[ids[bids]] = 0

 
         self.log( "Using %d pixels." % (np.sum(1.0*np.greater(wtflat,0))))
         if self.pwid == -1:
            # Just solving for a flux ratio
            flux = np.sum(wtflat*np.ravel(data))/\
                  np.sum(wtflat*np.ravel(timage))
            self.log( "Flux ratio = %.4f" % (flux))
            self.match = flux*timage
            resid = np.compress(wtflat, np.ravel(data)) -\
                    np.compress(wtflat, np.ravel(flux*timage))
            if self.sigimage is not None and self.master.sigimage is not None:
               self.csigma = np.sqrt(np.power(self.sigma,2) + \
                     np.power(self.tsigma, 2))
            self.fluxrat = 1.0
            self.dof = 1
            umean = np.mean(resid)
            urms = np.sqrt(np.mean(np.power(resid,2)))
            uchi2 = np.sum(np.power(resid*np.compress(wtflat,wtflat),2))
            uchi2u = uchi2 / (len(resid)-self.dof)
            self.log( "(CUT,MEAN,RMS,CHI2,RCHI2) = "
                      "(%9.1f,%12.6f,%12.6f,%12.6f,%12.6f)"\
                            % (0,umean,urms,uchi2,uchi2u))

            return
         else:
            # Full width of the kernel, make it odd
            pful = int(2*self.pwid+1)
            owid = self.pwid + 1.0
            # These are indices of the side of a box containing the kernel
            pbox = np.arange(-self.pwid,self.pwid+0.5,1.0).astype(np.int32)
            # These are the indices in the pbox that correspond to being
            # inside a circular aperture.
            self.ii = np.array([i for i in pbox for j in pbox \
                             if (i*i+j*j)<=owid**2 or 0])
            self.jj = np.array([j for i in pbox for j in pbox \
                             if (i*i+j*j)<=owid**2 or 0])
            ll = len(self.ii)
            self.log( "Constructing bases.")
            # The construction of bases depends on the image data or the template
            #  (depending on direction), whether or not there is a sky offset
            #  and whether or not there is spatial variation.
            it0 = time.time()
            basis_sigma = None
            if self.rev:
               f0 = np.compress(wtflat,np.ravel(timage))
               n0 = np.compress(wtflat,np.ravel(self.invnoise))
               basisimage = data
               if self.sigimage is not None:
                  # make a variance map 
                  basis_sigma = np.power(self.sigma, 2)
            else:
               f0 = np.compress(wtflat,np.ravel(data))
               n0 = np.compress(wtflat,np.ravel(self.invnoise))
               basisimage = timage
               if self.master.sigimage is not None:
                  basis_sigma = np.power(self.tsigma, 2)
            step = ll/79 + 1  # for performance meter
            cwt = np.compress(wtflat,wtflat)
            # i,j indeces of masked image
            coy,cox = np.compress(wtflat, 
               [np.ravel(self.oy), np.ravel(self.ox)], 1).astype(np.int32)
            flatimage = np.ravel(basisimage)
            basis = []
            for k in range(ll):
               basis.append(cwt*np.take(flatimage,
                     (coy+self.jj[k])*self.naxis1 + (cox+self.ii[k])))
               if k % step == 0:
                  sys.stdout.write(".")
                  sys.stdout.flush()
 
            if self.skyoff: basis = np.concatenate([basis,
               [np.ones(f0.shape,np.float64)]])
            basis = np.asarray(basis)
            if self.spatial:
                self.oxl = (self.ox-self.naxis1/2.0)/(self.naxis1/2.0)
                woxl = np.compress(wtflat,np.ravel(self.oxl))
                self.oyl = (self.oy-self.naxis2/2.0)/(self.naxis2/2.0)
                woyl = np.compress(wtflat,np.ravel(self.oyl))
                xbasis = [b*woxl for b in basis[:-1]]
                ybasis = [b*woyl for b in basis[:-1]]
                basis = np.concatenate([basis,xbasis,ybasis])
            it1 = time.time()
            self.log( "Bases constructed in %.4fs." % (it1-it0))
            self.log( "Decomposition.")
            it0 = time.time()
            self.log( "Size of basis matrix:")
            self.log(str(basis.shape))
            self.dof = basis.shape[0]
            if basis.shape[1] < basis.shape[0]:
               raise RuntimeError("Error:  Not enough sources to constrain "\
                     " kernel. Try increase decreasing threshold?")
            du,dv,dw = singular_value_decomposition(np.transpose(basis))
            it1 = time.time()
            self.log( "Decomposition in %.4fs." % (it1-it0))
            bases = []
            for sol1a in dw:
                psf2 = np.zeros((pful,pful),np.float64)
                for k in range(ll):
                    py = self.jj[k]+self.pwid
                    px = self.ii[k]+self.pwid
                    pr = np.sqrt(self.jj[k]**2 + self.ii[k]**2)
                    psf2[py,px] = sol1a[k]
                bases.append(psf2)
            bases = np.asarray(bases)
            npsx = int(np.sqrt(bases.shape[0]))+1
            npsy = bases.shape[0]//npsx + 1
            imbases = np.zeros((npsy*(bases.shape[1]+5), 
                                npsx*(bases.shape[2]+5)), np.float64)
            imb1 = 0
            for j in range(npsy):
              for i in range(npsx):
                 y0 = j*(bases.shape[1]+5)
                 y1 = y0 + bases.shape[1]
                 x0 = i*(bases.shape[2]+5)
                 x1 = x0 + bases.shape[2]
                 imbases[y0:y1,x0:x1] = bases[imb1,::-1,::]
                 imb1 += 1
                 if imb1 == len(bases): break
            imbases = imbases[::-1]
            if self.verb > 1: qdump(self.imbases,imbases,self.image)
            if self.verb > 1: self.log( dv)
            sol1 = np.transpose(dw)
            if self.mcut and self.mcut < len(dv):
                self.log( "Ratio of first eigenvalue with last ten:")
                rdv = dv[0]/dv
                self.log( rdv[-10:])
                if self.verb > 1:  qdump(self.rdv,rdv)
                self.rdv0 = rdv*1  # save a copy
                dv[self.mcut:] = 0 * dv[self.mcut:]
                sol2 = divz(1,dv[:self.mcut])
                sol3 = np.dot(np.transpose(du[::,:self.mcut]),cwt*f0)
                self.psf1 = np.dot(sol1[::,:self.mcut], (sol2*sol3))
                fitted = np.add.reduce(self.psf1[::,NA]*basis,0)
                resid = fitted - f0
                vc = dv[0]/dv[self.mcut-1]
                umean = np.mean(resid)
                urms = np.sqrt(np.mean(np.power(resid,2)))
                uchi2 = np.sum(np.power(resid*np.compress(wtflat,wtflat),2))
                uchi2u = uchi2 / (len(resid)-self.dof)
                self.log( "(CUT,MEAN,RMS,CHI2,RCHI2) = "
                          "(%9.1f,%12.6f,%12.6f,%12.6f,%12.6f)"\
                                % (vc,umean,urms,uchi2,uchi2u))
            else:
                self.log( "Ratio of first eigenvalue with last ten:")
                rdv = dv[0]/dv
                self.rdv0 = rdv*1   # save a copy
                self.log( str(rdv[-10:]))
                if self.verb > 1: qdump(self.rdv,rdv)
                cdv = np.zeros(rdv.shape,np.float64)
                if self.vcut == 0: ucv = [0,1]
                else: ucv = [1]
                for cv in ucv:
                   if cv:
                       if self.vcut == 0:
                          if self.verb > 1:  qdump("cdv.fits",rdv - cdv)
                          tck = splrep(np.arange(len(dv)),rdv-cdv,k=1,s=0)
                          vloc = sproot(tck,mest=len(dv))
                          vloc = int(vloc[0])
                          self.vcut = 10*rdv[vloc]
                       rdv = [self.vcut]
                       self.log( "Cutting at %1f." % (self.vcut))
                   for jvc in range(len(rdv)):
                       vc = rdv[jvc]
                       udv = 1.0*np.greater_equal(dv,max(dv)/vc)
                       jcut = int(np.sum(udv))
                       if cv: self.log( "Using %d out of %d." % \
                             (np.sum(udv),len(udv)))
                       sol2 = divz(1,dv[:jcut])
                       sol3 = np.dot(np.transpose(du[::,:jcut]),cwt*f0)
                       #np.savetxt('sol1.dat',sol1)
                       #np.savetxt('sol2.dat',sol2)
                       #np.savetxt('sol3.dat',sol3)

                       self.psf1 = np.dot(sol1[::,:jcut], (sol2*sol3))
                       fitted = np.add.reduce(self.psf1[::,NA]*basis,0)
                       resid = fitted - f0
                       umean = np.mean(resid)
                       urms = np.sqrt(np.mean(np.power(resid,2)))
                       uchi2 = np.sum(np.power(resid*np.compress(wtflat,wtflat),2))
                       uchi2u = uchi2 / (len(resid)-self.dof)
                       if not cv: cdv[jvc] = urms
                       self.log( "(CUT,MEAN,RMS,CHI2,RCHI2) = "
                                 "(%9.1f,%12.6f,%12.6f,%12.6f,%12.6f)" \
                                       % (vc,umean,urms,uchi2,uchi2u))
            del sol1,sol2,sol3,du,dv,dw,basis
         if self.skyoff and self.pwid > 0: 
            self.log( "Sky= %f" % (self.psf1[ll]))
         fluxrat = np.sum(self.psf1[:ll])
         self.log( "Flux ratio=%f"%fluxrat)
         if preserve:
            # Save the flux ratio for later if reversing.
            self.log( "Preserving flux?")
            self.fluxrat = fluxrat
            self.psf1[:ll] = self.psf1[:ll] / self.fluxrat
            self.log( "Flux ratio=%f"%self.fluxrat)
         else:
            self.fluxrat = 1.0

         if self.verb > 1:
           self.psf2 = np.zeros((pful,pful),np.float64)
           pxs = []
           pys = []
           prs = []
           pfs = []
           for k in range(ll):
               py = self.jj[k]+self.pwid
               px = self.ii[k]+self.pwid
               pr = np.sqrt(self.jj[k]**2 + self.ii[k]**2)
               self.psf2[py,px] = self.psf1[k]
               prs.append(pr)
               pxs.append(self.ii[k])
               pys.append(self.jj[k])
               pfs.append( self.psf2[py,px])
           prsf = np.transpose([pxs,pys,prs,pfs])
           psffile = open("psf.dat","w")
           [psffile.write("%f %f %f %f\n" % tuple(p)) for p in prsf]
           psffile.close()
           for p in self.psf2:
              fmt = pful*"  %7.4f"
              self.log( fmt % tuple(p))
         symm = False
         if symm:
            self.psf2 = (self.psf2+ self.psf2[::-1,::]+ self.psf2[::,::-1]+\
                  self.psf2[::-1,::-1])/4.0
            for k in range(ll):
                py = self.jj[k]+self.pwid
                px = self.ii[k]+self.pwid
                self.psf1[k] = self.psf2[py,px]
         self.match = np.zeros(data.shape,np.float64)
         self.grid = np.zeros(data.shape,np.float64)
         if basis_sigma is not None:
            # the noise in the convolved image:
            self.csigma = np.zeros(data.shape, np.float64)
         dgridx, dgridy = int(self.naxis1/32), int(self.naxis2/32)
         gridin = np.logical_not(np.fmod(self.ox+dgridx,dgridx*2.0)) * \
                  np.logical_not(np.fmod(self.oy+dgridy,dgridy*2.0))
         gridin = gridin.astype(np.int8)
         sys.stdout.write("Constructing matched image.\n")
         it0 = time.time()
         if quick_convolve:
            self.match = self.quick_convolve(basisimage, grid=0)
            self.grid = self.quick_convolve(gridin, grid=1)
            if basis_sigma is not None:
               self.csigma = self.quick_convolve(basis_sigma, grid=0)
         else:
            for k in range(ll):
               if k % step == 0:
                  sys.stdout.write(".");sys.stdout.flush()
               self.match += self.component(k,matching=basisimage)
               self.grid += self.component(k,matching=gridin,grid=1)
               if basis_sigma is not None:
                  self.csigma += self.component(k, matching=basis_sigma)
         it1 = time.time()
         self.log( "done.\nConstructed matched image in %.4fs." % (it1-it0))
         if basis_sigma is not None:
            self.csigma = np.sqrt(self.csigma)


   def quick_convolve(self, matching, grid=0):
      '''Use FFT to do a quick convolve if the psf is not spatially varying.'''

      shape = matching.shape
      kshape = (2*self.pwid+1,2*self.pwid+1)
      ksmall = np.zeros(kshape, np.float32)
      ii = self.ii + self.pwid
      jj = self.jj + self.pwid
      for k in range(len(self.ii)):  ksmall[jj[k],ii[k]] = self.psf1[k]

      ksmall = ksmall[::-1,::-1]

      kernel = np.zeros((kshape[0]+shape[0], kshape[1]+shape[1]), np.float32)
      kernel[:kshape[0], :kshape[1]] = ksmall[:,:]
      data = kernel*0.0
      dy = (data.shape[0] - matching.shape[0])//2
      dx = (data.shape[1] - matching.shape[1])//2
      my = matching.shape[0] + dy
      mx = matching.shape[1] + dx
      data[dy:my, dx:mx] = matching[:,:]

      Fdata = fft.rfft2(data)
      sys.stdout.write('.'*8)
      sys.stdout.flush()
      del data
      Fkernel = fft.rfft2(kernel)
      sys.stdout.write('.'*8)
      sys.stdout.flush()
      np.multiply(Fdata, Fkernel, Fdata)
      del Fkernel
      convolved = fft.irfft2(Fdata, s=kernel.shape)
      sys.stdout.write('.'*8)
      sys.stdout.flush()
      convolved = convolved[kshape[0]-1:shape[0]+kshape[0]-1,
                            kshape[1]-1:shape[1]+kshape[1]-1]
      return(convolved) 

   def component(self,k,matching,grid=0):
      ll = len(self.ii)
      if self.skyoff: s = 1
      else: s = 0
      p, i, j = self.psf1[k], self.ii[k%ll], self.jj[k%ll]
      if s+ll <= k < s+2*ll: p = p*self.oxl
      elif s+2*ll <= k < s+3*ll: p = p*self.oyl
      if k == ll and self.skyoff: return [p,0.0][grid]
      else: return p*QuickShift(matching,i,j)


   def GoCatGo(self,master,verb=0,rev=0,xwin=None,ywin=None, 
               skyoff=0,pwid=0, perr=5.0,nmax=40,nord=0,spatial=0,mcut=0,
               vcut=1e8,match=1, subt=0, registered=0,preserve=0, 
               min_sep=None, quick_convolve=0, do_sex=0, thresh=3., 
               sexdir="./sex", sexcmd='sex', starmin=0, maxdist=-1,
               angles=[0.0], use_db=0, interactive=0, starlist=None, 
               diff_size=None, bs=False, crowd=False, usewcs=False,
               magcat=None, racol='RA', deccol='DEC', magcol='mag',
               Niter=3, blur=None):
      self.master = master
      self.skyoff=skyoff
      self.pwid=pwid
      if xwin and ywin:
         self.mask.add_mask(xwin[0], ywin[0], xwin[1], ywin[1], 
               sense='outside')
      elif xwin:
         self.mask.add_mask(xwin[0], None, xwin[1], None, sense='outside')
      elif ywin:
         self.mask.add_mask(None, ywin[0], None, ywin[1], sense='outside')
      self.spatial=spatial
      self.rev=rev
      self.vcut=vcut
      self.mcut=mcut
      self.verb=verb
      self.master.verb = verb
      self.do_sex=do_sex
      self.sexdir = sexdir
      self.sexcmd = sexcmd
      self.starmin = starmin
      self.maxdist = maxdist
      self.thresh = thresh
      self.master.do_sex=do_sex
      self.master.sexdir = sexdir
      self.master.sexcmd = sexcmd
      self.master.thresh = thresh
      self.master.crowd = crowd
      #self.imread(bs=bs, usewcs=usewcs, blur=blur)
      self.imread(bs=bs)
      self.readcat(magcat, racol, deccol, magcol)
      self.master.readcat(magcat, racol, deccol, magcol)
      if self.snpos is not None:
         snx,sny = self.snpos.topixel()
      else:
         snx,sny = None,None
      self.crowd = crowd

      if starlist is not None:
         self.starlist = np.loadtxt(starlist)
      else:
         self.starlist = None

      if not registered and not usewcs:
         self.compute(nmax=nmax, min_sep=min_sep)
         self.master.compute(nmax=nmax, min_sep=min_sep)
         res = self.objmatch(nord=nord,perr=perr,angles=angles,
               interactive=interactive, use_db=use_db, iter=Niter)
         if res < 0:
            self.log('Failed object match... giving up.')
            return -1
      if blur is not None:
         self.blurImage(blur)
      self.mktemplate(registered, usewcs=usewcs)
      self.mkweight()
      if self.verb: qdump(self.weight, self.wt, self.image)

      if self.verb: qdump(self.rectemp,(self.timage-self.tbg)*\
            (self.exptime/self.master.exptime), extras=[('BACKGND',self.tbg)])
      if self.tsigma is not None and self.verb:
         qdump(self.recsigma,self.tsigma*(self.exptime/self.master.exptime),
            self.master.image)
      if self.verb: 
         qdump(self.masked,np.greater(self.wt,0)*(self.data),self.image,
            extras=[('BACKGND',self.bg)])

      if match:
         if self.rev == 'auto':
            # Do both forward and reverse and figure out which kernel
            # is more well-behaved (not de-convolving)
            self.rev = False
            self.mkmatch(preserve,quick_convolve=quick_convolve)
            rdv1 = self.rdv0
            forwardmatch = self.match
            self.rev = True
            self.mkmatch(preserve, quick_convolve=quick_convolve)
            rdv2 = self.rdv0

            # Condistion we're testing:  rdvs are larger for deconvolve
            idx = len(rdv1)//2
            print("AUTO:",np.sum(rdv1[:idx]<rdv2[:idx]), 
                          np.sum(rdv1[:idx]<rdv2[idx])/idx)
            if np.sum(rdv1[:idx] < rdv2[:idx])/len(rdv1[:idx]) > 0.5:
               # Switch back to forward
               self.log("AUTO:  Using forward matching")
               self.rev = False
               self.match = forwardmatch
            else:
               self.log("AUTO:  Using reverse matching")

         else:
            self.mkmatch(preserve,quick_convolve=quick_convolve)
            # Some stats. First, significant negative pixels in the kernel
            if self.pwid > -1:
               ll = len(self.ii)
               kbg = np.median(self.psf1[:ll])
               ksd = 1.49*np.median(np.absolute(self.psf1[:ll]-kbg))
               nneg = np.sum(np.less(self.psf1[:ll],kbg-3*ksd))
               self.log("Number of significant negative kernel "\
                         "pixels = {}({}%)".format(nneg, nneg/ll*100))

         if self.verb:
            if self.rev:
               qdump(self.template,
                     (self.timage-self.tbg)/self.fluxrat*\
                     self.exptime/self.master.exptime + self.bg,self.image, 
                     extras=[('BACKGND',self.bg)])
            else:
               qdump(self.template,(self.match + self.bg),self.image,
                     extras=[('BACKGND',self.bg)])
         if self.pwid > -1 and self.verb > 1: qdump(self.psfgrid,self.grid)
      if subt:
         ids = np.indices(self.match.shape)
         if self.rev: 
            if diff_size is not None and snx is not None and sny is not None:
               dists = np.sqrt(np.power(ids[1]-snx,2) + \
                       np.power(ids[0]-sny,2))
               diff = (self.match + self.bg) - \
                     np.where(np.less(dists,diff_size/self.scale),
                              (self.timage - self.tbg)*\
                              self.exptime/self.master.exptime/self.fluxrat, 0)
               fulldiff = ((self.match + self.bg) - \
                     (self.timage - self.tbg)*\
                     self.exptime/self.master.exptime/self.fluxrat)
            else:
               diff = ((self.match + self.bg) - \
                     (self.timage-self.tbg)*\
                     self.exptime/self.master.exptime/self.fluxrat)
               fulldiff = diff
            qdump(self.difference,diff,self.image, extras=[('BACKGND',self.bg)])
            if self.verb: qdump(self.resids,np.greater(self.wt,0)*fulldiff,self.image)
            if snx is not None and sny is not None:
               if self.verb: qdump(self.diffpostage, 
                     (self.match-self.timage/self.fluxrat)\
                     [int(sny)-200:int(sny)+201,int(snx)-200:int(snx)+201],
                     self.image)
               if self.verb: qdump(self.SNpostage, (self.match)\
                     [int(sny)-200:int(sny)+201,int(snx)-200:int(snx)+201],
                     self.image)
            if self.verb: qdump(self.SN, (self.match + self.bg), self.image, 
                  extras=[('BACKGND',self.bg)])
            if self.sigma is not None and self.verb:
               qdump(self.SN.replace('.fits','_sigma.fits'), self.sigma, 
                     self.image)
            if self.sigimage is not None and self.master.sigimage is not None:
               # Make the noise map of the difference image
               var = np.power(self.csigma, 2) + np.power(self.master.sigma, 2)
               if self.verb: qdump(self.difference.replace('.fits','_sigma.fits'), 
                     np.sqrt(var), self.image)
         else: 
            if diff_size is not None and snx is not None and sny is not None:
               dists = np.sqrt(np.power(ids[1]-snx,2) + \
                               np.power(ids[0]-sny,2))
               diff = self.data - \
                     np.where(np.less(dists,diff_size/self.scale),
                              self.match, 0)
               if self.sigimage is not None and self.master.sigimage is not None:
                  var = np.power(self.sigma, 2) + \
                        np.power(np.where(np.less(dists,diff_size/self.scale),
                                 self.csigma, 0),2)
               fulldiff = (self.data - self.match)
            else:
               diff = (self.data-self.match)
               fulldiff = diff
               if self.sigimage is not None and self.master.sigimage is not None:
                  var = np.power(self.csigma, 2) + np.power(self.sigma, 2) 
            qdump(self.difference,diff,self.image, extras=[('BACKGND',self.bg)])
            if self.verb: qdump(self.resids,np.greater(self.wt,0)*fulldiff,self.image)
            if snx is not None and sny is not None and self.verb:
               qdump(self.diffpostage, (self.data-self.match)\
                     [int(sny)-200:int(sny)+201,int(snx)-200:int(snx)+201],
                     self.image)
               qdump(self.SNpostage, (self.data)\
                     [int(sny)-200:int(sny)+201,int(snx)-200:int(snx)+201],
                     self.image)
            if self.verb: qdump(self.SN,self.data,self.image)
            if self.sigimage is not None and self.master.sigimage is not None:
               # Make the noise map of the difference image
               qdump(self.difference.replace('.fits','_sigma.fits'), 
                     np.sqrt(var), self.image)
      
      if plt is not None:
         self.plot_diff()
      return 0

   def plot_diff(self):

      fig,axs = plt.subplots(1,3,figsize=(15,5), 
            dpi=rcParams['figure.dpi']*0.7)
      plt.subplots_adjust(wspace=0)
      #if self.rev:
      #   data = self.match + self.bg
      #   temp = (self.timage - self.tbg)*self.exptime/self.master.exptime \
      #         + self.bg
      #   diff = (self.match + self.bg) - (self.timage - self.tbg)*\
      #         self.exptime/self.master.exptime/self.fluxrat
      #else:
      #   data = self.data
      #   temp = (self.match + self.bg)
      #   diff = (self.data - self.match)
      if self.rev:
         data = self.match
         temp = (self.timage - self.tbg)*self.exptime/self.master.exptime
         diff = self.match - (self.timage - self.tbg)*\
               self.exptime/self.master.exptime/self.fluxrat
      else:
         data = self.data - self.bg
         temp = self.match
         diff = self.data - self.bg - self.match

      if self.snpos is not None:
         snx,sny = self.snpos.topixel()
         snx = int(snx)
         sny = int(sny)
         sno = data[sny-200:sny+200,snx-200:snx+200]
         snt = temp[sny-200:sny+200,snx-200:snx+200]
         snd = diff[sny-200:sny+200,snx-200:snx+200]
      else:
         sno = data
         snt = temp
         snd = diff

      norm = simple_norm(snd, percent=99)
      axs[0].imshow(sno, origin='lower', norm=norm, cmap='gray_r')
      axs[1].imshow(snt, origin='lower', norm=norm, cmap='gray_r')
      axs[2].imshow(snd, origin='lower', norm=norm, cmap='gray_r')
      
      # Plot the kernel as an inset to the convolved image
      xidx = int(self.data.shape[1]/32)
      yidx = int(self.data.shape[0]/32)

      if self.pwid > -1:
         pdata = self.grid[yidx-self.pwid:yidx+self.pwid,
                           xidx-self.pwid:xidx+self.pwid]
      else:
         pdata = np.zeros((11,11))
         pdata[5,5] = 1.0
      norm = simple_norm(pdata, percent=99.8)
      if self.rev:
         inax = axs[0].inset_axes([0.8,0.8,0.18,0.18])
      else:
         inax = axs[1].inset_axes([0.8,0.8,0.18,0.18])
      inax.imshow(pdata, origin='lower', norm=norm, cmap='gray_r')
      inax.xaxis.set_visible(False)
      inax.yaxis.set_visible(False)

      axs[1].set_yticklabels([])
      axs[2].set_yticklabels([])

      axs[0].set_xlabel('Science')
      axs[1].set_xlabel('Reference')
      axs[2].set_xlabel('Difference')
      fig.tight_layout()

      fig.savefig(self.SN.replace('.fits', '_diff.jpg'))


