# ImageMatch
A tool for rectifying, PSF-matching, and subtracting one astronomical image from another

This is most useful for folks who work in transient astronomy.

## The problem (why you might want to use this software)

The (roughly) same field is observed on two different epochs. You want to subtract one from the
other to see what has changed. But the images won't be perfectly aligned. And the seeing
(blurring by atmosphere) might be worse on one night. Or maybe you've used two different
telescopes with different plate scales, seeing conditions, etc.

## The Solution (what this software does)

- Rectify the image by matching point-sources from one image to the other and solving for
  a geometric transformation from one to the other.
  
- Convolve one image with a "seeing kernel" that will blur it so that they match, giving
  clean subtractions. ImageMatch models this kernel as an NxN matrix, so can have any
  strange shape the data requires (and we've seen some steeeerange kernels!).
  
## Requirements

These are all available through anaconda or most other python distributions.

- python 2.7 or 3.x
- astropy 4+
- numpy/scipy
- vtk

In addition, you will need to install `source extractor`(https://www.astromatic.net/software/sextractor)
