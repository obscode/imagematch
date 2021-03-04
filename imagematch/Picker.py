'''This is a class that gives you a side-by-side view of two
FITS images and lets you choose corresponding objects in either.
Each time you click, the number increments.'''

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle
import numpy as np
import sys,os,string,re
from astropy.io import fits
from astropy.visualization import simple_norm

def autoscale(data, pmin, pmax):
   hist,bins = np.histogram(np.ravel(data), bins=1024)
   csum = np.cumsum(hist)
   psum = csum*1.0/csum[-1]*100
   id = np.nonzero(np.greater(psum, pmin))[0][0]
   vmin = bins[id]
   id = np.nonzero(np.greater(psum, pmax))[0][0]
   vmax = bins[id]
   return vmin,vmax

class FITSpicker:

   def __init__(self, im1, im2, scale1=1.0, scale2=1.0,
         equal_scale=True, x1s=None, y1s=None, x2s=None, y2s=None,
         recenter='max', box=20):

      if isinstance(im1, str):
         self.im1 = fits.open(im1)
      else:
         self.im1 = im1
      if isinstance(im2, str):
         self.im2 = fits.open(im2)
      else:
         self.im2 = im2

      self.data1 = self.im1[0].data
      self.data2 = self.im2[0].data

      self.scale1 = scale1
      self.scale2 = scale2

      self.recenter = recenter
      self.box = box

      self._x1s = x1s
      self._x2s = x2s
      self._y1s = y1s
      self._y2s = y2s

      self.x1s = []
      self.y1s = []
      self.circ1s = []
      self.lab1s = []
      self.x2s = []
      self.y2s = []
      self.circ2s = []
      self.lab2s = []

      self.fig = plt.figure(figsize=(15,7))
      self.ax1 = self.fig.add_axes([0.1,0.1,0.35,0.8])
      self.ax2 = self.fig.add_axes([0.5,0.1,0.35,0.8])
      self.ax1.imshow(self.data1, cmap=cm.gray_r, 
            norm=simple_norm(self.data1, percent=99),
            extent=[0, self.data1.shape[1]*scale1,self.data1.shape[0]*scale1,0])
      self.ax2.imshow(self.data2, cmap=cm.gray_r, 
            norm=simple_norm(self.data2, percent=99),
            extent=[0, self.data2.shape[1]*scale2,self.data2.shape[0]*scale2,0])
      self.ax2.text(1.05, 0.5, 
            'a = pick obj.\nx = flip x\ny = flip y\nd = delete obj',
            transform = self.ax2.transAxes)
      if self._x1s is not None:
         self.ax1.plot(self._x1s*scale1, self._y1s*scale1, 'o', ms=15,
               mec='red', alpha=0.2, mfc='none')
         self.ax2.plot(self._x2s*scale2, self._y2s*scale2, 'o', ms=15,
               mec='red', alpha=0.2, mfc='none')
      self.fig.canvas.mpl_connect('key_press_event', self.onKeyPress)
      if equal_scale:
         self.equal_scale()
      plt.show()

   def centroid(self, i, x, y, box=20, mode='max'):
      data = [self.data1,self.data2][i]
      scale = [self.scale1,self.scale2][i]

      ii,jj = int(x/scale),int(y/scale) 
      subd = data[jj-box:jj+box+1,ii-box:ii+box+1]
      np.savetxt('subd.dat', subd)
      jjj,iii = np.indices(subd.shape)
      subd = subd.ravel()
      iii = iii.ravel()
      jjj = jjj.ravel()
      np.savetxt('iii.dat', iii)
      np.savetxt('jjj.dat', jjj)
      if mode == 'max':
         idx = np.argmax(subd)
         i0,j0 = iii[idx], jjj[idx]
      elif mode == 'com':
         i0 = sum(iii*subd)/sum(subd)
         j0 = sum(jjj*subd)/sum(subd)
       
      return ii-box+i0, jj-box+j0


   def get_points(self):
      return (np.array(self.x1s)/self.scale1,np.array(self.y1s)/self.scale1,
             np.array(self.x2s)/self.scale2,np.array(self.y2s)/self.scale2)

   def equal_scale(self):
      '''Set the physica size of the images equal'''
      if self.data1.shape[0]*self.scale1 > self.data2.shape[0]*self.scale2:
         self.ax2.set_xlim(self.ax1.get_xlim())
         self.ax2.set_ylim(self.ax1.get_ylim())
      else:
         self.ax1.set_xlim(self.ax2.get_xlim())
         self.ax1.set_ylim(self.ax2.get_ylim())
      plt.draw()


   def add_object(self, i, x, y, refresh=True):
      ax = [self.ax1,self.ax2][i]
      cs = [self.circ1s,self.circ2s][i]
      ls = [self.lab1s, self.lab2s][i]
      xs = [self.x1s,self.x2s][i]
      ys = [self.y1s,self.y2s][i]
      data = [self.data1,self.data2][i]
      scale = [self.scale1,self.scale2][i]

      _x = [self._x1s, self._x2s][i]
      _y = [self._y1s, self._y2s][i]

      if _x is not None and _y is not None:
         idx = np.argmin(np.power(_x-x/scale,2) + np.power(_y-y/scale,2))
         x,y = _x[idx]*scale,_y[idx]*scale
      elif self.recenter:
         x,y = self.centroid(i, x,y,self.box, self.recenter)
         x = x*scale
         y = y*scale

      cs.append(Circle((x,y), radius=3, ec='red', fill=False))
      ax.add_patch(cs[-1])
      x0,x1 = ax.get_xlim()
      y0,y1 = ax.get_ylim()
      dx = (x1-x0)/100
      dy = (y1-y0)/100
      if dx > 0:
         halign = 'left'
      else:
         halign = 'right'
      if dy > 0:
         valign = 'bottom'
      else:
         valign = 'top'
      ls.append(ax.text(x+3, y+3, len(xs)+1, color='red',va=valign,ha=halign))
      xs.append(x/scale)
      ys.append(y/scale)

      if refresh:
         plt.draw()

   def delete_obj(self, i, x, y):
      # First, find the object closest to x,y
      ax = [self.ax1,self.ax2][i]
      cs = [self.circ1s,self.circ2s][i]
      ls = [self.lab1s, self.lab2s][i]
      xs = [self.x1s,self.x2s][i]
      ys = [self.y1s,self.y2s][i]
      scale = [self.scale1,self.scale2][i]

      dists = np.power(np.array(xs)-x/scale,2)+np.power(np.array(ys)-y/scale,2)
      id = np.argmin(dists)
      del xs[id]
      del ys[id]
      print(xs,ys)
      self.redraw()

   def redraw(self):
      # Clear out the old suff
      [circ.remove() for circ in self.circ1s + self.circ2s]
      self.circ1s = []
      self.circ2s = []
      [lab.remove() for lab in self.lab1s + self.lab2s]
      self.lab1s = []
      self.lab2s = []
      x1s = self.x1s*1; self.x1s = []
      x2s = self.x2s*1; self.x2s = []
      y1s = self.y1s*1; self.y1s = []
      y2s = self.y2s*1; self.y2s = []
      for i in range(len(x1s)):
         self.add_object(0,x1s[i]*self.scale1, y1s[i]*self.scale1, 
              refresh=False)
      for i in range(len(x2s)):
         self.add_object(1,x2s[i]*self.scale2, y2s[i]*self.scale2, 
              refresh=False)
      plt.draw()


   def onKeyPress(self, event):
      if event.key == 'x':
         event.inaxes.set_xlim(event.inaxes.get_xlim()[::-1])
         self.redraw()
      elif event.key == 'y':
         event.inaxes.set_ylim(event.inaxes.get_ylim()[::-1])
         self.redraw()
      elif event.key == 'a':
         if event.inaxes is self.ax1:
            self.add_object(0, event.xdata, event.ydata)
         else:
            self.add_object(1, event.xdata, event.ydata)
      elif event.key == 'd':
         if event.inaxes is self.ax1:
            self.delete_obj(0, event.xdata, event.ydata)
         else:
            self.delete_obj(1, event.xdata, event.ydata)
      elif event.key == 'q':
         plt.close(self.fig)


