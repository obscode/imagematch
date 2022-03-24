'''Basis vectors and matrices for solving linear equations.'''
from numpy import *
from numpy.linalg import svd, lstsq
from .npextras import *


def svdfit(b,y):
#      stuff = lstsq(b,y, rcond=None)
#      return stuff[0]

      decomp = svd(b, full_matrices=False)
      sol1 = transpose(decomp[2])
      sol2 = divz(1.0,decomp[1])
      sol3 = dot(transpose(decomp[0]),y)
      if sometrue(sol3):
        solr = (sol2*sol3)
        soll = dot(sol1,solr)
      else:
        soll = zeros(sol3.shape)
      return soll

def abasis(nord,ix,iy,rot=0):
   I=ones(ix.shape)
   Z=zeros(ix.shape)
   ux = ix
   uy = iy
   if nord == 0:
    sb = [ I,  Z,  ux,  uy]
    if not rot: eb = [ Z,  I,  uy,  ux]
    else: eb = [ Z,  I, uy,  -ux]
   if nord >= 1:
    sb = [ I,  Z, ux,   Z,   uy,  Z]
    eb = [ Z,  I,  Z,  ux,    Z, uy]
   if nord >= 2:
    ux = ix/1e3
    uy = iy/1e3
    ux2 = power(ux,2)
    uy2 = power(uy,2)
    sb2 = [ ux2, Z, ux*uy, Z, uy2, Z]
    eb2 = [Z, ux2, Z, ux*uy, Z, uy2]
    sb2 = [1e6*sb1 for sb1 in sb2]
    eb2 = [1e6*eb1 for eb1 in eb2]
    sb = sb + sb2
    eb = eb + eb2
   if nord >= 3:
    ux3 = ux2*ux
    uy3 = uy2*uy
    sb3 = [ ux3, Z, ux2*uy, Z, ux*uy2, Z, uy3, Z]
    eb3 = [Z, ux3, Z, ux2*uy, Z, ux*uy2, Z, uy3]
    sb3 = [1e9*sb1 for sb1 in sb3]
    eb3 = [1e9*eb1 for eb1 in eb3]
    sb = sb + sb3
    eb = eb + eb3
   if nord >= 4:
    ux4 = ux2*ux2
    uy4 = uy2*uy2
    sb4 = [ ux4, Z, ux3*uy, Z, ux2*uy2, Z, ux*uy3, Z, uy4, Z]
    eb4 = [Z, ux4, Z, ux3*uy, Z, ux2*uy2, Z, ux*uy3, Z, uy4]
    sb4 = [1e12*sb1 for sb1 in sb4]
    eb4 = [1e12*eb1 for eb1 in eb4]
    sb = sb + sb4
    eb = eb + eb4
   if nord >= 5:
    ux5 = ux3*ux2
    uy5 = uy3*uy2
    sb5 = [ ux5, Z, ux4*uy, Z, ux3*uy2, Z, ux2*uy3, Z, ux*uy4, Z, uy5, Z]
    eb5 = [Z, ux5, Z, ux4*uy, Z, ux3*uy2, Z, ux2*uy3, Z, ux*uy4, Z, uy5]
    sb5 = [1e15*sb1 for sb1 in sb5]
    eb5 = [1e15*eb1 for eb1 in eb5]
    sb = sb + sb5
    eb = eb + eb5
   if nord >= 6:
    ux6 = ux3*ux3
    uy6 = uy3*uy3
    sb6 = [ ux6, Z, ux5*uy,   Z, ux4*uy2, Z, ux3*uy3, Z, ux2*uy4, Z, ux*uy5, Z, uy6, Z]
    eb6 = [Z, ux6, Z, ux5*uy, Z, ux4*uy2, Z, ux3*uy3, Z, ux2*uy4, Z, ux*uy5, Z, uy6]
    sb6 = [1e18*sb1 for sb1 in sb6]
    eb6 = [1e18*eb1 for eb1 in eb6]
    sb = sb + sb6
    eb = eb + eb6
   sb = transpose(sb); eb = transpose(eb)
   return concatenate([sb,eb])

def mbasis(co,ux,uy,rot=0):
   if len(shape(ux)) == 0:
      ux = asarray([ux]); uy = asarray([uy]); oo = 1
   else: oo = 0
   nord = len(co)
   if "evaluate" in globals():
      if nord == 4:
         se = "%r + %r*ux + %r*uy" % (co[0],co[2],co[3])
         ee = "%r + %r*uy + %r*ux" % (co[1],co[2],pow(-1,rot)*co[3])
         sb = eval(se)
         eb = eval(ee)
      if nord >= 6:
         se = "%r + %r*ux + %r*uy" % (co[0],co[2],co[4])
         ee = "%r + %r*ux + %r*uy" % (co[1],co[3],co[5])
         if nord >= 12:
            se = se + " + %r*ux**2 + %r*ux*uy + %r*uy**2" % (co[6],co[8],co[10])
            ee = ee + " + %r*ux**2 + %r*ux*uy + %r*uy**2" % (co[7],co[9],co[11])
            if nord >= 20:
               se = se + " + %r*ux**3 + %r*ux**2*uy + %r*ux*uy**2 + %r*uy**3" % (co[12],co[14],co[16],co[18])
               ee = ee + " + %r*ux**3 + %r*ux**2*uy + %r*ux*uy**2 + %r*uy**3" % (co[13],co[15],co[17],co[19])
               if nord >= 30:
                  se = se + " + %r*ux**4 + %r*ux**3*uy + %r*ux**2*uy**2 + %r*ux*uy**3 + %r*uy**4" % (co[20],co[22],co[24],co[26],co[28])
                  ee = ee + " + %r*ux**4 + %r*ux**3*uy + %r*ux**2*uy**2 + %r*ux*uy**3 + %r*uy**4" % (co[21],co[23],co[25],co[27],co[29])
                  if nord >= 42:
                     se = se + " + %r*ux**5 + %r*ux**4*uy + %r*ux**3*uy**2 + %r*ux**2*uy**3 + %r*ux*uy**4 + %r*uy**5" % (co[30],co[32],co[34],co[36],co[38],co[40])
                     ee = ee + " + %r*ux**5 + %r*ux**4*uy + %r*ux**3*uy**2 + %r*ux**2*uy**3 + %r*ux*uy**4 + %r*uy**5" % (co[31],co[33],co[35],co[37],co[39],co[41])
                     if nord >= 56:
                        se = se + " + %r*ux**6 + %r*ux**5*uy + %r*ux**4*uy**2 + %r*ux**3*uy**3 + %r*ux**2*uy**4 + %r*ux*uy**5 + %r*uy**6" % (co[42],co[44],co[46],co[48],co[50],co[52],co[54])
                        ee = ee + " + %r*ux**6 + %r*ux**5*uy + %r*ux**4*uy**2 + %r*ux**3*uy**3 + %r*ux**2*uy**4 + %r*ux*uy**5 + %r*uy**6" % (co[43],co[45],co[47],co[49],co[51],co[53],co[55])
         sb = eval(se)
         eb = eval(ee)
   else:
      I=ones(ux.shape,ux.dtype)
      Z=zeros(ux.shape,ux.dtype)
      sb = zeros(ux.shape,ux.dtype)
      eb = zeros(ux.shape,ux.dtype)
      if nord == 4:
       sb = sb + co[0]; sb = sb + co[2]*ux; sb = sb + co[3]*uy
       eb = eb + co[1]; eb = eb + co[2]*uy; eb = eb + pow(-1,rot)*co[3]*ux
      if nord >= 6:
       sb = sb + co[0]; sb = sb + co[2]*ux; sb = sb + co[4]*uy
       eb = eb + co[1]; eb = eb + co[3]*ux; eb = eb + co[5]*uy
      if nord >= 12:
       ux2 = power(ux,2)
       uy2 = power(uy,2)
       sb = sb + co[6]*ux2; sb = sb + co[8]*ux*uy; sb = sb + co[10]*uy2
       eb = eb + co[7]*ux2; eb = eb + co[9]*ux*uy; eb = eb + co[11]*uy2
      if nord >= 20:
       ux3 = ux2*ux
       uy3 = uy2*uy
       sb = sb + co[12]*ux3; sb = sb + co[14]*ux2*uy;
       sb = sb + co[16]*ux*uy2; sb = sb + co[18]*uy3
       eb = eb + co[13]*ux3; eb = eb + co[15]*ux2*uy
       eb = eb + co[17]*ux*uy2; eb = eb + co[19]*uy3
      if nord >= 30:
       ux4 = ux2*ux2
       uy4 = uy2*uy2
       sb = sb + co[20]*ux4; sb = sb + co[22]*ux3*uy
       sb = sb + co[24]*ux2*uy2; sb = sb + co[26]*ux*uy3
       sb = sb + co[28]*uy4; eb = eb + co[21]*ux4
       eb = eb + co[23]*ux3*uy; eb = eb + co[25]*ux2*uy2
       eb = eb + co[27]*ux*uy3; eb = eb + co[29]*uy4
      if nord >= 42:
       ux5 = ux3*ux2
       uy5 = uy3*uy2
       sb = sb + co[30]*ux5; sb = sb + co[31]*ux4*uy; sb = sb + co[32]*ux3*uy2
       sb = sb + co[33]*ux2*uy3; sb = sb + co[34]*ux*uy4; sb = sb + co[35]*uy5
       eb = eb + co[36]*ux5; eb = eb + co[37]*ux4*uy; eb = eb + co[38]*ux3*uy2
       eb = eb + co[39]*ux2*uy3; eb = eb + co[40]*ux*uy4; eb = eb + co[41]*uy5
      if nord >= 56:
       ux6 = ux3*ux3
       uy6 = uy3*uy3
       sb = sb + co[42]*ux6; sb = sb + co[43]*ux5*uy; sb = sb + co[44]*ux4*uy2
       sb = sb + co[45]*ux3*uy3
       sb = sb + co[46]*ux2*uy4; sb = sb + co[47]*ux*uy5; sb = sb + co[48]*uy6
       eb = eb + co[49]*ux6; eb = eb + co[50]*ux5*uy; eb = eb + co[51]*ux4*uy2
       eb = eb + co[52]*ux3*uy3
       eb = eb + co[53]*ux2*uy4; eb = eb + co[54]*ux*uy5; eb = eb + co[55]*uy6
   if oo: eb=eb[0]; sb=sb[0]
   return [sb,eb]

