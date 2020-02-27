'''A module to help translate back and forth from python layer to VTK'''

import vtk
from vtk.util import numpy_support
import numpy as np


def NumToVTKArray(arr, name=None):
   __typeDict = {np.uint8:vtk.vtkUnsignedCharArray,
         np.byte:vtk.vtkUnsignedCharArray,
         np.bool:vtk.vtkUnsignedCharArray,
         np.int8:vtk.vtkUnsignedCharArray,
         np.int16:vtk.vtkShortArray,
         np.uint16:vtk.vtkUnsignedShortArray,
         np.int32:vtk.vtkIntArray, np.int64:vtk.vtkLongArray,
         np.float32:vtk.vtkFloatArray,
         np.float64:vtk.vtkDoubleArray, 
         np.complex64:vtk.vtkFloatArray, 
         np.complex128:vtk.vtkDoubleArray }
   nt = arr.dtype
   for td in __typeDict.keys():
      if td == nt: utd=td;break
   vtkarray = __typeDict[utd]()
   dims = arr.shape
   vtkarray.SetVoidArray(arr.ravel(),np.multiply.reduce(dims),1)
   if len(dims) > 1: vtkarray.SetNumberOfComponents(dims[0])
   else: vtkarray.SetNumberOfComponents(1)
   vtkarray.SetNumberOfValues(np.multiply.reduce(dims))
   if name: vtkarray.SetName(name)
   return vtkarray

def VTKArrayToVTKImage(v,nc=-1,wid=-1,hei=-1):
   ii = vtk.vtkImageData()

   shape2 = [v.GetNumberOfComponents(), v.GetNumberOfTuples()]
   ii.SetDimensions(shape2[1], shape2[0], 1)
   ii.AllocateScalars(v.GetDataType(), 1)

   ii.GetPointData().SetScalars(v)
   if wid>0 and hei>0:
      ii.SetExtent(0,wid-1,0,hei-1,0,0)
   return ii

def NumToVTKImage(numarray,name=None):
   dims = numarray.shape
   if len(dims) ==1:
      return NumToVTKArray(numarray, name)
   ii = vtk.vtkImageData()
   ii.SetDimensions(numarray.shape[0], numarray.shape[1], 1)
   ii.SetSpacing(1,1,1)
   ii.SetOrigin(0,0,0)
   ii.SetExtent(0, numarray.shape[0]-1, 0, numarray.shape[1]-1, 0, 0)
   vtktype = numpy_support.get_vtk_array_type(numarray.dtype)
   ii.AllocateScalars(vtktype, 1)
   pd = ii.GetPointData()
   arr = numpy_support.numpy_to_vtk(np.ndarray.flatten(numarray, 'F'))
   pd.SetScalars(arr)
   return ii

def VTKImageToNum(i):
   __typeDict = { vtk.VTK_ID_TYPE: np.int64, 
         vtk.VTK_CHAR:np.int8, 
         vtk.VTK_UNSIGNED_CHAR:np.uint8,
         vtk.VTK_SHORT:np.int16, 
         vtk.VTK_UNSIGNED_SHORT:np.uint16, 
         vtk.VTK_INT:np.int32, 
         vtk.VTK_FLOAT:np.float32, 
         vtk.VTK_DOUBLE:np.float64, 
         vtk.VTK_LONG:np.int64}
   ie = vtk.vtkImageExport()
   d = list(i.GetDimensions())
   d.reverse()
   if d[0] == 1: d = d[1:]
   if d[0] == 1: d = d[1:]
   #d.reverse()
   it = i.GetScalarType()
   scalars = i.GetPointData().GetScalars()
   if scalars: it = scalars.GetDataType()
   else: it = vtk.VTK_FLOAT
   nat = __typeDict[it]
   x = np.zeros(d,nat)
   ie.SetExportVoidPointer(x)
   ie.ImageLowerLeftOn()
   ie.SetInputData(i)
   ie.Export()
   return np.squeeze(x).T


def VTKImageShift(x,u,v,numret=False,interp=True,wrap=False,mirror=False,
   constant=None,cubic=False):
   if type(x) == type(np.arange(2)): i = NumToVTKImage(x)
   else: i = x
   if 0 and int(u) == u and int(v) == v:
      s = vtk.vtkImageTranslateExtent()
      s.SetInput(i)
      s.SetTranslation(u,v,0)
      o = s.GetOutput()
      s.Update()
   else:
      s = vtk.vtkImageReslice()
      s.AutoCropOutputOn()
      if wrap: s.WrapOn()
      else: s.WrapOff()
      if mirror or constant != None: s.MirrorOn()
      else: s.MirrorOff()
      if interp:
        s.InterpolateOn()
        if cubic:
          s.SetInterpolationModeToCubic()
        else:
          s.SetInterpolationModeToLinear()
      else: s.InterpolateOff()
      s.OptimizationOn()
      s.SetOutputOrigin(u,v,0)
      s.SetInputData(i)
      o=s.GetOutput()
      s.Update()
   if numret: return VTKImageToNum(o)
   else: return o


def VTKImageArith(x1,x2,op,numret=False,rep=0.0):
      if type(x1) == np.ndarray and type(x2) == np.ndarray:
        if x1.dtype == np.float64 or x2.dtype == np.float64:
          x1,x2 = x1.astype(np.float64), x2.astype(np.float64)
        elif x1.dtype == np.float32 or x2.dtype == np.float32:
          x1,x2 = x1.astype(np.float32), x2.astype(np.float32)
      if type(x1) == np.ndarray: i1 = NumToVTKImage(x1)
      else: i1 = x1
      if type(x2) == np.ndarray: i2 = NumToVTKImage(x2)
      else: i2 = x2
      m = vtk.vtkImageMathematics()
      numops = [ "multiply","divide","add","subtract"]
      if op == "divide":
        m.SetConstantC(rep); m.DivideByZeroToCOn()
      if type(x1) == np.ndarray and type(x2) == np.ndarray:
         m.SetInput1Data(i1)
         m.SetInput2Data(i2)
         vtkops = [ m.SetOperationToMultiply, m.SetOperationToDivide, m.SetOperationToAdd, m.SetOperationToSubtract]
      elif type(x1) == np.ndarray:
         m.SetInput1Data(i1)
         if i2 == 0:
            const = 0.0
         else:
            const = [i2,1.0/i2,i2,-i2][numops.index(op)]
         m.SetConstantK(const)
         m.SetConstantC(const)
         vtkops = [ m.SetOperationToMultiplyByK, m.SetOperationToMultiplyByK, m.SetOperationToAddConstant, m.SetOperationToAddConstant]
      if type(x1) != np.ndarray and type(x2) != np.ndarray:
         return eval(op)(x1,x2)
      else:
         vtkops[numops.index(op)]()
         o = m.GetOutput()
         m.Update()
         if numret: return VTKImageToNum(o)
         else: return o

def VTKAdd(x1,x2): return VTKImageArith(x1,x2,"add",numret=True)
def VTKSubtract(x1,x2): return VTKImageArith(x1,x2,"subtract",numret=True)
def VTKMultiply(x1,x2): return VTKImageArith(x1,x2,"multiply",numret=True)
def VTKDivide(x1,x2,rep=0): return VTKImageArith(x1,x2,"divide",numret=True,
      rep=rep)


def VTKImageLogic(x1,x2=None,op="or",numret=False):
      if type(x1) == np.ndarray: i1 = NumToVTKImage(x1)
      else: i1 = x1
      if type(x2) == np.ndarray: i2 = NumToVTKImage(x2)
      elif x2: i2 = x2
      m = vtk.vtkImageLogic()
      numops = [ "and","or","xor","nand","nor","not"]
      vtkops = [ m.SetOperationToAnd, m.SetOperationToOr, m.SetOperationToXor, m.SetOperationToNand, m.SetOperationToNor, m.SetOperationToNot]
      m.SetOutputTrueValue(1.0)
      vtkops[numops.index(op)]()
      m.SetInput1Data(i1)
      if x2 is not None: m.SetInput2Data(i2)
      o = m.GetOutput()
      m.Update()
      if numret: return VTKImageToNum(o)
      else: return o

def VTKOr(x1,x2): return VTKImageLogic(x1,x2=x2,op="or",numret=1)
def VTKAnd(x1,x2): return VTKImageLogic(x1,x2=x2,op="and",numret=1)
def VTKNot(x1): return VTKImageLogic(x1,x2=None,op="not",numret=1)

def VTKGrowN(x,n=2,m=None,numret=False,origsize=None,interp=False,wrap=False,
      constant=None,mirror=False,xshift=0,yshift=0):
      if type(x) == np.ndarray: i = NumToVTKImage(x)
      else: i = x
      d = i.GetDimensions()
      if not m: m = n
      if d[0] == 1: n = 1
      if d[1] == 1: m = 1
      im = vtk.vtkImageMagnify()
      im.SetMagnificationFactors(n,m,1)
      if interp: im.InterpolateOn()
      else: im.InterpolateOff()
      im.SetInputData(i)
      o = im.GetOutput()
      im.Update()
      o.SetSpacing(1,1,1)
      #o.Update()
      o = VTKImageShift(o,xshift,yshift,interp=True,wrap=wrap,constant=constant,
            mirror=mirror)
      if origsize is not None:
            if len(origsize) == 1: origsize=(1,)+origsize
            p = vtk.vtkImageMirrorPad()
            p.SetOutputWholeExtent(0,origsize[1]-1,0,origsize[0]-1,0,0)
            p.SetInputData(o)
            f = p.GetOutput()
            p.Update()
            o = f
      if numret: return VTKImageToNum(o)
      else: return o


def VTKImageTransform(x,dx,dy,numret=False,reverse=False,origsize=None,
      cubic=False,interp=True,scalex=1,scaley=1,constant=0,wrap=False,
      mirror=False):

      maxdx = max([int(max(abs(dx.ravel()))+1),(dx.shape[1]-x.shape[1])])
      maxdy = max([int(max(abs(dy.ravel()))+1),(dx.shape[0]-x.shape[0])])
      dx = dx.astype(np.float32)
      dy = dy.astype(np.float32)
      if scalex > 1:
         xx = np.arange(x.shape[1])
         dx = (xx[np.newaxis,::]+dx).astype(np.float32)
         dy = VTKGrowN(dy,scalex,1,numret=True)
         dx = VTKGrowN(dx,scalex,1,numret=True)
         dx = (dx-np.arange(scalex*x.shape[1])[np.newaxis,::]).\
               astype(np.float32)
      if scaley > 1:
         yy = np.arange(x.shape[0])
         dy = (yy[::,np.newaxis]+dy).astype(np.float32)
         dy = VTKGrowN(dy,1,scaley,numret=True)
         dx = VTKGrowN(dx,1,scaley,numret=True)
         dy = (dy-np.arange(scaley*x.shape[0])[::,np.newaxis]).\
               astype(np.float32)
      if type(x) == np.ndarray: i = NumToVTKImage(x)
      else: i = x
      if type(dx) == np.ndarray: tx = NumToVTKImage(dx)
      else: tx = dx
      if type(dy) == np.ndarray: ty = NumToVTKImage(dy)
      else: ty = dy
      dm = ty.GetDimensions()
      dz = np.zeros(dy.shape,dy.dtype)
      tz = NumToVTKImage(dz)
      a = vtk.vtkImageAppendComponents()
      a.AddInputData(tx)
      a.AddInputData(ty)
      a.AddInputData(tz)
      a.Update()
      t = a.GetOutput()
      r = vtk.vtkGridTransform()
      r.SetDisplacementGridData(t)
      r.Update()
      s = vtk.vtkImageReslice()
      s.WrapOff()
      s.MirrorOn()
      if interp:
         s.InterpolateOn()
         if cubic: s.SetInterpolationModeToCubic()
         else: s.SetInterpolationModeToLinear()
      s.SetOutputDimensionality(2)
      if reverse:
         r.SetInterpolationModeToLinear()
         r.SetInverseTolerance(0.001)
         r.SetInverseIterations(1000)
         r.DebugOff()
         ir = r.GetInverse()
         ir.SetInterpolationModeToLinear()
         ir.SetInverseTolerance(0.001)
         ir.SetInverseIterations(1000)
         ir.GlobalWarningDisplayOff()
         s.SetResliceTransform(ir)
         s.AutoCropOutputOff()
         if origsize: s.SetOutputExtent(0,origsize[1]-1,0,origsize[0]-1,0,0)
         else: s.SetOutputExtent(0,scalex*dm[0]-1,0,scaley*dm[1]-1,0,0)
      else:
         r.SetInterpolationModeToCubic()
         r.SetInverseTolerance(0.001)
         r.SetInverseIterations(1000)
         r.GlobalWarningDisplayOff()
         s.SetOutputExtent(0,scalex*dm[0]-1,0,scaley*dm[1]-1,0,0)
         s.AutoCropOutputOff()
         s.SetResliceTransform(r)
      if mirror: ip = vtk.vtkImageMirrorPad()
      elif wrap: ip = vtk.vtkImageWrapPad()
      else: ip = vtk.vtkImageConstantPad(); ip.SetConstant(constant)
      ip.SetOutputWholeExtent(0-maxdx,dm[0]-1+maxdx,0-maxdy,dm[1]-1+maxdy,0,0)
      ip.SetInputData(i)
      ip.Update()
      s.SetInputData(ip.GetOutput())
      o=s.GetOutput()
      s.Update()
      if numret: ri = VTKImageToNum(o)
      else: ri = o
      return ri

def VTKConvolve(im,numret=0,k=5,clip=1,x=1,y=1,wrap=0,mirror=1,constant=None,
                kernel=None):
      if type(im) == np.ndarray: i = NumToVTKImage(im)
      else: i = im
      d = i.GetDimensions()
      e = i.GetExtent()
      dtest = np.sort(d)
      if sum(dtest[:2]) == 2: onedim=1
      else: onedim = 0
      if (d[0] == 1): oned = 1
      elif (d[1] == 1): oned = 2
      else: oned = 0
      if x and not y: oned = 2
      if y and not x: oned = 1
      if onedim: oned = 1
      if constant is not None:
          ip = vtk.vtkImageConstantPad()
          if d[0] > 1: x0,x1=-k,d[0]+k-1
          else: x0,x1=0,0
          if d[1] > 1: y0,y1=-k,d[1]+k-1
          else: y0,y1=0,0
          ip.SetOutputWholeExtent(x0,x1,y0,y1,0,0)
          ip.SetConstant(constant)
          ip.SetInputData(i)
          i = ip.GetOutput()
          ip.Update()
      elif mirror:
          ip = vtk.vtkImageMirrorPad()
          if d[0] > 1: x0,x1=-k,d[0]+k-1
          else: x0,x1=0,0
          if d[1] > 1: y0,y1=-k,d[1]+k-1
          else: y0,y1=0,0
          ip.SetOutputWholeExtent(x0,x1,y0,y1,0,0)
          ip.SetInputData(i)
          i = ip.GetOutput()
          ip.Update()
      elif wrap:
          ip = vtk.vtkImageWrapPad()
          if d[0] > 1: x0,x1=-k,d[0]+k-1
          else: x0,x1=0,0
          if d[1] > 1: y0,y1=-k,d[1]+k-1
          else: y0,y1=0,0
          ip.SetOutputWholeExtent(x0,x1,y0,y1,0,0)
          ip.SetInputData(i)
          i = ip.GetOutput()
          ip.Update()
      c = vtk.vtkImageConvolve()
      if kernel is None:
          if k == 3: k1 = np.asarray([0.25,0.5,0.25])
          elif k == 5: k1 = np.asarray([0.0625,0.25,0.375,0.25,0.0625])
          if onedim: ke = np.ones((k,k),np.float64) * k1
          elif not oned: ke = k1[::,np.newaxis]*k1[np.newaxis,::]
          elif oned == 1: ke = np.zeros((k,k),np.float64); ke[::,2] = k1
          elif oned == 2: ke = np.zeros((k,k),np.float64); ke[2] = k1
          ke = np.ravel(ke)
          ke = ke / np.sum(ke)
          if onedim: ke = len(k1)*ke
      else:
          k = kernel.shape[0]
          ke = np.ravel(kernel)
      if k == 3: c.SetKernel3x3(ke)
      if k == 5: c.SetKernel5x5(ke)
      if k == 7: c.SetKernel7x7(ke)
      c.SetInputData(i)
      o = c.GetOutput()
      c.Update()
      if clip:
         ic = vtk.vtkImageClip()
         notx = -(not x) * 0
         noty = -(not y) * 0
         ic.SetOutputWholeExtent(notx,d[0]-1+notx,noty,d[1]-1+noty,0,0)
         ic.SetInputData(o)
         ic.ClipDataOn()
         o = ic.GetOutput()
         ic.Update()
      if numret: return VTKImageToNum(o)
      else: return o

def VTKGauss(x,sigma=5,numret=1,radius=3,sigmax=None,sigmay=None):
      if type(x) == np.ndarray: i = NumToVTKImage(x)
      else: i = x
      if sigmax is None: sigmax=sigma
      if sigmay is None: sigmay=sigma
      d = vtk.vtkImageGaussianSmooth()
      d.SetStandardDeviations(sigmax,sigmay,0)
      d.SetRadiusFactors(radius,radius,1)
      d.SetInputData(i)
      o = d.GetOutput()
      d.Update()
      if numret: return VTKImageToNum(o)
      else: return o

def VTKDilate(x,k=5,l=None,numret=0):
      if type(x) == np.ndarray: i = NumToVTKImage(x)
      else: i = x
      if l is None: l = k
      d = vtk.vtkImageContinuousDilate3D()
      d.SetKernelSize(k,l,1)
      d.SetInputData(i)
      o = d.GetOutput()
      d.Update()
      if numret: return VTKImageToNum(o)
      else: return o

def VTKAbs(x1):
      if type(x1) == np.ndarray: i1 = NumToVTKImage(x1)
      else: i1 = x1
      m = vtk.vtkImageMathematics()
      m.SetOperationToAbsoluteValue()
      m.AddInputData(i1)
      o = m.GetOutput()
      m.Update()
      return VTKImageToNum(o)


def VTKSqrt(x0,rep=0.0):
      bad = np.less(x0,0)
      x1 = VTKAbs(x0)
      if type(x1) == np.ndarray: i1 = NumToVTKImage(x1)
      else: i1 = x1
      m = vtk.vtkImageMathematics()
      m.SetOperationToSquareRoot()
      m.AddInputData(i1)
      o = m.GetOutput()
      m.Update()
      b = VTKImageToNum(o)
      return np.where(bad,rep,b)

def VTKInvert(x1,rep=0.0):
      if type(x1) == np.ndarray: i1 = NumToVTKImage(x1)
      else: i1 = x1
      m = vtk.vtkImageMathematics()
      m.SetOperationToInvert()
      m.AddInputData(i1)
      m.SetConstantC(rep)
      m.DivideByZeroToCOn()
      o = m.GetOutput()
      m.Update()
      return VTKImageToNum(o)


def VTKIslandRemoval(x,isval,repval,area,numret=0):
      if type(x) == np.ndarray: i = NumToVTKImage(x)
      else: i = x
      d = vtk.vtkImageIslandRemoval2D()
      d.SetIslandValue(isval)
      d.SetReplaceValue(repval)
      d.SetAreaThreshold(area)
      d.SquareNeighborhoodOn()
      d.SetInputData(i)
      o = d.GetOutput()
      d.Update()
      if numret: return VTKImageToNum(o)
      else: return o


def VTKGreaterEqual(x1,th):
      if type(x1) == np.ndarray: i1 = NumToVTKImage(x1)
      else: i1 = x1
      m = vtk.vtkImageThreshold()
      m.ThresholdByUpper(th)
      m.SetInputData(i1)
      m.ReplaceInOn()
      m.ReplaceOutOn()
      m.SetInValue(1.0)
      m.SetOutValue(0.0)
      m.SetOutputScalarTypeToFloat()
      o = m.GetOutput()
      m.Update()
      return VTKImageToNum(o)

def VTKLessEqual(x1,th):
      if type(x1) == np.ndarray: i1 = NumToVTKImage(x1)
      else: i1 = x1
      m = vtk.vtkImageThreshold()
      m.ThresholdByLower(th)
      m.SetInputData(i1)
      m.ReplaceInOn()
      m.ReplaceOutOn()
      m.SetInValue(1.0)
      m.SetOutValue(0.0)
      m.SetOutputScalarTypeToFloat()
      o = m.GetOutput()
      m.Update()
      return VTKImageToNum(o)

def VTKBetween(x1,th1,th2):
      if type(x1) == np.ndarray: i1 = NumToVTKImage(x1)
      else: i1 = x1
      m = vtk.vtkImageThreshold()
      m.ThresholdBetween(th1,th2)
      m.SetInputData(i1)
      m.ReplaceInOn()
      m.ReplaceOutOn()
      m.SetInValue(1.0)
      m.SetOutValue(0.0)
      m.SetOutputScalarTypeToFloat()
      o = m.GetOutput()
      m.Update()
      return VTKImageToNum(o)

