from numpy.core.numerictypes import *
import numpy

INTKEYS = ["NUMBER","VECTOR_ASSOC"]
STRKEYS = ["ID"]

def readsex(filename,verb=0):
   fu = open(filename,"r")
   icol = []
   keys = []
   fmts = []
   data = {}
   n=0
   for l in fu.readlines():
       if l[0] == "#":
         w = l.split()
         i = int(w[1])-1
         if len(icol) and i!= icol[-1]+1:
            key = keys[-1]
            nmag = i-icol[-1]-1
            keys[-1] = key+"(1)"
            data[keys[-1]] = data.pop(key)
            for j in range(nmag):
                icol.append(icol[-1]+1)
                keys.append("%s(%d)" %(key,j+2))
                data[keys[-1]] = []
                n+=1
         icol.append(int(w[1])-1)
         keys.append(w[2])
         if w[2] in STRKEYS: fmts.append("%s")
         elif w[2] in INTKEYS: fmts.append("%6d")
         elif w[2] == "FLAGS": fmts.append("%4d")
         elif w[2] == "CLASS_STAR": fmts.append("%5.2f")
         elif w[2] == "X_IMAGE": fmts.append("%9.3f")
         elif w[2] == "Y_IMAGE": fmts.append("%9.3f")
         else: fmts.append("%9.4f")
         data[w[2]] = []
         n+=1
       else:
         w = l.split()
         [data[keys[j]].append(w[j]) for j in range(n)]
   fu.close()
   if verb:
     print("Found Keys=")
     print(keys)
     print("Number of Objects = %d" % ( len(data[keys[0]])))
   for k in keys: data[k] = [(k in STRKEYS and d) or [float,int][k in INTKEYS](d) for d in data[k]]
   return (data,keys,icol,fmts)

def cvt(r,k):
    c = {0:float, 1: str, 2: int}[(k in INTKEYS and 2) or (k in STRKEYS and 1) or 0]
    return c(r)

def readsexcat(f):
    ns = []
    ks = []
    data = {}
    l = 0
    F = open(f,"r")
    for line in F:
        if line[0] == "#":
           n,k = line.split()[1:3]
           n = int(n)
           if ns: dn = n-ns[-1]-1
           else: dn = 0
           if dn:
              k2 = ks[-1]
              for i in range(dn):
                  ns.append(ns[-1]+1)
                  ks.append(k2+"(%d)"%(i+1))
           ns.append(n)
           ks.append(k)
        else:
           if l == 0:
              for k in ks: data[k] = []
           [data[k].append(cvt(r,k)) for n,k,r in zip(ns,ks,line.split())]
           l += 1
    F.close()
    for n,k in zip(ns,ks):
        if k in STRKEYS: pass
        else: data[k] = asarray(data[k])
    return data


