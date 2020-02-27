def rdarg(argv,key,conv=None,default=None,single=0):
   val = default
   if key in argv:
    if not single:
      val=argv[argv.index(key)+1]
      del argv[argv.index(key):argv.index(key)+2]
      if conv: val=map(conv,[val])[0]
    else:
      del argv[argv.index(key):argv.index(key)+1]
      val = 1
      if default == 1: val=0
   else:
      if conv and val: val=conv(val)
   return argv,val
