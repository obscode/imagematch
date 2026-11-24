'''Class for parsing configuration files. This builds on ConfigParser
but tries to do some auto-typing. You use it by initializing a config
object. The only argument is the config file:

>>> cf = config('myconfig.cfg')

cf will then have member variables, one for each config section. Each of these
has values assocated with conficuration variables.  So if the myconfig.cfg
file looked like this:

[Section1]
x = 5
y = blah
z = 1.0,5.0,10.0

You would access these items as cf.Section1.x and cf.Section1.y and they should be properly typed (x is an int and y is a string). A special case is the value
of cf.Section1.z:  because of the commas, we interpret it as a list, converting
all values to numeric (int or float) if possible, or leave as strings.
'''

import configparser,os

def guess_value(value):
   '''Try to guess the most likely type and return it typed, rather than
   as a string (which is the behaviour of ConfigParser.'''

   if value == '':
      return None
   
   # check for True/False
   if value.lower() == 'true':
      return True
   elif value.lower() == 'false':
      return False

   # See if we're dealing with a list:
   if value[0] in ['[','('] and value[-1] in [']',')']:
      value = value[1:-1]
   if value.find(',') > 0:
      value = value.split(',')
   else:
      value = [value]

   ret = []
   # Now we check numerics.  Ints are most restrictive, followed by floats
   try:
      ret = map(int,value)
      if len(ret) == 1: return ret[0]
      return ret
   except:
      try:
         ret = map(float,value)
         if len(ret) == 1:  return ret[0]
         return ret
      except:
         if len(value) == 1:  return value[0]
         return value

class section:
   '''A class used by the config class to hold the options of a section.'''
   
   def __init__(self, parent, section):
      self.cf = parent.cf
      self.section = section
      self.options = self.cf.options(section)

   def __getattr__(self, key):

      if key not in self.__dict__['options']:
         #raise AttributeError, "Section %s has no option %s" % \
         #      (self.section,key)
         return None
      else:
         return guess_value(self.__dict__['cf'].get(self.section,key))

   def __dir__(self):
      ''' implement this to get tab autocomplete in ipython.'''
      return self.__dict__['options']


class config:
   '''A wrapper around the ConfigParser class.  This class gives a more
   pythonic access to options as member variables and also tries to
   guess the type of the variable.'''

   def __init__(self, configfile):
      self.cf = configparser.ConfigParser()
      self.cf.optionxform=str
      if type(configfile) is type(""):
         if not os.path.isfile(configfile):
             raise IOError("No such config file %s" % configfile)
         self.cf.read(configfile)
      else:
         self.cf.readfp(configfile)
      self.sections = {}
      for sec in self.cf.sections():
         self.sections[sec] = section(self, sec)

   def __getattr__(self, key):
      if key in self.__dict__['sections']:
         return self.__dict__['sections'][key]
      else:
         raise AttributeError("configuration file has no section %s" % key)

   def __dir__(self):
      '''implement this to get auto-complete in ipython.'''
      return self.__dict__['sections']

def fixConfig(cfgfile):
   '''Update an older config file to the new format.  This is a one-time operation.'''
   cfg = config(cfgfile)
   if hasattr(cfg, "rect") and hasattr(cfg.rect, 'use_wcs') and \
      hasattr(cfg.sex, 'use_wcs'):
      return cfg
   if hasattr(cfg.match, 'use_wcs'):
      print("Warning: This is an older config format, saving to new format")
      print("  Old config file will be renamed to %s.old" % cfgfile)

      os.rename(cfgfile, cfgfile+'.old')

      cfg.cf['rect'] = {}
      cfg.cf.set('rect','use_wcs',cfg.cf.get('match','use_wcs'))
      cfg.cf.remove_option('match','use_wcs')
      cfg.cf.set('rect','registered',cfg.cf.get('match','registered'))
      cfg.cf.remove_option('match','registered')
      cfg.cf.set('rect','nord',cfg.cf.get('match','nord'))
      cfg.cf.remove_option('match','nord')
      cfg.cf.set('rect','Niter',"3")
      cfg.cf.set('rect','perr',cfg.cf.get('match','perr'))
      cfg.cf.remove_option('match','perr')
      cfg.cf.set('rect','crowd',cfg.cf.get('match','crowd'))
      cfg.cf.remove_option('match','crowd')
      cfg.cf.set('rect','db',cfg.cf.get('match','db'))
      cfg.cf.remove_option('match','db')
      cfg.cf.set('rect','starlist',cfg.cf.get('match','starlist'))
      cfg.cf.remove_option('match','starlist')
      cfg.cf.set('rect','starmin',cfg.cf.get('match','starmin'))
      cfg.cf.remove_option('match','starmin')
      cfg.cf.set('sex','use_wcs',cfg.cf.get('rect','use_wcs'))
      with open(cfgfile,'w') as f:
         cfg.cf.write(f)
      cfg = config(cfgfile)
      return cfg
   else:
      print("Config file is not compatible with this version of ImageMatch." \
      "  Please update your config file.")
      return None


