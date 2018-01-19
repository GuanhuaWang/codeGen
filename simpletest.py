'''
/*[[[cog
import cog
fnames = ['DoSomething', 'DoAnotherThing', 'DoLastThing']
for fn in fnames:
    cog.outl("void %s();" % fn)
]]]*/
//[[[end]]]
'''

import numpy as np
import pandas as pd

def read_file(dir_name):
    #input_arr = open(name,"r")
    #print input_arr.read()
    print dir_name
    data = pd.read_csv(dir_name,sep=',')
    print data
    
def main():
    read_file('dir.csv')


if __name__ == "__main__":
  main()