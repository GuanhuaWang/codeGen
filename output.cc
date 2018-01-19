//It is a code generator from Python to C/CUDA

/*[[[cog
import cog
import numpy as np
from numpy import genfromtxt

def read_file(dir_name):
#	print dir_name
	data = genfromtxt(dir_name,delimiter=',')
#	print data[:,0]
	return data

#if __name__ == "__main__":
data = read_file('dir.csv')
row = data.shape[0]
column = data.shape[1]
for fn in str(column):
	cog.outl("void %s();" % fn)
]]]*/
void 5();
//[[[end]]]

