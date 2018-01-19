import numpy as np
from numpy import genfromtxt

def read_file(dir_name):
	print dir_name
	data = genfromtxt(dir_name,delimiter=',')
	print data.shape[0]
	return data

if __name__ == "__main__":
	data = read_file('dir.csv')
	print data
