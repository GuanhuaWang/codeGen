# codeGen
a simple python to c/cuda code generator

## install Cog

https://pypi.python.org/pypi/cogapp

unpack and install

`python setup.py install`

## run test

`python cog.py test.txt`

## simple example for code generation from python to c

/*[[[cog
import cog
fnames = ['DoSomething', 'DoAnotherThing', 'DoLastThing']
for fn in fnames:
    cog.outl("void %s();" % fn)
]]]*/
