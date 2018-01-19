# codeGen
a simple python to c/cuda code generator

## install Cog

https://pypi.python.org/pypi/cogapp

unpack and install

`python setup.py install`

## run test

`python cog.py -o output.txt simpletest.py`

## simple example for code generation from python to c

Basically, one can write the python code within `[[[cog...]]]` and end the code generation process with `[[[end]]]`

`/*[[[cog`

`import cog`

`fnames = ['DoSomething', 'DoAnotherThing', 'DoLastThing']`

`for fn in fnames:`

`    cog.outl("void %s();" % fn)`

`]]]*/`

`//[[[end]]]`

It will generate the following c code:

`void DoSomething();`

`void DoAnotherThing();`

`void DoLastThing();`
