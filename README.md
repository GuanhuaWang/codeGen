# codeGen 
A simple c/cuda code generator.

We build a data pipeline which 

(1) read the network topology file (e.g. `all-to-all.txt`) 

(2) using problem solver to generate an optimal data transfer scheme (e.g. `dir.csv`)

(3) generate c/cuda code (e.g. `broadcast.cu`) according to the generated transfer scheme (e.g. `dir.csv`).

## How to Use

**1. compile code generator**

`cd ~/fiddlelink/codeGen/`

`make`

Here it does not matter we failed in building `broadcast.cu` since we have not generated it yet.

If there exists `dir.csv`, please remove it before run shell script below.

**2. run pipeline shell script**

`bash pipe.sh`

