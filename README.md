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

## Output Example 

//////////////////////////////////////////////////////////////

//////////Welcome to use FiddleLink/////////////

/////////////////////////////////////////////////////////////

Fri Jan 26 22:24:10 UTC 2018


/////////////////////////////////////////////////////////////

////Problem Solver to Generate Scheme/////

/////////////////////////////////////////////////////////////

num_nodes  4  num_partitions  3

read topology file  /home/ubuntu/fiddlelink/opt-problem/all-to-all.txt

status: optimal

optimal value 1.00000000002

total time  0.333333333338

Source -> Dest : Weights

0 -> 1 : [[0. 1. 0.]]

0 -> 2 : [[0. 0. 1.]]

0 -> 3 : [[ 1.  0. -0.]]

1 -> 0 : [[ 0. -0. -0.]]

1 -> 2 : [[0. 1. 0.]]

1 -> 3 : [[-0.  0.  1.]]

2 -> 0 : [[ 0.  0. -0.]]

2 -> 1 : [[0. 0. 1.]]

2 -> 3 : [[0. 1. 0.]]

3 -> 0 : [[-0.  0.  0.]]

3 -> 1 : [[1. 0. 0.]]

3 -> 2 : [[1. 0. 0.]]

writing generated scheme to dir.csv


/////////////////////////////////////////////////////////////

////////////////Code Generation/////////////////////

/////////////////////////////////////////////////////////////

======================Print scheme=======================

0 1 0 1 0 

0 2 0 0 1 

0 3 1 0 0 

1 0 0 0 0 

1 2 0 1 0 

1 3 0 0 1 

2 0 0 0 0 

2 1 0 0 1 

2 3 0 1 0 

3 0 0 0 0 

3 1 1 0 0 

3 2 1 0 0 



==============Total Data Size in Transfer================

total data size on GPU0 is 0.390625 GB


================Count distinct GPU in use================

0 1 2 3 

count of distinct GPU node is 4


=================OneHot to Number transfer===============

row 0,partition 2

row 1,partition 3

row 2,partition 1

row 3,partition 0

row 4,partition 2

row 5,partition 3

row 6,partition 0

row 7,partition 3

row 8,partition 2

row 9,partition 0

row 10,partition 1

row 11,partition 1


==============Pthread open peer access===================

tx is 0, rx is 1

tx is 0, rx is 2

tx is 0, rx is 3

tx is 1, rx is 0

tx is 1, rx is 2

tx is 1, rx is 3

tx is 2, rx is 0

tx is 2, rx is 1

tx is 2, rx is 3

tx is 3, rx is 0

tx is 3, rx is 1

tx is 3, rx is 2


===================Print data transfer===================

start transfer -- rx: 1, tx: 0, addr_rx: addr[1][1], addr_tx: addr[0][1], batch_size: 0.130208 GB

start transfer -- rx: 2, tx: 0, addr_rx: addr[2][2], addr_tx: addr[0][2], batch_size: 0.130208 GB

start transfer -- rx: 3, tx: 0, addr_rx: addr[3][0], addr_tx: addr[0][0], batch_size: 0.130208 GB

start transfer -- rx: 2, tx: 1, addr_rx: addr[2][1], addr_tx: addr[1][1], batch_size: 0.130208 GB

start transfer -- rx: 3, tx: 1, addr_rx: addr[3][2], addr_tx: addr[1][2], batch_size: 0.130208 GB

start transfer -- rx: 1, tx: 2, addr_rx: addr[1][2], addr_tx: addr[2][2], batch_size: 0.130208 GB

start transfer -- rx: 3, tx: 2, addr_rx: addr[3][1], addr_tx: addr[2][1], batch_size: 0.130208 GB

start transfer -- rx: 1, tx: 3, addr_rx: addr[1][0], addr_tx: addr[3][0], batch_size: 0.130208 GB

start transfer -- rx: 2, tx: 3, addr_rx: addr[2][0], addr_tx: addr[3][0], batch_size: 0.130208 GB


==============Generate code to broadcast.cu==============

