//This is generated cuda code using scheduling scheme
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <pthread.h>
#include "fiddlelink.h"

int main(){
	long int total_size = 419430400;
	long int batch_size = 139810133;

	int* mem[4];
	for(int x = 0;x<4;x++){
		cudaSetDevice(x);
		cudaMalloc((void**)&mem[x],total_size);
	}

	void* addr[4][3];
	for(int i = 0; i<4;i++){
		for(int j =0; j<3;j++){
			addr[i][j] = (void*)((long long unsigned)mem[i] + j * batch_size);
		}
	}

	pair_stream(1,0,addr[1][1],addr[0][1],batch_size,1);
	pair_stream(2,0,addr[2][2],addr[0][2],batch_size,1);
	pair_stream(3,0,addr[3][0],addr[0][0],batch_size,1);
	pair_stream(2,1,addr[2][1],addr[1][1],batch_size,1);
	pair_stream(3,1,addr[3][2],addr[1][2],batch_size,1);
	pair_stream(1,2,addr[1][2],addr[2][2],batch_size,1);
	pair_stream(3,2,addr[3][1],addr[2][1],batch_size,1);
	pair_stream(1,3,addr[1][0],addr[3][0],batch_size,1);
	pair_stream(2,3,addr[2][0],addr[3][0],batch_size,1);

	for(int i = 0; i<4;i++){
		cudaFree(mem[i]);
	}
}
