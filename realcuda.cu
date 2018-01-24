#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <pthread.h>
#include "fiddlelink.h"
#define SIZE 10240
#define ROW 12


void read_schedule(const char* file_name, int matrix[12][5])
{
   char buffer[1024] ;
   char *record,*line;
   int i=0,j=0;
   FILE *fstream = fopen(file_name,"r");
   if(fstream == NULL)
   {
      printf("\n file opening failed ");
      return ;
   }
   while((line=fgets(buffer,sizeof(buffer),fstream))!=NULL)
   {
     record = strtok(line,",");
     while(record != NULL)
     {
     //printf("record : %s",record) ; 
     matrix[i][(j++)%5] = atoi(record) ;
     record = strtok(NULL,",");
     }
     ++i ;
   }

	return ;
}

void *peer_access(void *addr){
	int *tx1 = (int*)addr;
	int *rx1 = (int*)(addr + sizeof(int));
	int tx = *tx1;
	int rx = *rx1;
	printf("tx is %d, rx is %d\n",tx,rx);
	cudaSetDevice(tx);
	cudaDeviceEnablePeerAccess(rx,0);
}

int distinct(int arr[],int n){
	int count = 0;
	for(int i = 0;i<n;i++){
		int j;
		for(j=0;j<i;j++)
			if(arr[i]==arr[j])
				break;

		if(i == j){
			printf("%i ",arr[i]);
			count++;
		}
	}	
	printf("\n");
	printf("count is %i\n", count);
	return count;
}

int* one_hot(int scheme[ROW][5]){
	static int one_hot[ROW]={0};
	for(int i =0;i<ROW;i++){
		for(int j =2;j<5;j++){
			if(scheme[i][j]==1){
				one_hot[i] = j-1;	
			}	
		}
	}
/*
	for(int i =0;i<ROW;i++){
		printf("row %i,partition %i\n",i,one_hot[i]);
	}
*/
	return one_hot;
}

int main(){
	const char* name = "dir.csv";
	int matrix[ROW][5];
	int column0[ROW];	

	read_schedule(name,matrix);

	for(int i =0;i<ROW;i++){
		column0[i] = matrix[i][0];
        	for(int j = 0;j<5;j++){
                	printf("%d ",matrix[i][j]);
        	}
        printf("\n");
  	}

	int batch = sizeof(matrix[0])/sizeof(int)-2;
	long int total_size = sizeof(int)*SIZE*SIZE;
	printf("total data size on GPU0 is %f GB\n",total_size/(1024.0*1024.0*1024.0));
        long int batch_size = total_size/batch;

	int count = distinct(column0,ROW);
	int* mem[count];	

	for(int x = 0;x<count;x++){
		cudaSetDevice(x);
		cudaMalloc((void**)&mem[x],total_size);
	}

	int* partition;

	partition = one_hot(matrix);

        for(int i =0;i<ROW;i++){
                printf("row %i,partition %i\n",i,partition[i]);
        }

// Split mem_addr on each GPU to # of batch pieces
	void* addr[count][batch];
	for(int i = 0; i<count;i++){
		for(int j =0; j<batch;j++){
			addr[i][j] =(void*)((long long unsigned)mem[i] + j*batch_size);
		}
	}	
	
// pre transfer setup (enable peer access, allocate GPU mem)

//Open multi-thread for enable peer access in parallel
	int peer[ROW][2];
	for(int i =0;i<ROW;i++){
		peer[i][0]=matrix[i][0];
		peer[i][1]=matrix[i][1];
	}

	pthread_t tid[ROW];
	for(int j = 0; j<ROW; j++){
		pthread_create(&tid[j],NULL,peer_access,&peer[j]);
	}

	for(int m = 0; m<ROW; m++){
		pthread_join(tid[m],NULL);
	}

// Start transfer based on data scheduling scheme (colomn 2 - N)
	for(int i =0;i<ROW;i++){
		if(partition[i]!=0)
			pair_stream(matrix[i][1],matrix[i][0],addr[matrix[i][1]][partition[i]],addr[matrix[i][0]][partition[i]],batch_size,1);	
	}


// post transfer (e.g. Free memory)

	return 0;
}
