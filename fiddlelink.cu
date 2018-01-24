#include<assert.h>
#include<stdlib.h>
#include<stdio.h>
#include<time.h>
#include<cuda.h>
#include "fiddlelink.h"

void *peer_access(void *addr){
        int *tx1 = (int*)addr;
        int *rx1 = (int*)(addr + sizeof(int));
        int tx = *tx1;
        int rx = *rx1;
        printf("tx is %d, rx is %d\n",tx,rx);
        cudaSetDevice(tx);
        cudaDeviceEnablePeerAccess(rx,0);
}



void pair_stream(int rx, int tx, void* dst, void* src, double size, int type){

	if(type == 3){
		if(dst == src)
			{
                                printf("stream src and dst addtress should be different.\n");
                                exit(0);
                }

                else{
			
			int split = sizeof(char)*8;
			long int chunk = size/split;
			
			cudaStream_t stream[2];
			cudaStreamCreate(&stream[0]);
			cudaStreamCreate(&stream[1]);
			
                        cudaDeviceDisablePeerAccess(tx);
                        cudaDeviceDisablePeerAccess(rx);

			cudaMemcpyAsync(dst,src,chunk,cudaMemcpyDeviceToDevice,stream[0]);

			cudaDeviceEnablePeerAccess(tx,0);
			cudaDeviceEnablePeerAccess(rx,0);

			cudaMemcpyAsync(dst,src,chunk,cudaMemcpyDeviceToDevice,stream[1]);

			long int address_src = (long long unsigned)src;
			long int address_dst = (long long unsigned)dst;
			
			for(int i = 0; i<split;i++)
			{
				long int address_src1 = address_src+i*chunk;
				long int address_dst1 = address_dst+i*chunk;
				void* src1 = (void*)address_src1;
				void* dst1 = (void*)address_dst1;

                                long int address_src2 = address_src + (i+1)*chunk;
                                long int address_dst2 = address_dst + (i+1)*chunk;
                                void* src2 = (void*)address_src2;
                                void* dst2 = (void*)address_dst2;

                                long int address_src3 = address_src + (i+2)*chunk;
                                long int address_dst3 = address_dst + (i+2)*chunk;
                                void* src3 = (void*)address_src3;
                                void* dst3 = (void*)address_dst3;

				if(cudaMemcpyAsync(dst1,src1,chunk,cudaMemcpyDeviceToDevice,stream[1])==cudaSuccess && cudaMemcpyAsync(dst1,src1,chunk,cudaMemcpyDeviceToDevice,stream[0])==cudaSuccess )
					{

						cudaDeviceEnablePeerAccess(tx,0);
						cudaMemcpyAsync(dst2,src2,chunk,cudaMemcpyDeviceToDevice,stream[1]);					
						cudaDeviceDisablePeerAccess(tx);
						cudaMemcpyAsync(dst3,src3,chunk,cudaMemcpyDeviceToDevice,stream[0]);
						i = i + 1;
					}
				else if (cudaMemcpyAsync(dst1,src1,chunk,cudaMemcpyDeviceToDevice,stream[0])==cudaSuccess && cudaMemcpyAsync(dst1,src1,chunk,cudaMemcpyDeviceToDevice,stream[1])!=cudaSuccess)
					{	
						cudaDeviceDisablePeerAccess(tx);
						cudaMemcpyAsync(dst2,src2,chunk,cudaMemcpyDeviceToDevice,stream[0]);
					}

				else if (cudaMemcpyAsync(dst1,src1,chunk,cudaMemcpyDeviceToDevice,stream[0])!=cudaSuccess && cudaMemcpyAsync(dst1,src1,chunk,cudaMemcpyDeviceToDevice,stream[1])==cudaSuccess)
					{
						cudaDeviceEnablePeerAccess(tx,0);
                                                cudaMemcpyAsync(dst2,src2,chunk,cudaMemcpyDeviceToDevice,stream[1]);
					}
			}

			cudaStreamDestroy(stream[0]);
			cudaStreamDestroy(stream[1]);
		}


	}


	else if(type == 4){

			cudaStream_t stream[2];
                        cudaStreamCreate (&stream[0]);
                        cudaStreamCreate (&stream[1]);

			
			
                        long int stream0_size = size*8/16;
                        long int stream1_size = size*6/16;
			long int stream2_size = size*2/16;

                        long int address_src = (long long unsigned)src;
                        long int address_src1 = address_src+stream0_size;

                        long int address_dst = (long long unsigned)dst;
                        long int address_dst1 = address_dst+stream0_size;

                        void* src1 = (void*)address_src1;
                        void* dst1 = (void*)address_dst1;

			long int address_src2 = address_src1+stream1_size;
			long int address_dst2 = address_dst1+stream1_size;

			void* src2 = (void*)address_src2;
			void* dst2 = (void*)address_dst2;

			cudaMemcpyAsync(dst,src,stream0_size,cudaMemcpyDeviceToDevice,stream[0]);
			cudaDeviceEnablePeerAccess(tx,0);

                        cudaMemcpyAsync(dst1,src1,stream1_size,cudaMemcpyDeviceToDevice,stream[1]);
		
                        if (cudaMemcpyAsync(dst1,src1,stream1_size,cudaMemcpyDeviceToDevice,stream[1])==cudaSuccess)

				{
					cudaDeviceEnablePeerAccess(tx,0);
					cudaMemcpyAsync(dst2,src2,stream2_size,cudaMemcpyDeviceToDevice,stream[1]);

				}

/*
			else if(cudaMemcpyAsync(dst1,src1,stream1_size,cudaMemcpyDeviceToDevice,stream[1])==cudaSuccess && cudaMemcpyAsync(dst,src,stream0_size,cudaMemcpyDeviceToDevice,stream[0])==cudaSuccess)
				{	
					long int address_src3 = address_src2+stream2_size*2/3;
                        		long int address_dst3 = address_dst2+stream2_size*2/3;

                        		void* src3 = (void*)address_src3;
                        		void* dst3 = (void*)address_dst3;
					cudaMemcpyAsync(dst2,src2,stream2_size,cudaMemcpyDeviceToDevice,stream[1]);
					cudaDeviceDisablePeerAccess(tx);
					cudaMemcpyAsync(dst3,src3,stream2_size/3,cudaMemcpyDeviceToDevice,stream[0]);
				}*/		
			else if (cudaMemcpyAsync(dst,src,stream0_size,cudaMemcpyDeviceToDevice,stream[0])==cudaSuccess && cudaMemcpyAsync(dst1,src1,stream1_size,cudaMemcpyDeviceToDevice,stream[1])!=cudaSuccess)
				{

					cudaDeviceDisablePeerAccess(tx);
					cudaMemcpyAsync(dst2,src2,stream2_size,cudaMemcpyDeviceToDevice,stream[0]);
				
				}


		}


        else if(type == 2){
		if(dst == src)
			{
				printf("stream src and dst addtress should be different.\n");
				exit(0);
			}
		else{
			long int stream0_size = size*11/16;
			long int stream1_size = size*5/16;

			long int address_src = (long long unsigned)src;
			long int address_src1 = address_src+stream0_size;

		        long int address_dst = (long long unsigned)dst;
			long int address_dst1 = address_dst+stream0_size;

			void* src1 = (void*)address_src1;
			void* dst1 = (void*)address_dst1;

			cudaSetDevice(tx);
                        cudaStream_t stream[2];
                        cudaStreamCreate (&stream[0]);
                        cudaStreamCreate (&stream[1]);
//			cudaDeviceDisablePeerAccess(rx);
  //                      cudaMemcpyAsync(dst1,src1,stream1_size,cudaMemcpyDeviceToDevice,stream[1]);			
				
			cudaDeviceEnablePeerAccess(rx,0);
			cudaMemcpyAsync(dst,src,stream0_size,cudaMemcpyDeviceToDevice,stream[0]);
                        cudaDeviceDisablePeerAccess(rx);
                        cudaMemcpyAsync(dst1,src1,stream1_size,cudaMemcpyDeviceToDevice,stream[1]);
			cudaStreamDestroy(stream[0]);
			cudaStreamDestroy(stream[1]);	

		}


	}

        else if(type == 1){

		if(dst == src)
			{
				printf("stream src and dst address should be different.\n");
				exit(0);
			}
		else{
                	cudaSetDevice(tx);
			cudaDeviceEnablePeerAccess(rx,0);
                	cudaStream_t stream[1];
                	cudaStreamCreate (&stream[0]);

                	cudaMemcpyAsync(dst,src,size,cudaMemcpyDeviceToDevice,stream[0]);
			cudaStreamDestroy(stream[0]);
		}

        }

        else if(type == 0){
	
		if(dst == src)
			{
				printf("stream src and dst address should be different.\n");	
				exit(0);
			}
		else{

			cudaSetDevice(tx);
		//	cudaDeviceDisablePeerAccess(rx);
                	cudaStream_t stream[1];
                	cudaStreamCreate(&stream[0]);
                	cudaMemcpyAsync(dst,src,size,cudaMemcpyDeviceToDevice,stream[0]);
			cudaStreamDestroy(stream[0]);
		}

        }

}

