#include <stdio.h>
#include <string.h>
#include <stdlib.h>

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
     //printf("record : %s",record) ;    //here you can put the record into the array as per your requirement.
     matrix[i][(j++)%5] = atoi(record) ;
     record = strtok(NULL,",");
     }
     ++i ;
   }

	return ;
}

int main(){
	const char* name = "dir.csv";
	int matrix[12][5];

	read_schedule(name,matrix);

	for(int i =0;i<12;i++){
        	for(int j = 0;j<5;j++){
                	printf("%d ",matrix[i][j]);
        	}
        printf("\n");
  	}

	
	return 0;
}
