#include <time.h>
#include <assert.h>
#include <stdio.h>
#include <cudnn.h>
#include <cstdlib>
#define CUDA_CALL(x) do {						\
  cudaError_t ____rc = (x);					\
  assert(____rc == cudaSuccess);					\
} while (0)

/* Image channels, height, width. */
#define C	  3	
#define H	  1024
#define W	  1024

/* Tile size. */
#define TW		18
#define TH		18


/* Filter height, width */
#define FH		21
#define FW		21

#define pr_debug(msg...)	printf(msg)
//#define pr_debug(msg...)

#define checkCUDNN(expression)                    \
{                                                  \
	cudnnStatus_t status = (expression);           			  \
	if (status != CUDNN_STATUS_SUCCESS) {        			    \
		printf("cuDNN error on line %d: %s\n" , __LINE__ ,	\
				cudnnGetErrorString(status));  						\
		exit(EXIT_FAILURE);                           \
	}                                               \
}																									\


struct CuDNN_Setup { 
  cudnnHandle_t cudnn;
  cudnnTensorDescriptor_t input_descriptor;
  cudnnTensorDescriptor_t output_descriptor;
  cudnnPoolingDescriptor_t pooling_descriptor;
  double alpha, beta;
};


void PrepareCuDNN(CuDNN_Setup *pS,int c, int h, int w, int fh, int fw,
		  double *x, double *y) {
	/* Create context object*/
	checkCUDNN(cudnnCreate(&pS->cudnn));

	/* Describe input tensor*/
	checkCUDNN(cudnnCreateTensorDescriptor(&pS->input_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(pS->input_descriptor,
				/*format=*/CUDNN_TENSOR_NCHW,
				/*dataType=*/CUDNN_DATA_DOUBLE,
				/*batch_size=*/1,
				/*channels=*/c,
				/*image_height=*/h,
				/*image_width=*/w));

	/* Describe output tensor*/
	checkCUDNN(cudnnCreateTensorDescriptor(&pS->output_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(pS->output_descriptor,
				/*format=*/CUDNN_TENSOR_NCHW,
				/*dataType=*/CUDNN_DATA_DOUBLE,
				/*batch_size=*/1,
				/*channels=*/c,
				/*image_height=*/h,
				/*image_width=*/w));

	/* Describe Pooling tensor */
	checkCUDNN(cudnnCreatePoolingDescriptor(&pS->pooling_descriptor));
	checkCUDNN(cudnnSetPooling2dDescriptor(pS->pooling_descriptor,
				/*mode*/CUDNN_POOLING_MAX,
				/*NAN Propagation*/CUDNN_NOT_PROPAGATE_NAN,
				/*Window Height*/fh,
				/*Window Width*/fw,
				/*Vertical Padding*/(fh-1)/2,
				/*Horizontal Padding*/(fw-1)/2,
				/*Vertical Stride*/1,
				/*Horizontal Stride*/1));
     
	pS->alpha=1.0;
	pS->beta=0.0;
}


void CleanupCuDNN(CuDNN_Setup *pS){

	/* Destroy descriptors */
	cudnnDestroyTensorDescriptor(pS->input_descriptor);
	cudnnDestroyTensorDescriptor(pS->output_descriptor);
	cudnnDestroyPoolingDescriptor(pS->pooling_descriptor);
	cudnnDestroy(pS->cudnn);
}

void DoMaxPooling(CuDNN_Setup *pS,double *x, double *y) {

	checkCUDNN(cudnnPoolingForward(pS->cudnn,
				/*pooling desc*/pS->pooling_descriptor,
				/*alpha*/&pS->alpha,
				/*input*/pS->input_descriptor,
				/*input loc*/ x,
				/*beta*/&pS->beta,
				/*ouput*/pS->output_descriptor,
				/*output loc*/ y));
}



__global__ void maxPoolKernel(int c,int h, int w, int fh, int fw, double*x, double *y){
	
	const int ph = (fh - 1)/2;
        const int pw = (fw - 1)/2;
	int tw = blockDim.x + 2*pw, th = blockDim.y+2*ph;
        int totalElementsPerChannel = tw * th;
        extern __shared__ double tile[];

        int tileRowOff = blockIdx.y * blockDim.y , tileColOff = blockIdx.x * blockDim.x;
	

        int ci = threadIdx.z;
                for(int el=threadIdx.y*blockDim.x + threadIdx.x ; el<totalElementsPerChannel ; el+= blockDim.x*blockDim.y){
                        int pRow = el / tw, pCol = el % tw;
                        int globRow = pRow + tileRowOff - ph, globCol = pCol + tileColOff - pw;
                        tile[ci*totalElementsPerChannel + el] = (  globRow<0 || globRow>=h || globCol<0 || globCol>=w ) ? 0 : x[ ci*w*h + globRow * w + globCol ];
                }
        

        __syncthreads();

	

        int ki = threadIdx.z;
                int col = threadIdx.x, row = threadIdx.y;
                double res = 0;               
                for(int rx = 0 ; rx < fh ; rx ++){
                       for(int cx = 0 ; cx < fw ; cx++ ){
                                double imageVal  = tile[ki*totalElementsPerChannel + (row+rx)*tw + (col+cx)];
                                res = res > imageVal ? res : imageVal;
                       }
                }
                

                int frow = tileRowOff + row;
                int fcol = tileColOff + col;
                if(frow<h && fcol<w)
                        y[ki*h*w + frow*w + fcol ] = res;
        
}




#define DIV_RUP(x, y)	(((x)+(y)-1)/(y))

////////////////////////////////////////////////////////////////////////////////
// Create Image in CPU memory
////////////////////////////////////////////////////////////////////////////////
void fill_image(int c, int h, int w, int fill, double *cpup, double offset)
{
  int siz = c*h*w*sizeof(double);
  
  if (fill) {
    memset(cpup, 0, siz);
    for (int ci = 0; ci < c; ++ci) {
      for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
          cpup[ci*w*h + j*w + i]= offset + ci * (i+j);// + rand()%10 ;
        }
      }
    }
  }
}




/////////////////////////////////////////////////////////////////////////////////
// Print image in cpu memory if small, compute checksum if larger
/////////////////////////////////////////////////////////////////////////////////
void print_image(const char *name, int c, int h, int w, double *xcpu)
{

	long nelem=c*h*w;
	if(  nelem < 1024) {
		printf("%s: ", name);
		for (int ci = 0; ci < c; ++ci) {
			printf("[\n");
			for (int j = 0; j < h; ++j) {
				printf(" ");
				for (int i = 0; i < w; ++i) {
					double elem = xcpu[ci*h*w + j*w + i];
					printf("%.2f ", elem);
				}				
				printf("\n");
			}
			printf("]\n");
		}
	}

	double checksum=0;
	for (int e = 0; e < nelem; e++)
		checksum += xcpu[e];
	printf("%s checksum: %.2f\n",name, checksum);
}




static double TimeSpecToSeconds(struct timespec* ts){
    return (double)ts->tv_sec + (double)ts->tv_nsec / 1000000000.0;
}

#define REPS 3
int main(int ac, char *av[]){
srand (time(NULL)); 
/*Initialize images*/
  double *x, *y;
  int imsz = C*H*W*sizeof(double);
  int outsz = C*H*W*sizeof(double);
  
  CUDA_CALL(cudaMallocHost(&x,imsz));
  CUDA_CALL(cudaMallocHost(&y,outsz));
  
  fill_image(C, H, W, 1,x,0.0);

 
  //print_image("I =", C, H, W, x);

  struct timespec start;
  struct timespec end;
  double timeConv = 0, singletime,copytime;
  double timecuDNN = 0;

  /* Run CUDA kernel*/
  dim3 img_blk(TW, TH, C);
  dim3 img_grid(DIV_RUP(W, TW), DIV_RUP(H, TH));
 // dim3 img_blk_old(TW,TH);
  int i;

  double *imdev;
  double *outdev;
  CUDA_CALL(cudaMalloc(&imdev, imsz));
  CUDA_CALL(cudaMalloc(&outdev, outsz));


  {
  int shmem_size = C*(TW+FW/2*2)*(TH+FH/2*2)*sizeof(double);
  for (i = 0; i < REPS; ++i) {
    //fill_image(C, H, W, 1,x,0.0);  
    print_image("I =", C, H, W, x);
    if(clock_gettime(CLOCK_MONOTONIC, &start)){ printf("CLOCK ERROR"); }
 
    CUDA_CALL(cudaMemcpy(imdev, x, imsz, cudaMemcpyHostToDevice));                            // copy image to device
    cudaDeviceSynchronize();
    if(clock_gettime(CLOCK_MONOTONIC, &end)) { printf("CLOCK ERROR"); }
    copytime = TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start);
    printf("Copy host->dev kernel %lf sec\n", copytime);

    if(clock_gettime(CLOCK_MONOTONIC, &start)){ printf("CLOCK ERROR"); }
    if(FW%2==0 || FH%2==0){ printf("------\nError: Kernel Works only for filter of odd demensions\n------\n"); exit(1);} else
    maxPoolKernel<<<img_grid, img_blk, shmem_size>>>(C, H, W, FH, FW, imdev, outdev);
   // maxPoolKernelOld<<<img_grid,img_blk_old, shmem_size>>>(C, H, W, FH, FW, imdev, outdev);
    cudaDeviceSynchronize();
    if(clock_gettime(CLOCK_MONOTONIC, &end)) { printf("CLOCK ERROR"); }
    singletime = TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start);
    printf("time kernel %lf sec\n", singletime);
    timeConv += singletime;

    if(clock_gettime(CLOCK_MONOTONIC, &start)){ printf("CLOCK ERROR"); }
    CUDA_CALL(cudaMemcpy(y, outdev, outsz, cudaMemcpyDeviceToHost));                          // copy result back
    cudaDeviceSynchronize();
    if(clock_gettime(CLOCK_MONOTONIC, &end)) { printf("CLOCK ERROR"); }
    copytime = TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start);
    printf("Copy dev->host kernel %lf sec\n", copytime);

    print_image("CUDA O =", C, H, W, y);                                                      // check result
    printf("\n");
  }
  timeConv/=REPS;
  }

  printf("-------------------------------\n\n");
  CuDNN_Setup S;
  PrepareCuDNN(&S,C, H, W, FH, FW, imdev, outdev);

  /* Run cuDNN kernel */
  for (i = 0; i < REPS; ++i) {
    //fill_image(C, H, W, 1,x,0.0);
    print_image("I =", C, H, W, x);
    if(clock_gettime(CLOCK_MONOTONIC, &start)){ printf("CLOCK ERROR"); }
    CUDA_CALL(cudaMemcpy(imdev, x, imsz, cudaMemcpyHostToDevice));                            // copy image to device
    cudaDeviceSynchronize();
    if(clock_gettime(CLOCK_MONOTONIC, &end)) { printf("CLOCK ERROR"); }
    copytime = TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start);
    printf("Copy host->dev cudnn %lf sec\n", copytime);

    if(clock_gettime(CLOCK_MONOTONIC, &start)){ printf("CLOCK ERROR"); }
    DoMaxPooling(&S,imdev, outdev);
    cudaDeviceSynchronize();
    if(clock_gettime(CLOCK_MONOTONIC, &end)) { printf("CLOCK ERROR"); }
    singletime = TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start);
    timecuDNN+=singletime;
    printf("time cudnn %lf sec\n", singletime);

    if(clock_gettime(CLOCK_MONOTONIC, &start)){ printf("CLOCK ERROR"); }
    CUDA_CALL(cudaMemcpy(y, outdev, outsz, cudaMemcpyDeviceToHost));                          // copy result back
    cudaDeviceSynchronize();
    if(clock_gettime(CLOCK_MONOTONIC, &end)) { printf("CLOCK ERROR"); }
    copytime = TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start);
    printf("Copy dev->host kernel %lf sec\n", copytime);

    print_image("CUDNN O =", C, H, W, y);                                                      // check result
    printf("\n");
  }
  timecuDNN/=REPS;

  printf("\n\n <Time>: Max Pool Kernel: %lf sec, Max Pool cuDNN: %lf sec\n", timeConv, timecuDNN);

  CleanupCuDNN(&S);


  //CUDA_CALL(cudaFree(filt));
  CUDA_CALL(cudaFree(outdev));
  CUDA_CALL(cudaFree(imdev));

  return 0;
}
// I = checksum: 3218079744.00
// Copy host->dev kernel 0.002154 sec
// time kernel 0.017305 sec
// Copy dev->host kernel 0.041188 sec
// CUDA O = checksum: 122756344698240.00

