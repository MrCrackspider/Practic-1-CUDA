#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;

__global__ void find(int* src, int srcHeight, int srcWidth, int* trg, int trgHeight, int trgWidth, int startX, int startY)
{
	int cnt = 0;
	int m = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;

	if (src[(k + startY) * srcWidth + m + startX] == trg[k * trgWidth + m]) cnt++;
	if (cnt = trgHeight * trgWidth) return;
	else cnt = 0;
}


int main()
{
	int DeviceCount;
	cudaGetDeviceCount(&DeviceCount);
	cout << "Device count: " << DeviceCount << endl;
	cudaDeviceProp DeviceProp;
	cudaGetDeviceProperties(&DeviceProp, 0);
	cout << "Device name: " << DeviceProp.name << endl << "Total global memory (bytes): " << DeviceProp.totalGlobalMem << endl;
	cout << "Shared memory per block (bytes): " << DeviceProp.sharedMemPerBlock << endl << "registers per block: " << DeviceProp.regsPerBlock << endl;
	cout << "Warp size: " << DeviceProp.warpSize << endl << "Max threads per block: " << DeviceProp.maxThreadsPerBlock << endl;
	cout << "Total constant memory: " << DeviceProp.totalConstMem << endl;
	cudaSetDevice(0);

	Mat srcImage = imread("C:/Users/kotle/Desktop/source.PNG", IMREAD_GRAYSCALE);;
	if (srcImage.empty())
	{
		cout << "Could not open or find the target image" << std::endl;
		return -1;
	}

	int srcHeight = srcImage.rows;
	int srcWidth = srcImage.cols;
	int srcSize = srcHeight * srcWidth * sizeof(int);
	cout << "Source height: " << srcHeight << " Source width: " << srcWidth << endl;

	int* src = new int[srcHeight * srcWidth];

	for (int i = 0; i < srcHeight; i++)
	{
		for (int j = 0; j < srcWidth; j++)
		{
			src[i * srcWidth + j] = (int)srcImage.at<uchar>(i, j);
		}
	}

	Mat trgImage = imread("C:/Users/kotle/Desktop/target.PNG", IMREAD_GRAYSCALE);;
	if (trgImage.empty())
	{
		cout << "Could not open or find the target image" << std::endl;
		return -1;
	}

	int trgHeight = trgImage.rows;
	int trgWidth = trgImage.cols;
	int trgSize = trgHeight * trgWidth * sizeof(int);
	cout << "Target height: " << trgHeight << " Target width: " << trgWidth << endl;

	int* trg = new int[trgHeight * trgWidth];

	for (int i = 0; i < trgHeight; i++)
	{
		for (int j = 0; j < trgWidth; j++)
		{
			trg[i * trgWidth + j] = (int)trgImage.at<uchar>(i, j);
		}
	}

	bool founded = false;

	// выделение памяти для исходного массива и копирование
	int* dev_src = 0;
	int* dev_trg = 0;
	int* dev_founded = 0;

	dim3 threads(32, 32, 1);
	dim3 blocks(srcWidth * srcHeight / 1024 + 1, srcWidth * srcHeight / 1024 + 1, 1);

	cudaEvent_t start, stop;
	float gpuTime = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaMalloc((void**)& dev_src, srcSize);
	cudaMalloc((void**)& dev_trg, trgSize);
	cudaMalloc((void**)& dev_founded, sizeof(bool));

	cudaMemcpy(dev_src, src, srcSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_trg, trg, trgSize, cudaMemcpyHostToDevice);
	// минимум 648 блоков

	for (size_t i = 0; i < srcHeight - (trgHeight - 1); i++)
	{
		for (size_t j = 0; j < srcWidth - (trgWidth - 1); j++)
		{
			if (src[i * srcWidth + j] == trg[0] && src[i * srcWidth + j + 1] == trg[1])
			{
				find <<<blocks, threads>>> (dev_src, srcHeight, srcWidth, dev_trg, trgHeight, trgWidth, dev_founded, j, i);
			}
		}
	}
	
	cudaDeviceSynchronize();

	cudaMemcpy(&founded, dev_founded, sizeof(bool), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);
	/*
	if (cnt == trgHeight * trgWidth)
	{
		for (size_t i = 0; i < trgWidth; i++)
		{
			src[(startY * srcWidth + startX) + i] = 255;
			src[((startY + trgHeight) * srcWidth + startX) + i] = 255;
		}
		for (size_t i = 0; i < trgHeight; i++)
		{
			src[(startY + i) * srcWidth + startX] = 255;
			src[(startY + i) * srcWidth + startX + trgWidth] = 255;
		}
		return;
	}
	*/
	cout << "GPU Time " << gpuTime << endl;

	for (size_t i = 0; i < srcHeight; i++)
	{
		for (size_t j = 0; j < srcWidth; j++)
		{
			srcImage.at<uchar>(i, j) = src[i * srcWidth + j];
		}
	}

	std::cout << "Position X: " << position[0] << " Position Y: " << position[1];

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(dev_src);
	cudaFree(dev_trg);
	cudaFree(dev_founded);

	namedWindow("Result image", WINDOW_AUTOSIZE);
	imshow("Result image", srcImage);
	waitKey(0);

	cudaDeviceReset();

	return 0;
}