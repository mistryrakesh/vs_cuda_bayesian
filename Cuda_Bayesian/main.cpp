#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "utils.h"
#include "defs.h"
#include "bayesian.cuh"
#include "gpu_timer.h"

using namespace std;

void init(bof::Cell *m) {
    for (int i = 0; i < GRID_ROWS; ++i) {
        for (int j = 0; j < GRID_COLS; ++j) {
            m[i * GRID_ROWS + j].xpos = j;
            m[i * GRID_ROWS + j].ypos = i;
            m[i * GRID_ROWS + j].xVelocityDistribution[NUM_VELOCITY / 2] = 1;
            m[i * GRID_ROWS + j].yVelocityDistribution[NUM_VELOCITY / 2] = 1;
            m[i * GRID_ROWS + j].occupiedProbability = 0;
        }
    }
}

void printOccupancy(bof::Cell *m) {
    for (int i = 0; i < GRID_ROWS; ++i) {
        for (int j = 0; j < GRID_COLS; ++j) {
            m[i * GRID_ROWS + j].toString();
        }
    }
}

int main(int argc, char **argv) {
    size_t numBytes = sizeof(bof::Cell) * GRID_ROWS * GRID_COLS;

    bof::Cell *h_m = new bof::Cell[GRID_ROWS * GRID_COLS];
    bof::Cell *d_prevOccGrid;
    bof::Cell *d_occGrid;

    cout << "Size of grid (bytes): " << numBytes << endl;

    init(h_m);
    // printOccupancy(h_m);

    checkCudaErrors((cudaMalloc(&d_prevOccGrid, numBytes)));
    checkCudaErrors(cudaMemcpy(d_prevOccGrid, h_m, numBytes, cudaMemcpyHostToDevice));
    checkCudaErrors((cudaMalloc(&d_occGrid, numBytes)));
    checkCudaErrors(cudaMemcpy(d_occGrid, h_m, numBytes, cudaMemcpyHostToDevice));

    /* Call kernel */
    GpuTimer timer;
    timer.Start();
    float dt = 0.5f; // compute time delay here
    callKernel(d_occGrid, d_prevOccGrid, dt);
    timer.Stop();
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    cout << "Elapsed time: " << timer.Elapsed() << endl;
    checkCudaErrors(cudaMemcpy(h_m, d_occGrid, numBytes, cudaMemcpyDeviceToHost));

    // printOccupancy(h_m);
    /* cleanup */
    cudaFree(d_prevOccGrid);
    cudaFree(d_occGrid);
    delete h_m;

    return 0;
}
