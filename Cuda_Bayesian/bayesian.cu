#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "defs.h"
#include "bayesian.cuh"

bof::Cell::Cell() {
    for (int i = 0; i < NUM_VELOCITY; ++i) {
        this->xVelocityDistribution[i] = 0;
        this->yVelocityDistribution[i] = 0;
    }

    this->occupiedProbability = 0;
    this->xpos = 0;
    this->ypos = 0;
}

__host__ __device__ bof::Cell::Cell(const float xVelocityDistribution[7], const float yVelocityDistribution[7], const float occupiedProbability, const int xpos, const int ypos) {
    for (int i = 0; i < NUM_VELOCITY; ++i) {
        this->xVelocityDistribution[i] = xVelocityDistribution[i];
        this->yVelocityDistribution[i] = yVelocityDistribution[i];
    }

    this->occupiedProbability = occupiedProbability;
    this->xpos = xpos;
    this->ypos = ypos;
}

__host__ __device__ void bof::Cell::getAntecedents(bof::Cell *antecedents, int *antSize, const bof::Cell *prevOccGrid) {
    return;
}

__host__ __device__ void bof::Cell::getPrediction(float *alphaO, float *alphaE, const int xVelocity, const int yVelocity, bof::Cell *antecedents, const int antSize, const bof::Cell *prevOccGrid) {
    return;
}

__host__ __device__ void bof::Cell::getEstimation(float *alphaOccMatrix, float *alphaEmpMatrix, const float lvkSum) {
    return;
}

__host__ __device__ float bof::Cell::getNewOccupiedProbability(const float *alphaOccMatrix) {
    return 0.0f;
}

__host__ __device__ void bof::Cell::updateVelocityProbabilities(const float *alphaOccMatrix, const float *alphaEmpMatrix, const int *xVelocityKeys, const int *yVelocityKeys) {
    return;
}

__host__ __device__ int bof::Cell::isReachable(const int xVelocity, const int yVelocity, const bof::Cell *cell) {
    return 0;
}

__host__ __device__ void bof::Cell::updateDistributions(bof::Cell *prevOccGrid) {
    return;
}

void bof::Cell::toString() {
    std::cout << "occupied: " << occupiedProbability << std::endl;
}

__global__ void computeDistributions(bof::Cell *cell) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= GRID_COLS || y >= GRID_ROWS) {
        return;
    }

    unsigned int index = y * GRID_ROWS + x;
    cell[index].occupiedProbability = cell[index].occupiedProbability + 1;
}

void callKernel(bof::Cell *cell) {
    /* Compute gridsize and blocksize */
    const unsigned int width = 32;
    const dim3 blockSize(width, width, 1);

    unsigned int gridCols = GRID_COLS / width + (GRID_COLS % width != 0);
    unsigned int gridRows = GRID_ROWS / width + (GRID_ROWS % width != 0);
    const dim3 gridSize(gridRows, gridCols, 1);

    computeDistributions<<<gridSize, blockSize >>>(cell);
}