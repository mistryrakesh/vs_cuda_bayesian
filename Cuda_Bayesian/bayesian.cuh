#ifndef __bayesian_cuh__
#define __bayesian_cuh__

#include <cuda_runtime.h>

#include "defs.h"

namespace bof {

    class Cell {
    public:
        int xpos;
        int ypos;
        float xVelocityDistribution[NUM_VELOCITY];
        float yVelocityDistribution[NUM_VELOCITY];
        float occupiedProbability;

        /**
         * Default Constructor. Sets everything to 0;
         * To be called by host only.
         */
        Cell();

        /**
         * Constructor
         *
         * @param xVelocityDistribution     velocity distribution along x-axis
         * @param yVelocityDistribution     velocity distribution along y-axis
         * @param occupiedProbability       probability of occupancy
         * @param xpos                      x-position of cell
         * @param ypos                      y-position of cell
         */
        __host__ __device__ Cell(const float xVelocityDistribution[NUM_VELOCITY], const float yVelocityDistribution[NUM_VELOCITY], const float occupiedProbability, const int xpos, const int ypos);

        /**
         * Returns the set of antecedents for the current cell
         *
         * @param antecedents   pointer to a list of antecedents computed by the function
         * @param antSize       size of 'antecedents' array
         * @param prevOccGrid   linearized 2D array of previous occupancy grid
         */
        __host__ __device__ void getAntecedents(Cell *antecedents, int *antSize, const Cell *prevOccGrid);
        
        /**
         * Computes 'alphaO' and 'alphaE' for a given velocity
         *
         * @param alphaO        pointer to computed alphaO value
         * @param alphaE        pointer to computed alphaE value
         * @param xVelocity     velocity in x-axis
         * @param yVelocity     velocity in y-axis
         * @param antecedents   array of antecedents for the current cell
         * @param antSize       size of 'antecedents'
         * @param prevOccGrid   linearized 2D array of previous occupancy grid
         */
        __host__ __device__ void getPrediction(float *alphaO, float *alphaE, const int xVelocity, const int yVelocity, Cell *antecedents, const int antSize, const Cell *prevOccGrid);

        /**
         * Normalizes 'alphaOccMatrix' and 'alphaEmpMatrix'
         *
         * @param alphaOccMatrix    linearized 2D array of alphaOccMatrix
         * @param alphaEmpMatrix    linearized 2D array of alphaEmpMatrix
         * @param lvkSum            normalization divisor
         */
        __host__ __device__ void getEstimation(float *alphaOccMatrix, float *alphaEmpMatrix, const float lvkSum);

        /**
         * Computes the new occupancy probability of the cell
         *
         * @param alphaOccMatrix    linearized 2D array of alphaOccMatrix
         */
        __host__ __device__ float getNewOccupiedProbability(const float *alphaOccMatrix);

        /**
         * Computes the new velocity distribution of the cell
         *
         * @param alphaOccMatrix    linearized 2D array of alphaOccMatrix
         * @param alphaEmpMatrix    linearized 2D array of alphaEmpMatrix
         * @param xVelocityKeys
         * @param yVelocityKeys
         */
        __host__ __device__ void updateVelocityProbabilities(const float *alphaOccMatrix, const float *alphaEmpMatrix, const int *xVelocityKeys, const int *yVelocityKeys);

        /**
         * Computes if the current cell is reachable from a given 'cell' with given velocity
         *
         * @param xVelocity     velocity in x-axis
         * @param xVelocity     velocity in x-axis
         * @param cell          previous cell
         */
        __host__ __device__ int isReachable(const int xVelocity, const int yVelocity, const Cell *cell);

        /**
         * Parent function for updating occupancy grid from previous occupancy grid
         *
         * @param prevOccGrid   linearized 2D array of previous occupancy grid
         */
        __host__ __device__ void updateDistributions(Cell *prevOccGrid);

        /**
         * Generic print function for debug purposes
         */
        void toString();
    };

}

/**
 * Kernel for computing distribution of each cell
 *
 * @param cell linearized 2D array of measured occupancy grid
 */
__global__ void computeDistributions(bof::Cell *cell);

void callKernel(bof::Cell *cell);

#endif // __bayesian_cuh__
