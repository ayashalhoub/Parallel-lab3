#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ARRAY_SIZE 1000000
#define ROOT_PROCESS 0

void merge(int* array, int start, int middle, int end) {
    int leftSize = middle - start + 1;
    int rightSize = end - middle;
    int* leftArray = (int*)malloc(leftSize * sizeof(int));
    int* rightArray = (int*)malloc(rightSize * sizeof(int));

    for (int i = 0; i < leftSize; i++)
        leftArray[i] = array[start + i];
    for (int j = 0; j < rightSize; j++)
        rightArray[j] = array[middle + 1 + j];

    int leftIndex = 0, rightIndex = 0, mergedIndex = start;
    while (leftIndex < leftSize && rightIndex < rightSize) {
        if (leftArray[leftIndex] <= rightArray[rightIndex])
            array[mergedIndex++] = leftArray[leftIndex++];
        else
            array[mergedIndex++] = rightArray[rightIndex++];
    }

    while (leftIndex < leftSize)
        array[mergedIndex++] = leftArray[leftIndex++];
    while (rightIndex < rightSize)
        array[mergedIndex++] = rightArray[rightIndex++];

    free(leftArray);
    free(rightArray);
}

void parallelMergeSort(int* array, int start, int end) {
    if (start < end) {
        int middle = start + (end - start) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            parallelMergeSort(array, start, middle);

            #pragma omp section
            parallelMergeSort(array, middle + 1, end);
        }

        merge(array, start, middle, end);
    }
}

int main(int argc, char* argv[]) {
    int processRank, processCount;
    int* fullArray = NULL;
    int* localChunk = NULL;
    int localChunkSize;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    if (processRank == ROOT_PROCESS) {
        fullArray = (int*)malloc(ARRAY_SIZE * sizeof(int));
        srand(time(NULL));
        for (int i = 0; i < ARRAY_SIZE; i++) {
            fullArray[i] = rand() % 100000;
        }
    }

    localChunkSize = ARRAY_SIZE / processCount;
    localChunk = (int*)malloc(localChunkSize * sizeof(int));

    MPI_Scatter(fullArray, localChunkSize, MPI_INT, localChunk, localChunkSize, MPI_INT, ROOT_PROCESS, MPI_COMM_WORLD);

    #pragma omp parallel
    {
        #pragma omp single
        parallelMergeSort(localChunk, 0, localChunkSize - 1);
    }

    MPI_Gather(localChunk, localChunkSize, MPI_INT, fullArray, localChunkSize, MPI_INT, ROOT_PROCESS, MPI_COMM_WORLD);

    if (processRank == ROOT_PROCESS) {
        for (int i = 1; i < processCount; i++) {
            merge(fullArray, 0, i * localChunkSize - 1, (i + 1) * localChunkSize - 1);
        }

        printf("Array sorted successfully.\n");

        free(fullArray);
    }

    free(localChunk);
    MPI_Finalize();
    return 0;
}
