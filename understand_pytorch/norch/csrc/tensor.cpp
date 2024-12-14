#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
// #include <cuda_runtime_api.h>
#include "tensor.h"
// #include "cuda.h"
// #include "cpu.h"


typedef struct {
    float* data;
    int* strides;
    int* shape;
    int ndim;
    int size;
    char* device;
} Tensor;

Tensor* create_tensor(float* data, int* shape, int ndim){
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (tensor == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    tensor->data = data;
    tensor->shape = shape;
    tensor->ndim = ndim;

    tensor->size = 1;
    for (int i=0; i < ndim; i++) {
        tensor->size *= shape[i];
    }

    tensor->strides = (int*)malloc(ndim * sizeof(int));
    if (tensor->strides == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    int stride = 1;
    for (int i = ndim - 1; i >=0 ; i--){
        tensor->strides[i] = stride;
        stride *= shape[i];
    }

    return tensor;
}

float get_item(Tensor* tensor, int* indices) {
    int index = 0;
    for (int i = 0; i < tensor->ndim; i++) {
        index += indices[i] * tensor->strides[i];
    }

    float result;
    result = tensor->data[index];

    return result;
}

Tensor* add_tensor(Tensor* tensor1, Tensor* tensor2) {
    if (tensor1->ndim != tensor2->ndim) {
        fprintf(stderr, "Tensors must have the same number of dimensions %d and %d for addition\n", tensor1->ndim, tensor2->ndim);
        exit(1);
    }

    int ndim = tensor1->ndim;
    int* shape = (int*)malloc(ndim * sizeof(int));
    if (shape == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < ndim; i++) {
        if (tensor1->shape[i] != tensor2->shape[i]) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
    
        }
        shape[i] = tensor1->shape[i];
    }
    float* result_data = (float*)malloc(tensor1->size * sizeof(float));
    if (result_data == NULL){
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    add_tensor_cpu(tensor1, tensor2, result_data);

    return create_tensor(result_data, shape, ndim, device);
}