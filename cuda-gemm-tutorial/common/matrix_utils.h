#pragma once

#include "types.h"
#include <random>
#include <cmath>
#include <cstdio>
#include <algorithm>

// ============================================================================
// Matrix Utilities
// ============================================================================
// Host-side utilities for matrix initialization, validation, and comparison.
// Templates support both fp16_t and bf16_t.
// ============================================================================

// ----------------------------------------------------------------------------
// Initialization
// ----------------------------------------------------------------------------

// Fill matrix with uniform random values in [min_val, max_val]
template<typename T>
void init_matrix_random(T* data, int rows, int cols,
                        float min_val = -1.0f, float max_val = 1.0f,
                        unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(min_val, max_val);
    for (int i = 0; i < rows * cols; i++) {
        data[i] = from_float<T>(dis(gen));
    }
}

// Fill matrix with a constant value
template<typename T>
void init_matrix_constant(T* data, int rows, int cols, float value) {
    for (int i = 0; i < rows * cols; i++) {
        data[i] = from_float<T>(value);
    }
}

// Fill square identity matrix
template<typename T>
void init_matrix_identity(T* data, int size) {
    for (int i = 0; i < size * size; i++) data[i] = from_float<T>(0.0f);
    for (int i = 0; i < size; i++)         data[i * size + i] = from_float<T>(1.0f);
}

// ----------------------------------------------------------------------------
// Error Metrics
// ----------------------------------------------------------------------------

// Maximum absolute difference: max |a[i] - b[i]|
template<typename T>
float max_abs_error(const T* a, const T* b, int n) {
    float err = 0.0f;
    for (int i = 0; i < n; i++) {
        err = std::max(err, std::abs(to_float(a[i]) - to_float(b[i])));
    }
    return err;
}

// Root-mean-square relative error: sqrt(sum(d^2) / sum(b^2))
template<typename T>
float rms_relative_error(const T* a, const T* b, int n) {
    float sq_err = 0.0f, sq_ref = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = to_float(a[i]) - to_float(b[i]);
        float r = to_float(b[i]);
        sq_err += d * d;
        sq_ref += r * r;
    }
    return std::sqrt(sq_err / (sq_ref + 1e-10f));
}

// Return true if matrices are numerically close
template<typename T>
bool matrices_are_close(const T* a, const T* b, int n,
                        float abs_tol = 1e-2f, float rel_tol = 1e-2f) {
    return max_abs_error(a, b, n) < abs_tol &&
           rms_relative_error(a, b, n) < rel_tol;
}

// ----------------------------------------------------------------------------
// Debug Printing
// ----------------------------------------------------------------------------

// Print first max_rows x max_cols elements of matrix
template<typename T>
void print_matrix(const T* data, int rows, int cols,
                  int max_rows = 8, int max_cols = 8,
                  const char* label = nullptr) {
    if (label) printf("%s (%dx%d):\n", label, rows, cols);
    int pr = std::min(rows, max_rows);
    int pc = std::min(cols, max_cols);
    for (int i = 0; i < pr; i++) {
        for (int j = 0; j < pc; j++) {
            printf("%8.4f", to_float(data[i * cols + j]));
        }
        if (cols > pc) printf("  ...");
        printf("\n");
    }
    if (rows > pr) printf("  ...\n");
}
