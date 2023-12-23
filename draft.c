#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ROWS 10
#define COLS 10
#define LATENT_FEATURES 10 
#define LEARNING_RATE 0.02
#define REGULARIZATION 0.001
#define ITERATIONS 5000
#define TOLERANCE 1e-6

void randmat(double arr[10][10], int rows, int cols);
void gradientdescent(double R[10][10], double P[10][10], double Q[10][10]);
void printMatrix(double matrix[10][10], int rows, int cols);

// Function for initializing random values to a matrix
void randmat(double arr[10][10], int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            arr[i][j] = (double)rand() / RAND_MAX;
        }
    }
}

// Gradient descent
void gradientdescent(double R[10][10], double P[10][10], double Q[10][10]) {
    for (int iter = 0; iter < ITERATIONS; iter++) {
        double error = 0;

        // Compute error and update P, Q
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                double Result = 0;
                for (int k = 0; k < LATENT_FEATURES; k++) {
                    Result += P[i][k] * Q[k][j];
                }

                if (R[i][j] > 0) {
                    double prediction = Result;
                    error += pow(R[i][j] - prediction, 2);

                    for (int k = 0; k < LATENT_FEATURES; k++) {
                        double gradientP = -2 * (R[i][j] - prediction) * Q[k][j] + 2 * REGULARIZATION * P[i][k];
                        double gradientQ = -2 * (R[i][j] - prediction) * P[i][k] + 2 * REGULARIZATION * Q[k][j];

                        P[i][k] -= LEARNING_RATE * gradientP;
                        Q[k][j] -= LEARNING_RATE * gradientQ;
                    }
                }
            }
        }

        // Check for convergence
        if (error < TOLERANCE) {
            printf("Converged after %d iterations\n", iter + 1);
            break;
        }
    }
}

// Matrix print function
void printMatrix(double matrix[10][10], int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f\t", matrix[i][j]);
        }
        printf("\n");
    }
}

int main() { 
    double R[ROWS][COLS] = {{1, 2, 0, 3, 5, 6, 7, 8, 9, 0}, {5, 0, 9, 5, 7, 8, 9, 0, 4, 3}, 
    {6, 0, 8, 4, 0, 3, 8, 4, 5, 0}, {1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9, 0},{2, 3, 4, 5, 6, 7, 8, 9, 0, 1},
    {8, 9, 0, 1, 2, 3, 4, 5, 6, 0}, {9, 0, 1, 2, 3, 4, 5, 6, 7, 8}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 
    {4, 5, 6, 7, 8, 9, 0, 1, 2, 3},{6, 7, 8, 5, 4, 3, 2, 4, 0, 0}};
    double P[ROWS][LATENT_FEATURES];
    double Q[LATENT_FEATURES][COLS];

    printf("Original Matrix R:\n");
    printMatrix(R, ROWS, COLS);

    randmat(P, ROWS, LATENT_FEATURES);
    randmat(Q, LATENT_FEATURES, COLS);

    gradientdescent(R, P, Q);

    printf("\nFactorized Matrix P:\n");
    printMatrix(P, ROWS, LATENT_FEATURES);

    printf("\nFactorized Matrix Q:\n");
    printMatrix(Q, LATENT_FEATURES, COLS);

    // Display the product of matrices P and Q
    double product[ROWS][COLS];
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            product[i][j] = 0;
            for (int k = 0; k < LATENT_FEATURES; k++) {
                product[i][j] += P[i][k] * Q[k][j];
            }
        }
    }

    printf("\nProduct of Matrices P and Q:\n");
    printMatrix(product, ROWS, COLS);

    return 0;
}
