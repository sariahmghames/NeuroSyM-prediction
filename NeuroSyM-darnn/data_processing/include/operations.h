#ifndef OPERATIONS_H  /* prevent C++ header files from being included multiple times. */
#define OPERATIONS_H

#include <bits/stdc++.h> // is basically a way to import every single C++ header file.
#include <iostream>
//#include <math.h>


#define N 3
#define Nt 4 
#define PI 3.14159265


double * substract(double a[], double b[], int size) ; 


double eucl_norm(double *a, int size) ;

 
// Function to find

double dotProduct(double vect_A[], double vect_B[]) ;

double ** dotProduct4b4(double T_A[][Nt], double T_B[][Nt]) ;

double* dotProduct4b1(double T_A[][Nt], double T_B[]) ;

// cross product of two vector array.

double * crossProduct(double vect_A1[], double vect_A[], double vect_B[], int space_dim) ;

#endif


