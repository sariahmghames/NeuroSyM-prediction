//#include<iostream>
#include<math.h>
#include<cmath>
#include<complex>
//#include<algorithm>
#include<operations.h>

using namespace std ; 



double * substract(double a[], double b[], int size){
   double* c = new double[size];


   for(int i = 0; i < size; i++){
       c[i] = a[i] - b[i];
   }

   return c ; 
}


double eucl_norm(double *a, int size){
   double c = 0 ;


   for(int i = 0; i < size; i++){
       c += pow(*(a+i),2);
   }

   return sqrt(c) ; 
}




double dotProduct(double vect_A[3], double vect_B[3])
{
 
   double product = 0.0 ;

   // Loop for calculate dot product
   for (int i = 0; i < 3; i++)
 
      product += vect_A[i] * vect_B[i];
   
   return product;
}


double ** dotProduct4b4(double T_A[4][4], double T_B[4][4])
{
 
   double sum_p = 0.0 ;

   double** Tf = new double*[4]; 

   for (int i = 0; i < 4; ++i)
   {
    Tf[i] = new double[4];
   }


   // Loop for calculate dot product
   for (int i = 0; i < 4; i++){
      for (int j = 0; i < 4; j++){
 
         sum_p += T_A[i][j] * T_B[j][i]; 

         Tf[i][j] = sum_p ;
         sum_p = 0.0 ;
      }
   }
   
   return Tf;
}


double* dotProduct4b1(double T_A[4][4], double T_B[4])
{
 

   double* Vf = new double[4];
   //vector<double> Vf ;
   double sum_p = 0.0 ;


   // Loop for calculate dot product
   for (int i = 0; i < 4; i++){
      for (int j = 0; i < 4; j++){
 
         sum_p += T_A[i][j] * T_B[j];
      }
         //Vf.push_back(sum_p) ;
         Vf[i] = sum_p ;
         sum_p = 0.0 ;
      
   }
   
   return Vf;
}




 
// Function to find
// cross product of two vector array.
double * crossProduct(double vect_A1[3], double vect_A[3], double vect_B[3], int space_dim)
 
{
   double *vect_Ad; 
   double *vect_Bd;
   double* cross_P = new double[space_dim];

   vect_Ad = substract(vect_A1, vect_A, space_dim) ;
   vect_Bd = substract(vect_B, vect_A, space_dim) ;

   cross_P[0] = vect_Ad[1] * vect_Bd[2] - vect_Ad[2] * vect_Bd[1];
   cross_P[1] = vect_Ad[2] * vect_Bd[0] - vect_Ad[0] * vect_Bd[2];
   cross_P[2] = vect_Ad[0] * vect_Bd[1] - vect_Ad[1] * vect_Bd[0];

   return cross_P ;

}


