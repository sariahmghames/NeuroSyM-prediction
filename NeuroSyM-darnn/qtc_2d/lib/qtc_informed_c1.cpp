// QTC-based informed spatial interactions

// QTC_C1 with 4 symbols

#include<iostream>
#include <fstream>
#include<math.h>
#include<vector>
#include <string>
#include <map>
#include<list>
#include <sstream> // std::stringstream
#include <algorithm>
#include <iterator>

using namespace std ; 


int qtc_rel = 4 ; 

void informed_qtc_c1(){

	vector<int> qtc_val = {-1, 0, +1} ; 

	vector<int> qtc_c1_inf ; 
	vector<int> qtc_c1_ninf ; 
	list<vector<int>> qtc_c1_comb ; 
	vector<int> vec ; 

	ofstream myfile;
  	myfile.open("qtcc1_labels.txt");


	for (int q1 = 0; q1 < qtc_val.size(); q1++){
		for (int q2 = 0; q2 < qtc_val.size(); q2++){
			for (int q3 = 0; q3 < qtc_val.size(); q3++){
				for (int q4 = 0; q4 < qtc_val.size(); q4++){
					vec.insert(vec.begin(), {qtc_val[q1], qtc_val[q2], qtc_val[q3], qtc_val[q4]});

					qtc_c1_comb.push_back(vec) ;

					myfile << "{" << qtc_val[q1] << ',' << qtc_val[q2] << ',' << qtc_val[q3] <<  ',' << qtc_val[q4] << "}" << '\n' ;
					

					vec.clear() ; 	
				}
			}
		}
	}
	myfile.close() ;

}


int main(){
	informed_qtc_c1() ;
	return 0;
}

