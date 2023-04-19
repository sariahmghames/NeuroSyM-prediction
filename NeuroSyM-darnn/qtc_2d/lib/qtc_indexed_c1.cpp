
#include<iostream>
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
vector<int> qtc_val = {-1, 0, +1} ; 
map<vector<int>, int> qtcc1_inf_dict;
map<vector<int>, double> qtcc1_cnd_dict;
vector<int> qtc_vec ;




map<vector<int>, int> qtc_inf(){ // this function implements binary weighting

	list<vector<int>> qtc_C2 ;  // 81 relations possible in 2D
	vector<int> vec ; 
	list<vector<int>> relations_C2_out = not_relations_C2() ; 
	std::ostringstream oss ; 
	int trigger = 0 ;
	vector<int> difference;


	for (int q1 = 0; q1 < qtc_val.size(); q1++){
		for (int q2 = 0; q2 < qtc_val.size(); q2++){
			for (int q3 = 0; q3 < qtc_val.size(); q3++){
				for (int q4 = 0; q4 < qtc_val.size(); q4++){
					vec.insert(vec.begin(), {qtc_val[q1], qtc_val[q2], qtc_val[q3], qtc_val[q4]});
					qtcc1_inf_dict[vec] = 0 ;

					qtc_C2.push_back(vec) ;
					vec.clear() ; 
						
					
				}
			}
		}
	}

	vec.insert(vec.begin(), {10, 10, 10, 10});
	qtc_C2.push_back({10, 10, 10, 10}) ; // ped x left a frame
	qtcc1_inf_dict[vec] = 0 ;
	vec.clear() ;
	qtcc1_inf_dict[{-1,-1,-1,1}] = 1 ; 
	qtcc1_inf_dict[{-1,-1,0,0}] = 1; 
	qtcc1_inf_dict[{-1,-1,1,-1}]=1; 
	qtcc1_inf_dict[{-1,0,-1,1}]= 1;
	qtcc1_inf_dict[{-1,0,0,0}]= 1;
	qtcc1_inf_dict[{-1,0,1,-1}]= 1;
	qtcc1_inf_dict[{0,-1,-1,1}]= 1;
	qtcc1_inf_dict[{0,-1,0,0}]= 1; 
	qtcc1_inf_dict[{0,-1,1,-1}]= 1;

	return qtcc1_inf_dict ;
}




map<vector<int>, double> qtc_cnd(){  // this is the effective function in use for the Neuro-sumbolic approach, we hand copied the input from the generated cnd_labels.txt .. WIP for automatic import

	list<vector<int>> qtc_C2 ;  // 81 relations possible in 2D
	vector<int> vec ; 
	list<vector<int>> relations_C2_out = not_relations_C2() ; 
	std::ostringstream oss ; 


	qtcc1_inf_dict[{-1,-1,-1,-1}] =0.0625;
	qtcc1_inf_dict[{-1,-1,-1,0}]=0.0417;
	qtcc1_inf_dict[{-1,-1,-1,1}]=0.0625;
	qtcc1_inf_dict[{-1,-1,0,-1}]=0.0417;
	qtcc1_inf_dict[{-1,-1,0,0}]=0.0278;
	qtcc1_inf_dict[{-1,-1,0,1}]=0.0417;
	qtcc1_inf_dict[{-1,-1,1,-1}]=0.0625;
	qtcc1_inf_dict[{-1,-1,1,0}]=0.0417;
	qtcc1_inf_dict[{-1,-1,1,1}]=0.0625;
	qtcc1_inf_dict[{-1,0,-1,-1}]=0.0417;
	qtcc1_inf_dict[{-1,0,-1,0}]=0.0278;
	qtcc1_inf_dict[{-1,0,-1,1}]=0.0417;
	qtcc1_inf_dict[{-1,0,0,-1}]=0.0278;
	qtcc1_inf_dict[{-1,0,0,0}]=0.0185;
	qtcc1_inf_dict[{-1,0,0,1}]=0.0278;
	qtcc1_inf_dict[{-1,0,1,-1}]=0.0417;
	qtcc1_inf_dict[{-1,0,1,0}]=0.0278;
	qtcc1_inf_dict[{-1,0,1,1}]=0.0417;
	qtcc1_inf_dict[{-1,1,-1,-1}]=0.0625; 
	qtcc1_inf_dict[{-1,1,-1,0}]=0.0417;
	qtcc1_inf_dict[{-1,1,-1,1}]=0.0625 ;
	qtcc1_inf_dict[{-1,1,0,-1}]=0.0417;
	qtcc1_inf_dict[{-1,1,0,0}]=0.0278;
	qtcc1_inf_dict[{-1,1,0,1}]=0.0417;
	qtcc1_inf_dict[{-1,1,1,-1}]=0.0625; 
	qtcc1_inf_dict[{-1,1,1,0}]=0.0417;
	qtcc1_inf_dict[{-1,1,1,1}]=0.0625 ;
	qtcc1_inf_dict[{0,-1,-1,-1}]=0.0417;
	qtcc1_inf_dict[{0,-1,-1,0}]=0.0278;
	qtcc1_inf_dict[{0,-1,-1,1}]=0.0417;
	qtcc1_inf_dict[{0,-1,0,-1}]=0.0278;
	qtcc1_inf_dict[{0,-1,0,0}]=0.0185;
	qtcc1_inf_dict[{0,-1,0,1}]=0.0278;
	qtcc1_inf_dict[{0,-1,1,-1}]=0.0417;
	qtcc1_inf_dict[{0,-1,1,0}]=0.0278;
	qtcc1_inf_dict[{0,-1,1,1}]=0.0417;
	qtcc1_inf_dict[{0,0,-1,-1}]=0.0278;
	qtcc1_inf_dict[{0,0,-1,0}]=0.0185;
	qtcc1_inf_dict[{0,0,-1,1}]=0.0278;
	qtcc1_inf_dict[{0,0,0,-1}]=0.0185;
	qtcc1_inf_dict[{0,0,0,0}]=0.0123 ;
	qtcc1_inf_dict[{0,0,0,1}]=0.0185;
	qtcc1_inf_dict[{0,0,1,-1}]=0.0278;
	qtcc1_inf_dict[{0,0,1,0}]=0.0185;
	qtcc1_inf_dict[{0,0,1,1}]=0.0278;
	qtcc1_inf_dict[{0,1,-1,-1}]=0.0417; 
	qtcc1_inf_dict[{0,1,-1,0}]=0.0278;
	qtcc1_inf_dict[{0,1,-1,1}]=0.0417 ;
	qtcc1_inf_dict[{0,1,0,-1}]=0.0278;
	qtcc1_inf_dict[{0,1,0,0}]=0.0185;
	qtcc1_inf_dict[{0,1,0,1}]=0.0278;
	qtcc1_inf_dict[{0,1,1,-1}]=0.0417; 
	qtcc1_inf_dict[{0,1,1,0}]=0.0278;
	qtcc1_inf_dict[{0,1,1,1}]=0.0417 ;
	qtcc1_inf_dict[{1,-1,-1,-1}]=0.0625;
	qtcc1_inf_dict[{1,-1,-1,0}]=0.0417 ;
	qtcc1_inf_dict[{1,-1,-1,1}]=0.0625;
	qtcc1_inf_dict[{1,-1,0,-1}]=0.0417 ;
	qtcc1_inf_dict[{1,-1,0,0}]=0.0278;
	qtcc1_inf_dict[{1,-1,0,1}]=0.0417 ;
	qtcc1_inf_dict[{1,-1,1,-1}]=0.0625;
	qtcc1_inf_dict[{1,-1,1,0}]=0.0417 ;
	qtcc1_inf_dict[{1,-1,1,1}]=0.0625;
	qtcc1_inf_dict[{1,0,-1,-1}]=0.0417; 
	qtcc1_inf_dict[{1,0,-1,0}]=0.0278;
	qtcc1_inf_dict[{1,0,-1,1}]=0.0417 ;
	qtcc1_inf_dict[{1,0,0,-1}]=0.0278;
	qtcc1_inf_dict[{1,0,0,0}]=0.0185;
	qtcc1_inf_dict[{1,0,0,1}]=0.0278;
	qtcc1_inf_dict[{1,0,1,-1}]=0.0417; 
	qtcc1_inf_dict[{1,0,1,0}]=0.0278;
	qtcc1_inf_dict[{1,0,1,1}]=0.0417 ;
	qtcc1_inf_dict[{1,1,-1,-1}]=0.0625;
	qtcc1_inf_dict[{1,1,-1,0}]=0.0417 ;
	qtcc1_inf_dict[{1,1,-1,1}]=0.0625;
	qtcc1_inf_dict[{1,1,0,-1}]=0.0417 ;
	qtcc1_inf_dict[{1,1,0,0}]=0.0278;
	qtcc1_inf_dict[{1,1,0,1}]=0.0417 ;
	qtcc1_inf_dict[{1,1,1,-1}]= 0.0625; 
	qtcc1_inf_dict[{1,1,1,0}]=0.0417;
	qtcc1_inf_dict[{1,1,1,1}]=0.0625 ;
	qtcc1_inf_dict[{10,10,10,10}]= 0 ;


	return qtcc1_cnd_dict ;
}


