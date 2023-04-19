#include<iostream>
#include<math.h>
#include<vector>
#include <string>
#include <map>
#include<operations.h>
#include<read_json.h>

using namespace std ; 


void reduce_scene_obj(map<string, vector<double>>& m){

	for(auto mapIt = begin(m); mapIt != end(m); ++mapIt) {

		vector<double> coord_obj = mapIt->second ;

		double* vec = new double[3];
        vec[0] = coord_obj[0] ; 
        vec[1] = coord_obj[1] ;
        vec[2] = coord_obj[2] ;


        double abs_dist = eucl_norm(vec, 3) ;
		if (abs_dist > RADIUS) 
			m.erase(mapIt->first) ;  

	}

}



map<string, std::vector<double>> define_scene_obj(bool shrink_scene= false, bool domain_shift = true){

	map<string, vector<double>> scene_objects ; 



	if (domain_shift==false){
		scene_objects["zbar-order-pt"] = {1.8, 1.25, 0.17} ;
		scene_objects["zbar-ready-pt"] = {2, -1.25, 0.17} ;
		scene_objects["zexit-door"] = {-1.8, -4.6, 0} ;
		scene_objects["zwater-filling-pt"] = {0.8, -4.3, 0.05} ; 
	}
	else {

		cout << "Please define your scene static objects position" << endl ;
	}



	if (shrink_scene == true)
		reduce_scene_obj(scene_objects) ;

	return scene_objects ;
}









