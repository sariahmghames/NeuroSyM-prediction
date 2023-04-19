#ifndef READ_JSON_H
#define READ_JSON_H

#include<iostream>
#include <stdio.h>
#include <map>
#include<vector>



#define RADIUS 3.7

using namespace std ;


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


int scene_duration() ; 

map<string, map<string, vector<double> >> extract_json_data(bool shrink_scene, bool cut_frame) ;

#endif



    
