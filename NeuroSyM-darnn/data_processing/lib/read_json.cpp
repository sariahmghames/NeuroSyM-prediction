#include <read_json.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include "geometry_msgs/Point.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include<vector>
#include <iterator>
//#include <zip.h>
//#include <filesystem> //c++ 17
#include <dirent.h>
#include <boost/filesystem.hpp>
#include <set>
#include <json/json.h>
#include <map>
#include <operations.h>



boost::filesystem::path full_path(boost::filesystem::current_path());


namespace fs = boost::filesystem;

using namespace std ;
using namespace boost::filesystem;


std::set<fs::path> sorted_files ;
int scene_time_frame ; 
vector<double> x_u = {1.0, 0.0, 0.0} ;
double sight = 60 * PI / 180.0 ;


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int one_file_size ; int one_subfile_size ; int size_att2 ;
string file_key ; string file_key1 ; 
int file_loop = 0; int loop_frame = 0 ; 
string::size_type sz; 


Json::Reader reader;
Json::Value content ;
Json::Value attribute1 ;
Json::Value attribute2 ;
string lowest_att ;
double lowest_att_value ;
vector<string> pcd_files ;



// Change your dataset path
path pcd_path = "~/darko_qtc/tmp_data/pointclouds/upper_velodyne/packard-poster-session-2019-03-20_1/";  // replace later by " path p(argc>1? argv[1] : "."); "
path img_path = "~/darko_qtc/tmp_data/stitched_images/packard-poster-session-2019-03-20_1/";


map<string, vector<double>> inner_map ;


map<string, map<string, vector<double> >> full_peds_traj  ;

map<string, map<string, vector<double> >>::iterator it_mapfull;
map<string, vector<double> >::iterator it_innermap;
int file2nb = 0 ;
vector<double> vec ;
vector<vector<double>> vec1 ;
vector<double>::iterator it_vec;






void resize_scene(MAP<MAP<vector<double>>>& m2) {

  vector<double> x, y, z ;
  vector<double> v1(3, 0) ;
  double sight_t, abs_dist, dotp, forward_proj ;
  for(auto mapIt = begin(m2); mapIt != end(m2); ++mapIt){
    for(auto mapIt1 = begin(mapIt->second); mapIt1 != end(mapIt->second); ++mapIt1){
      if (mapIt1->first == "cx")
        x = mapIt1->second ;
      if (mapIt1->first == "cy")
        y = mapIt1->second ;
      if (mapIt1->first == "cz")
        z = mapIt1->second ; }

      bool ped_in_radius = false ;
      int switch1 = 0 ;
      for (int t = 0; t < scene_time_frame; ++t){

        double* vec = new double[3];
        vec[0] = x[t] ; 
        vec[1] = y[t] ;
        vec[2] = z[t] ;

        for(int i = 0; i < 3; i++){
          v1[i] = *(vec+i) ;
        }
        
        abs_dist = eucl_norm(vec, 3) ;   

        if (abs_dist < RADIUS){
          ped_in_radius = true ; 
          switch1 = 1 ; }
        else 
          ped_in_radius = false ; 

      }

      if (switch1 == 0) 
        m2.erase(mapIt->first) ;
      
  }


}



void InitMap(MAP<MAP<vector<double>>>& m1, string label_id, string lowest_att)
{

  it_mapfull = m1.find(label_id) ;

  if (it_mapfull == m1.end()) {    

    inner_map.insert({lowest_att, vector<double> (scene_time_frame, 0.0)}) ;
    m1.insert({label_id, inner_map}) ;
    inner_map.clear() ;
    
  }
  else {
    inner_map = it_mapfull->second ; 
    it_innermap = inner_map.find(lowest_att) ;
    if (it_innermap == inner_map.end()) {
       m1[label_id].insert({lowest_att, vector<double> (scene_time_frame, 0.0)}) ;
     }
    inner_map.clear() ;
   }


}




void map_fill_frame(MAP<MAP<vector<double>>>& m1, int T) {

// fill remaining fields of one frame of all peds with 0's 

 for(auto mapIt = begin(m1); mapIt != end(m1); ++mapIt)
  {

    for(auto mapIt1 = begin(mapIt->second); mapIt1 != end(mapIt->second); ++mapIt1)
    {
      for(int i = 0 ; i < T; i++ ){
      
        if (mapIt1->second[i]){
        }
        else
          mapIt1->second[i] = 0 ; 
      
      }

    }

  }

}






int scene_duration(){

  std::ifstream jsonfile("/home/sariah/darko_qtc/tmp_data/labels/3d_labels/packard-poster-session-2019-03-20_1" ); // or add std::ifstream::binary after the .json file

  bool parseSuccess = reader.parse(jsonfile, content, false);


  if (parseSuccess){
  if(is_directory(pcd_path))
    {
        for (auto &entry : fs::directory_iterator(pcd_path)){
          sorted_files.insert(entry.path());
        }

        scene_time_frame = sorted_files.size() ; 


    }
  }

  return scene_time_frame ;
}




map<string, map<string, vector<double> >> extract_json_data(bool shrink_scene= false, bool cut_frame= false) {

  std::ifstream jsonfile("~/darko_qtc/tmp_data/labels/3d_labels/packard-poster-session-2019-03-20_1.json" ); 

  bool parseSuccess = reader.parse(jsonfile, content, false);


  if (parseSuccess){
  if(is_directory(pcd_path))
    {
        for (auto &entry : fs::directory_iterator(pcd_path)){
          sorted_files.insert(entry.path());
        }

        scene_time_frame = sorted_files.size() ; 
        cout << "sorted files = " << scene_time_frame << "\n" ;


        for (auto &filename : sorted_files)
        {   
 
            file_loop++  ;

            string file_frame = filename.filename().string() ;

            one_file_size = content["labels"][file_frame].size() ;


            for (int i = 0; i < one_file_size ; i++) { 
              loop_frame++ ;
              file_key = file_frame;  // e.g: "000000.pcd"
              file2nb = stoi(file_key, &sz) ; 


              attribute1 = content["labels"][file_key][i] ;
              for (Json::Value::const_iterator it1 = attribute1.begin(); it1 != attribute1.end(); ++it1) {
                  one_subfile_size = (it1.key().asString()).size() ;
                  file_key1 = it1.key().asString() ; // action_label

                  if (file_key1 == "box"){
                      attribute2 = attribute1[file_key1] ; // e.g value of action_label  // or attribute1.get (const std::string &key, const Value &defaultValue) const --> key is action_label
                      size_att2 = attribute2.size() ; // has 1 value
                      Json::Value::const_iterator label_it = next(it1, 2) ; // .key().asString() ;
                      string label = label_it.key().asString() ; // iterator

                      for (Json::Value::const_iterator it2 = attribute2.begin(); it2 != attribute2.end(); ++it2) {
                          lowest_att = it2.key().asString(); // e.g "sitting"; "cx"
                          string label_id = attribute1[label].asString() ;
                          lowest_att_value = attribute2[lowest_att].asDouble() ;

                          InitMap(full_peds_traj, label_id, lowest_att) ;
                          ////peds_traj[label_id][lowest_att].push_back(lowest_att_value) ;
                          

                          vector<double> &vec = full_peds_traj[label_id][lowest_att] ;
                          it_vec = vec.begin();
                          vec[file2nb] = lowest_att_value;  // returns iterator 

                      }

                  }

            }
          }

        }

    }

  }

  map<string, map<string, vector<double> >> local_peds_traj  = full_peds_traj;

  if (shrink_scene == true)
    resize_scene(local_peds_traj) ;
  return local_peds_traj ;

}



int main (int argc, char *argv[]){
  extract_json_data(false) ;

  return 0 ;
}