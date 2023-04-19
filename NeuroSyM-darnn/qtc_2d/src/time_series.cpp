#include<math.h>
#include<vector>
#include<numeric>
#include<operations.h>
#include<iterator>
#include<stdio.h> 
#include <utility> // std::pair
#include <stdexcept> 
#include <sstream> 
#include <string>
#include<fstream> 
#include <stdlib.h>     /* exit, EXIT_FAILURE */
#include<read_json.h>
#include <map>
#include<algorithm> 
#include<random>
#include<objects.h>
#include<qtc_indexed_c1.h>


const int space_dim = 2 ;
const int qtc_rel = 4 ;
double *ak ; 
double *al; ;
int nb_ped = 37 ;
bool add_noise = false ; 
bool base3 = false ; 
bool store_qtc_index = false ; 
bool add_context = true ;
bool interactions = true ;



vector<double> ped_k_x , ped_k_y, ped_k_z, ped_l_x, ped_l_y, ped_l_z ;


map<string,  vector<double> > abs_traj ;

vector<int> v ; 
vector<string> centers ;

map<string,  vector<vector<double>> > ts_one_cluster ;
vector<map<string, vector<vector<double>> > > clusters ; 



string Xts_map_to_csv(map<string, vector<vector<double>> > data, int i, string center, int ii){

    string pedk = data.begin()->first ; 
    string csv_file = "X"+ to_string(ii) + "_ts_bytes-cafe-2019-02-07_0_" +  center + ".csv" ;
    // Create an output filestream object
    std::ofstream myFile(csv_file);
    vector<vector<double>> arr ; 

    int loop = 0 ; 

    // Send data to the stream
    auto mapIt = begin(data) ;

    while(mapIt != end(data)){
        cout << mapIt->first << endl ; 
        myFile <<  mapIt->first << "," ; 
        ++mapIt ;
        
    }

    mapIt = begin(data) ;
    auto mapIt1 = begin(mapIt->second) ;
    myFile << "\n" ;
    
    arr = mapIt->second ; 


    for (int r = 0; r < arr.size() ; ++r){  

        while(loop != data.size()){
            arr = mapIt-> second ;


            myFile  << arr[r][ii] << "," ;
            
            ++mapIt ; 
            loop++ ; 
        }

        myFile << "\n" ;
        mapIt = begin(data) ;
        loop = 0 ; 

    }

    myFile << "\n" ;

    // Close the file
    myFile.close();

    return csv_file ;

}



string Yts_map_to_csv(map<string, vector<vector<double>> > data, int i, string center, int ii){

    string pedk = data.begin()->first ; 
    string csv_file = "Y" + to_string(ii) + "_ts_bytes-cafe-2019-02-07_0_" + center + ".csv" ;

    // Create an output filestream object
    std::ofstream myFile(csv_file);
    vector<vector<double>> arr ; 
    int loop = 0 ; 

    // Send data to the stream
    auto mapIt = begin(data) ;


    while(mapIt != end(data)){
        myFile <<  mapIt->first << "," ; 
        ++mapIt ;    
    }


    mapIt = begin(data) ;
    auto mapIt1 = begin(mapIt->second) ;
    myFile << "\n" ;
    
    arr = mapIt->second ; 

    for (int r = 1; r < arr.size() ; ++r){  

        while(loop != data.size()){
            arr = mapIt-> second ;

            myFile << arr[r][ii]  << "," ;

            ++mapIt ; 
            loop++ ; 
        }
        
        myFile << "\n" ;
        mapIt = begin(data) ;
        loop = 0 ; 
    }


    while(loop != data.size()){
        arr = mapIt-> second ;


        myFile << arr[(arr.size())-1][ii]  << "," ; // arr[-1][ii]

        ++mapIt ; 
        loop++ ; 
    }

    myFile << "\n" ;

    // Close the file
    myFile.close();

    return csv_file ;

}




int main (int argc, char *argv[]){

    map<string, map<string, vector<double> >> peds_traj ;
    map<string, vector<double>> scene_objects ;

    cout << "Start importing dataset" << endl ;
    peds_traj = extract_json_data(false, false) ;
    cout << peds_traj.size() ;

    cout << " start mapping to csv" ;
    map_to_csv(peds_traj) ;


    if (interactions == true){

        vector<double> abs_traj_kl(ped_k_x.size()) ;
        double* vec = new double[2];

        int loop_mapIt = 1 ; 


        int scene_time_frame = scene_duration() ; 

        if (add_context == true){

            scene_objects = define_scene_obj(false, false) ; 

            for(auto mapIt = begin(scene_objects); mapIt != end(scene_objects); ++mapIt)
            {
                vector<double> v1(scene_time_frame, mapIt->second[0]) ;
                vector<double> v2(scene_time_frame, mapIt->second[1]) ;
                vector<double> v3(scene_time_frame, mapIt->second[2]) ;

                peds_traj.insert(peds_traj.end(),  std::pair<string,map<string, vector<double>>>(mapIt->first, {})) ; 
                peds_traj.insert(peds_traj.end(), std::pair<string,map<string, vector<double>>>(mapIt->first, {})) ; 
                peds_traj.insert(peds_traj.end(), std::pair<string,map<string, vector<double>>>(mapIt->first, {})) ; 
                peds_traj[mapIt->first]["cx"] = v1 ;
                peds_traj[mapIt->first]["cy"] = v2 ;
                peds_traj[mapIt->first]["cz"] = v3 ;

            }
        }


        int peds_traj_size = peds_traj.size() ;
        cout << "peds_traj size =" << peds_traj.size() << endl ;
        string ped_k ;
        int traj_size ;

        try{    
            for(auto mapIt0 = peds_traj.begin(); mapIt0 != peds_traj.end(); ++mapIt0)
            {
                ped_k = mapIt0->first ; //label
                cout << "ped_k=" << ped_k << endl ; 
                for(auto mapIt01 = begin(mapIt0->second); mapIt01 != end(mapIt0->second); ++mapIt01){
                    if  (mapIt01->first == "cx")
                        ped_k_x = mapIt01->second ; 
                    else if (mapIt01->first == "cy")
                        ped_k_y = mapIt01->second ; 
                    else if (mapIt01->first == "cz")
                        ped_k_z = mapIt01->second ;
                    else {} ; }
                vector<vector<double>>  k_traj; 
                vector<double> abs_traj_k;
                double* vec_k = new double[3];

                for (int i = 0; i < ped_k_x.size(); i++){
                    vec_k[0] = ped_k_x[i] ; 
                    vec_k[1] = ped_k_y[i] ;
                    vec_k[2] = ped_k_z[i] ;
                    abs_traj_k.push_back(eucl_norm(vec_k, 3)) ;
                    vector<double> vectk{vec_k[0], vec_k[1], vec_k[2]} ;
                    k_traj.push_back(vectk) ;
                    vectk.clear() ;

                }


                ts_one_cluster[ped_k] = k_traj ;
                k_traj.clear() ;

                for(auto mapIt1 = peds_traj.begin(); mapIt1 != peds_traj.end(); ++mapIt1) {
                    if(mapIt1->first != mapIt0->first){
                        int loop_mapIt1 = 1 ;

                        
                        string ped_l = mapIt1->first ; //label
                        string ped_kl = ped_k + "-" + ped_l ;
              
                        for(auto mapIt11 = begin(mapIt1->second); mapIt11 != end(mapIt1->second); ++mapIt11){
                            if (mapIt11->first == "cx")
                                ped_l_x = mapIt11->second ; 
                            else if (mapIt11->first == "cy")
                                ped_l_y = mapIt11->second ; 
                            else if (mapIt11->first == "cz")
                                ped_l_z = mapIt11->second ;
                            else {} ; 
                        }

                        
                        vector<vector<double>> l_traj ;

                        vector<double> abs_traj_l ;
                        traj_size = ped_l_x.size() ;

                        
                        double* vec_l = new double[3];


                        for (int j = 0; j < traj_size; j ++){

                            double* vec_kl = new double[2];
                            vec_kl[0] = ped_k_x[j] - ped_l_x[j] ; 
                            vec_kl[1] = ped_k_y[j] - ped_l_y[j] ;


                            double abs_dist = eucl_norm(vec_kl, 2) ;


                            if ((abs_dist <= RADIUS) && (abs_dist > 0.0) && ((vec_kl[0] != ped_k_x[j]) && (vec_kl[0] != ped_l_x[j])) &&  ((vec_kl[1] != ped_k_y[j]) && (vec_kl[1] != ped_l_y[j])) ) {

                                for (int j = 0; j < traj_size; j++){

                                    vec_l[0] = ped_l_x[j] ; 
                                    vec_l[1] = ped_l_y[j] ;
                                    vec_l[2] = ped_l_z[j] ;
                                    abs_traj_l.push_back(eucl_norm(vec_l, 3)) ;
                                    vector<double> vectl{vec_l[0], vec_l[1], vec_l[2]} ;
                                    l_traj.push_back(vectl) ;
                                    vectl.clear() ;
                                }


                                int time_steps_k =  sizeof(k_traj) / sizeof(k_traj[0]) ;
                                int time_steps_l =  sizeof(l_traj) / sizeof(l_traj[0]) ;


                                if (time_steps_k != time_steps_l){
                                    //ERROR
                                    cout << "ped k size =" << time_steps_k << endl;
                                    cout << "ped l size =" << time_steps_l << endl ;
                                    cout << "ERROR: The input series must have the same length" << endl ;
                                }


                                //ts_one_cluster[ped_l] = abs_traj_l ;
                                ts_one_cluster[ped_l] = l_traj ;

                                //abs_traj_l.clear() ;
                                l_traj.clear() ;

                                ++loop_mapIt1 ; 

                                break; // if only at single time step ped l is inside the circle of radius RADIUS and centered at ped k, then we do qtc for the whole time frame of couple kl
                            
                            }
                        
                        }
                    }

                }


            if (ts_one_cluster.size() != 0){
                clusters.push_back(ts_one_cluster) ;
                centers.push_back(ped_k) ;

            }
            ts_one_cluster.clear() ; 

            ++loop_mapIt ;   


            }

            cout << "loop_mapIt=" << loop_mapIt << endl ;
            cout << "clusters size=" << clusters.size() << endl ;

        }


        catch (std::bad_alloc & exception) 
        { 
            std::cerr << "bad_alloc detected: " << exception.what() << endl ; 
        } 

        int max = 0 ;
        int m = 3;

        vector<vector<double>> vec0(traj_size, vector<double> (m, 0)) ;

        for (int i =0; i < clusters.size() ; i++ ) {
            if (clusters[i].size() > max)
                max = clusters[i].size() ;
        }


        for (int i =0; i < clusters.size() ; i++ ) {
            
            int init_cluster_size = clusters[i].size() ;
            if (init_cluster_size < max){
                // Fill the remaining columns 
                for (int s= 0; s< (max-init_cluster_size); ++s){

                    clusters[i].insert(clusters[i].end(), std::pair<string,vector<vector<double>>>("unk"+to_string(s), vec0)) ;
                }

            }

            //string file1 = Xts_map_to_csv(clusters[i], i, centers[i], 0) ; // clusters[i] type : map<string, vector<vector<double>> > data,
            string file2 = Xts_map_to_csv(clusters[i], i, centers[i], 1) ;
            //string file3 = Xts_map_to_csv(clusters[i], i, centers[i], 2) ;


        }

        for (int i =0; i< clusters.size() ; i++ ) {

            //string file1 = Yts_map_to_csv(clusters[i], i, centers[i], 0) ; 
            string file2 = Yts_map_to_csv(clusters[i], i, centers[i], 1) ;
            //string file3 = Yts_map_to_csv(clusters[i], i, centers[i], 2) ;
        }

        cout << "max =" << max ; 

    }

    return 0 ;

}

