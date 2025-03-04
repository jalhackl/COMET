#ifndef NETWORK_H
#define NETWORK_H

#include <list>
#include <vector>
#include <stdio.h>
#include <fstream>
#include <string>
#include <cstring>
#include <stdlib.h>
#include <sstream>

using namespace std;

class Network
{
  public:
  int nodes; // total number of nodes in elastic network 
  int number_cluster; // actual number of cluster
  vector<int> whichCluster; // cluster node i belongs to; first all nodes set to zero;
  vector<int> nodesCluster; // number of nodes in cluster i; first all set to one
  vector<vector<float> > cov;
   
  //constructor
  Network();

  //copy constructor
  Network(const Network &);

  //destructor
  ~Network(){};
  

  int readCov(char *filename);

  double str2dou(const string& str);

  int str2int(const string& str);

  int writeDomains();

  int writeCov(char *filename);

  double searchLargestCov(int *index1, int *index2);
};

#endif


 
 
