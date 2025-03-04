/*---------------------------------------------------------
   DomainClusterer: 
   Search for positive-covariance segments in covariance matrix 
    
    This file is part of the program suite CoMoDo
    (Covariance of Motion Domains)

    Copyright (C)
    Silke A. Wieninger
    G. Matthias Ullmann
    Bayreuth 2014

    www.bisb.uni-bayreuth.de

    This program is free software: you can redistribute
    it and/or modify it under the terms of the
    GNU Affero General Public License as published by the
    Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will
    be useful, but WITHOUT ANY WARRANTY; without even the
    implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE. See the GNU General Public License
    for more details.

    You should have received a copy of the
    GNU Affero General Public License along with this
    program.  If not, see <http://www.gnu.org/licenses/>.
-----------------------------------------------------------*/

#include <stdio.h>
#include <iostream>
#include "Network.h"

using namespace std;

int main(int argc, char *argv[])
{
  /* Variables */
  char filename[100];
  int i, j, k, index1, index2, help;    
  int noNodes1, noNodes2, noCorr1, noCorr2;
  int stopcriterion, cut_domains, verbose=0, matrices=0;
  double largestcov, cut_cov, ave_cov;
  Network EN; 
  
  if(argc < 4)
    {fprintf(stderr, "Usage: DomainClusterer [covariance file name] [-c cutoff Cov/Corr or -d domain number] (-v verbose) (-m step size for intermediate covariance matrices)\n");
      exit(0);} 
 
  if(strcmp(argv[2],"-c")==0)
    {//use covariance cutoff as stopping criterion
      stopcriterion=1;
      cut_cov=atof(argv[3]);}
  if(strcmp(argv[2],"-d")==0)
    {//use domain number as stopping criterion
      stopcriterion=2;
      cut_domains=atoi(argv[3]);}
  
  if(argc == 5)
    {if(strcmp(argv[4],"-v")==0)
	verbose=1; // verbose output; default verbose=0 
      else 
	{fprintf(stderr, "Usage: DomainClusterer [covariance file name] [-c cutoff Cov/Corr or -d domain number] (-v verbose) (-m step size for intermediate covariance matrices)\n");
	  exit(0);}  
    }
  
  if(argc == 6)
    {if(strcmp(argv[4],"-m")==0)
	matrices=atoi(argv[5]); // write out intermediate covariance matrices; default matrices=0
      else
	{fprintf(stderr, "Usage: DomainClusterer [covariance file name] [-c cutoff Cov/Corr or -d domain number] (-v verbose) (-m step size for intermediate covariance matrices)\n");
	  exit(0);}  
    }
  
  if(argc == 7)
    {if(strcmp(argv[4],"-v")==0)
	{verbose=1; // verbose output; default verbose=0 
	  if(strcmp(argv[5],"-m")==0)
	    matrices=atoi(argv[6]); // write out intermediate covariance matrices; default matrices=0
	  else
	    {fprintf(stderr, "Usage: DomainClusterer [covariance file name] [-c cutoff Cov/Corr or -d domain number] (-v verbose) (-m step size for intermediate covariance matrices)\n");
	     exit(0);} 
	}
  
      if(strcmp(argv[4],"-m")==0)
	{matrices=atoi(argv[5]); // write out intermediate covariance matrices; default matrices=0
	  if(strcmp(argv[6],"-v")==0)
	    verbose=1; // verbose output; default verbose=0  
	else
	    {fprintf(stderr, "Usage: DomainClusterer [covariance file name] [-c cutoff Cov/Corr or -d domain number] (-v verbose) (-m step size for intermediate covariance matrices)\n");
	     exit(0);} 
	}  
    }
  
  /* read covariance data from file */
  sprintf(filename,"%s",argv[1]);
  EN.nodes=EN.readCov(filename);

  cout << "Nodes: " << EN.nodes << endl;

  if(stopcriterion==1)  
    cout << "cut cov: " << cut_cov << endl;
  else
    cout << "cut dom: " << cut_domains << endl;
  
  /* Initialization */
  vector<int>::iterator w=EN.whichCluster.begin();
  EN.whichCluster.insert(w,EN.nodes+1,0); // cluster node i belongs to
  for (i=0; i<EN.nodes; i++)
    EN.whichCluster[i]=i; // first each node builds its own cluster
  
  w=EN.nodesCluster.begin();
  EN.nodesCluster.insert(w,EN.nodes+1,1); // number of nodes in cluster i; first set to 1

  EN.number_cluster=EN.nodes; // number of cluster; first equal to number of nodes

  // search largest covariance in cov
  largestcov=EN.searchLargestCov(&index1, &index2);
  
  // cluster until stopping criterion is met, either due to covariance cutoff or due to domain number
  while ((stopcriterion==1 && largestcov>=cut_cov) || (stopcriterion==2 && EN.number_cluster> cut_domains))
    {
      noNodes1=EN.nodesCluster[index1]; // number nodes of cluster with index1
      noNodes2=EN.nodesCluster[index2]; // number nodes of cluster with index2
      
      if(noNodes2>noNodes1) // keep larger cluster
        {
          help=index2;               // exchange index1 with index2, such that always index2 is assigned to index1
          index2=index1;
          index1=help;
          help=noNodes2; 
          noNodes2=noNodes1;
          noNodes1=help;
        }
      noCorr1=noNodes1*noNodes1;
      noCorr2=noNodes2*noNodes2; 

      // merge cluster with highest intercluster covariance
      
      // calculate new intracluster covariances 
      EN.cov[index1][index1]=(EN.cov[index1][index1]*noCorr1+EN.cov[index2][index2]*noCorr2+EN.cov[index1][index2]*2*noNodes1*noNodes2)/(noCorr1+noCorr2+2*noNodes1*noNodes2);

      // assign all nodes of cluster with index2 to cluster with index1
      for(k=0; k<EN.nodes; k++)
        if(EN.whichCluster[k]==index2)
          {
            EN.whichCluster[k]=index1;
            EN.nodesCluster[k]=0;
            EN.nodesCluster[index1]++;
          }    
      EN.number_cluster--;

      // calculate new intercluster covariances
      for (k=0; k<EN.nodes; k++)
        if(EN.nodesCluster[k]!=0 && k!=index1 & k!=index2)
          {
            EN.cov[k][index1]=(EN.cov[k][index1]*noNodes1+EN.cov[k][index2]*noNodes2)/(noNodes1+noNodes2);
            EN.cov[index1][k]=EN.cov[k][index1];
          }

      if(verbose==1) // verbose output
        {
          cout << endl;
          cout << "number cluster: " << EN.number_cluster+1 << endl;
          cout << "largest intercov " << largestcov << " between cluster " << index1+1 << " (nodes: " << noNodes1 << ") and " << index2+1 << " (nodes: " << noNodes2 << ")" << endl;
          cout << "merge cluster " << index2+1 << " with cluster " << index1+1 << ", intraCorr " << EN.cov[index1][index1]  << endl;
        }
      
      // search largest covariance in cov
      largestcov=EN.searchLargestCov(&index1, &index2);
     
      if(matrices!=0) // write out intermediate covariance matrices
	{
	  if(EN.number_cluster%matrices==0 || EN.number_cluster<matrices)
	    {sprintf(filename, "cov%d.out", EN.number_cluster);
	      EN.writeCov(filename);
	    }
	}
    }
  
  cout << endl;
  cout << "Final Domain Number: " << EN.number_cluster << endl;
  
  // write out dynamic domains
  EN.writeDomains();
 	
  return 0;
  
}
