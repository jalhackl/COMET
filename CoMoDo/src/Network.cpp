/*---------------------------------------------------------
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
#include "Network.h"
#include <stdio.h>
#include <iostream>

using namespace std;

Network::Network()
{
  number_cluster=0;
}

 

double Network::str2dou(const string& str)
{
  double val = 0.0;
  stringstream buffer(str);
  buffer >> val;
  return val;
}

int Network::str2int(const string& str)
{

  int val = 0;
  stringstream buffer(str);
  buffer >> val;
  return val;
}

int Network::readCov(char *filename)
{
  FILE *input;
  char line[1000];
  int index1, index2, i=1, j=1, atomNo=0;
  float value;

  input=fopen(filename,"r");

  if(!input)
    {cout << "cannot open file" << endl;
      exit(1);
    }
  vector<float> p;
 
  while(fgets(line,1000,input)!=0)
    {
      if(sscanf(line,"%d %d %f", &index1, &index2, &value)!=3)
	{cout << "Problem in readCov: wrong numbering in covariance file" << endl;
	exit(0);}
      
      else if (index1==0 or index2==0)
	{cout << "Problem in readCov: wrong numbering in covariance file" << endl;
	  exit(0);}

	/* store all covariances of node i in vector p */
	if(index1==i) //ATTENTION: covariance input file must have right order and start with 1!!!
		{p.push_back(value);
                  j++;
		}
   	else if(index1==i+1 and index2==1) // node i finished; store all vectors p in matrix cov 
          {if(atomNo==0)
              atomNo=p.size();
            else if(p.size()!=atomNo)
              {cout << "Problem in readCov: wrong numbering in covariance file, residue " << i << endl;
                exit(0);}
            cov.push_back(p); 
            i++;
            j=2;
            p.clear();
            p.push_back(value);
          }
        else
          {cout << "Problem in readCov: wrong numbering in covariance file, residue " << i << endl;
            exit(0);}
    }
  cov.push_back(p);
  if(p.size()!=atomNo)
    {cout << "Problem in readCov: wrong numbering in covariance file (last residue)" << endl;
      exit(0);}
  return i;
}


int Network::writeDomains()
{
  int i, j, dom, firstnode, lastnode;
  bool bool_first, bool_moreThan1; 
  dom=0;
  for(i=0; i<nodes; i++)
    {
      lastnode=0;
      if(nodesCluster[i]!=0)
		{dom++;
			cout << endl;
	  	cout << "Nodes in Domain " << dom << ": " << nodesCluster[i] << endl;
	  	cout << "nodes " ;
	  	bool_first=true;
	  	bool_moreThan1=false;
	  	for(j=0; j<nodes; j++)
	    	if(whichCluster[j]==whichCluster[i])
	      	{
			if(bool_first)
		  	{printf("%d", j+1);
		   		bool_first=false;
		    	lastnode=j;
		    	firstnode=j;
		  	}
			else if(lastnode+1!=j)
		  	{if(bool_moreThan1)
		    	printf("-%d,", lastnode+1);
		    	else
		    	printf(",");
		    	bool_first=true;
		    	bool_moreThan1=false;
		    	j--;
		    	lastnode=j;
		  	}
		else
		  {lastnode=j;
		    bool_moreThan1=true;
		  }
	      } 
	  if(bool_moreThan1)
	    printf("-%d\n", lastnode+1);	
	  else
	    printf("\n");
	  
	}
    }
	
	return 0;
}

int Network::writeCov(char *filename)
{
  int i, j, index1=0, index2=0;
  FILE *output;
  output=fopen(filename,"w");
  for(i=0; i<nodes; i++)
    {if(nodesCluster[i]!=0)
	{index1++;
	  index2=0;
	  for(j=0; j<nodes; j++)
	    {
	      if(nodesCluster[j]!=0)
		{
		  index2++;
		  fprintf(output, "%5d %5d %9.5e\n", index1, index2, cov[i][j]);
		}
	    }
	}
    }
	
	return 0;
}

double Network::searchLargestCov(int *index1, int *index2)
{
  int i,j;
  double largestcov;

  for (i=0; i<nodes; i++)
    for (j=i+1; j<nodes; j++)
      if(nodesCluster[i]!=0 && nodesCluster[j]!=0)
	{largestcov=cov[i][j];
	  *index1=i;
	  *index2=j;
	}
  

  for (i=0; i<nodes; i++)
	for (j=i+1; j<nodes; j++)
	  if(cov[i][j]>largestcov && nodesCluster[i]!=0 && nodesCluster[j]!=0)
	    {
	      largestcov=cov[i][j];
	      *index1=i;
	      *index2=j;
	    }
  return largestcov;
}

