/*---------------------------------------------------------
   DomainTester: 
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
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <list>
#include <cstdlib>
#include <map>
#include <vector>
#include <sstream>

using namespace std;
int readCov(char *filename, vector<vector<float> > &cov);
int number(char *s1, int *pos);

int main(int argc, char *argv[])
{
  /* Variables */ 
  char filename[100];
  vector<vector<float> > cov;
  vector<vector<float> >::iterator z;
  vector<int> inPosCovSeg;
  vector<int>::iterator h;
  int domainSize, natoms, ndomains=0, nSepDomains=0;			
  int i, j, k, l, lastNode=0, lastNodeSeg=-1;
  int cut; 
  bool goOn;
  
  if(argc != 3)
    {fprintf(stderr, "Usage: DomainTester [covariance file name] [minimal positive-covariance segment size]\n");
      exit(0);}  
  cut=atoi(argv[2]);
 
  /* read covariances , store in matrix cov */
  sprintf(filename,"%s",argv[1]);
  readCov(filename, cov);

  z=cov.begin();
  natoms=(*z).size();
  cout << "atom number " << natoms << endl;
  
  /* initialize vector inPosCovSeg */
  h=inPosCovSeg.begin();      // is node i part of a positive-covariance segment?
  inPosCovSeg.insert(h,natoms,0); // 0 no, 1 yes 
 
  for(i=0; i<=natoms-1; i++)
    {domainSize=1;
      goOn=true;
      for(j=i+1; j<=natoms-1; j++)
	{
	  for(k=i; k<=j; k++) // any nodes k between i and j with negative covariance to j?
	    if(cov[j][k]<0  || (j==natoms-1)) // end of positive-covariance area
	      {
		if((domainSize>=cut) && (j>lastNode)) // new positive-covariance segment which is large enough and not just subset of already found segment
		  {cout << "Segment "<< i+1 << "-" << j << " size " << domainSize << endl;
		    ndomains++;
		    for(l=i; l<j; l++)
		      inPosCovSeg[l]=1; // node i is part of a positive-covariance segment
		  
		    if(i>=lastNodeSeg) // count non-overlapping, separate positive covariance segments
		      {nSepDomains++;
			lastNodeSeg=j;
		      }
		  }
		domainSize=0;
		lastNode=j;
		goOn=false;
		break;
	      }
	  if(goOn==true)
	    domainSize++; // all covariances between i and j are positive, go to next k
	  else
	    break; // go to next node i
	}
    }

  int inNoDomain=0; // number of nodes which belong to no positive-covariance segment
  
  for(i=0; i<natoms; i++)
    if(inPosCovSeg[i]==0)
      inNoDomain++;
  
  
  cout << "nodes: " <<  natoms << " in no segment: " << inNoDomain  << " segmentNo: " << ndomains << " separate segments: " << nSepDomains << endl;

  return 0;
}

int readCov(char *filename, vector<vector<float> > &cov)
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
      if (sscanf(line,"%d %d %f", &index1, &index2, &value)!=3)
        {cout << "Problem in readCov: wrong numbering in covariance file" << endl;
	  exit(0);}
      else if (index1==0 or index2==0)
	{cout << "Problem in readCov: wrong numbering in covariance file" << endl;
	  exit(0);}
      
      /* store all covariances of node i in vector p */
      if(index1==i and index2==j) //ATTENTION: covariance input file must have right order and start with 1!!!
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
  return 0;
}


int number(char *s1, int *pos)
{
  int n=0;
  while(s1[*pos]>='0' && s1[*pos]<='9')
    {n=10*n+(s1[*pos]-'0');
    (*pos)++;
    }
  return n;
}
