#!/usr/bin/perl
use strict;
use warnings;

my $protein=$ARGV[0];
my $countDomains=0;
my @domainAreasStart;
my @domainAreasStop;
my $countAreas;
my @parts;
my @elements;
my @domains;
my $numberParts;
my $numberFragments;
my @lineShort;
my $subline;
my $lastNode;
my @mapNodes;

open(my $in,  "<",  "$protein.domains") or die "Can't open domains !";
while (my $line = <$in>)      # assigns each line in turn to $_
{
    if($line =~ /^nodes/)
    {$countDomains += 1;
     $countAreas = 0;
     @lineShort = split /\s/, $line;
     @domains = split /,/, $lineShort[1];
     $numberFragments = @domains;
     print "Number of domain fragments: $numberFragments\n";
     for(my $i=0; $i<$numberFragments; $i += 1)
     {if($domains[$i]=~/-/)
      {@elements = split /-/, $domains[$i];
       $domainAreasStart[$countAreas]=$elements[0];
       $domainAreasStop[$countAreas]=$elements[1];
       $countAreas += 1;
      }
      else
      {$domainAreasStart[$countAreas]=$domains[$i];
       $domainAreasStop[$countAreas]=$domains[$i];
       $countAreas += 1;
      }
     }
     
     $lastNode=1;
     for(my $i=0; $i < $countAreas; $i += 1)
     {for(my $j=$domainAreasStart[$i]; $j <= $domainAreasStop[$i]; $j += 1)
      {$mapNodes[$j]=$lastNode;
       $lastNode += 1;
      }
     }	 

    
     open(my $in2,  "<",  "$protein.cov") or die "Can't open cov !";
     open(my $out, ">", "$protein.$countDomains.cov");
     while (my $line = <$in2>)     # assigns each line in turn to $_
     {
	 @parts = split /\s+/, $line;
	 for(my $i=0; $i < $countAreas; $i += 1)
	 {
	     if($parts[0] >= $domainAreasStart[$i] && $parts[0] <= $domainAreasStop[$i])
		 {for(my $j=0; $j < $countAreas; $j += 1)
		      {if($parts[1] >= $domainAreasStart[$j] && $parts[1] <= $domainAreasStop[$j])
		       {printf $out "%d %d ", $mapNodes[$parts[0]], $mapNodes[$parts[1]];
			printf $out "%e\n", $parts[2];
		       }
		      }
		 }
	 }
     }	 
	
     close($in2); 
     close($out);
    }    
   
}
close($in);	
    

