#!/usr/bin/perl
use strict;
use warnings;

my $protein=$ARGV[0];
my $countDomains=0;
my @domainAreasStart;
my @domainAreasStop;
my $countAreas;
my @domainSingles;
my $countSingles;
my @parts;
my @elements;
my @domains;
my $numberParts;
my $numberFragments;
my @lineShort;
my $subline;
my $lastNode;


open(my $in,  "<",  "$protein.domains") or die "Can't open domains !";
while (my $line = <$in>)      # assigns each line in turn to $_
{
    if($line =~ /^nodes/)
    {$countDomains += 1;
     $countAreas = 0;
     $countSingles = 0;
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
      {$domainSingles[$countSingles]=$domains[$i];
       $countSingles += 1;
      }
     }
     $lastNode=0;
     open(my $in2,  "<",  "$protein.pqrm") or die "Can't open pqrm !";
     open(my $out, ">", "$protein.$countDomains.pqrm");
     while (my $line = <$in2>)     # assigns each line in turn to $_
     {
	 if($line =~ /^ATOM/)
	 {
	     @parts = split /\s+/, $line;
	     for(my $i=0; $i < $countAreas; $i += 1)
	     {
		 if($parts[1] >= $domainAreasStart[$i] && $parts[1] <= $domainAreasStop[$i])
		 {$subline=substr($line,13);
		  printf $out "ATOM %6d  ", $lastNode+1;
		  printf $out "$subline";
		  $lastNode += 1;
		  last;
		 }
	     }
	     for(my $i=0; $i<$countSingles; $i +=1)
	     {
		 if($parts[1] == $domainSingles[$i])
		 {$subline=substr($line,13);
		  printf $out "ATOM %6d  ", $lastNode+1;
		  print $out "$subline";
		  $lastNode += 1;
		  last;
		 }
	     }
	 }
	 
	 elsif($line =~ /^HETATM/)
	 {
	     @parts = split /\s+/, $line;
	     for(my $i=0; $i < $countAreas; $i += 1)
	     {
		 if($parts[1] >= $domainAreasStart[$i] && $parts[1] <= $domainAreasStop[$i])
		 {$subline=substr($line,13);
		  printf $out "HETATM %4d  ", $lastNode+1;
		  printf $out "$subline";
		  $lastNode += 1;
		  last;
		 }
	     }
	     for(my $i=0; $i<$countSingles; $i +=1)
	     {
		 if($parts[1] == $domainSingles[$i])
		 {$subline=substr($line,13);
		  printf $out "HETATM %4d  ", $lastNode+1;
		  print $out "$subline";
		  $lastNode += 1;
		  last;
		 }
	     }
	 }


	 elsif($line =~ /^(TER)/)
	 {print $out $line;
	 $lastNode=0;}
	 
	 elsif($line =~ /^(CONECT)/)
	 {
	     @parts = split /\s+/, $line;
	     $numberParts=@parts;
	     
	     for(my $i=0; $i < $countAreas; $i += 1)
	     {
		 if($parts[1] >= $domainAreasStart[$i] && $parts[1] <= $domainAreasStop[$i])
		 {printf $out "CONECT%5d",$lastNode+1;
		  for(my $j=2; $j<$numberParts; $j += 1)
		  {if($parts[$j] >= $domainAreasStart[$i] && $parts[$j] <= $domainAreasStop[$i])
		   {printf $out "%5d",$parts[$j]-$parts[1]+$lastNode+1;}
		  }
		  $lastNode += 1;
		  printf $out "\n";
		  $i += 1;
		  last;
		 }
	     }
	     for(my $i=0; $i < $countSingles; $i += 1)
	     {
		 if($parts[1] == $domainSingles[$i])
		 {printf $out "CONECT%5d\n",$lastNode+1;
		  $lastNode += 1;
		 }
	     }
	 }
     }
     print $out "END\n" ;
     close($in2); 
     close($out);
    }    
    $lastNode=0;
}
close($in);	
    

