#!/usr/bin/perl
use strict;
use warnings;

my $protein=$ARGV[0];
my @values;
my $sum=0;
my $number=0;
my $nodes;
my $mean;
my @covs;
my $count=0;

open(my $in, "<", "$protein.cov") or die "Can't open cov!";
while (my $line = <$in>) 
{
    @values = split /\s/, $line;
    $sum=$sum+$values[2];
    $covs[$number]=$values[2];
    $number += 1;
}

$mean = $sum/$number;
$nodes = sqrt($number);

open(my $out, ">", "$protein.scaled.cov");

for (my $i=1; $i<=$nodes; $i += 1) 
{for (my $j=1; $j<=$nodes; $j += 1)
 {
     printf $out "%d %d %e\n", $i, $j, $covs[$count]-$mean;
     $count += 1;
 }
}

close($in);
close($out);

