#!/bin/csh
set prot = $1
set PosCovSegMinSize = $2
set iteration = $3
set programPath = $CoMoDo/src
set examplePath = $CoMoDo/examples/data
set scriptPath = $CoMoDo/examples/scripts
echo "iteration: $iteration"

# run DomainTester: check if several dynamic domains exist
$programPath/DomainTester $prot.cov $PosCovSegMinSize > $prot.posCovSeg
grep nodes $prot.posCovSeg | awk '{printf("%e %d\n", $6/$2, $11);}' > percentNoSegment.dat 

# protein consists only of one domain if more than half of the nodes don't belong to a positive-covariance segment or if less than two non-overlapping positive-covariance segments were found
set boolDomain = `cat percentNoSegment.dat | awk '{if($1 >= 0.5 || $2 < 2) i=1; else i=0; printf("%d\n", i)}'`

# 1-domain protein
if ( $boolDomain == 1) then
	echo "Cluster is only 1 dynamic domain"
	echo "$iteration" >> $examplePath/$prot/domains.dat
	# store pqrm files in directory structures_NormCoMoDo and quit
	if ($iteration =~ *\_*\_*\_*) then
		mv $prot.pqrm ../../../../structures_NormCoMoDo/$prot.$iteration.pqrm
	else if ($iteration =~ *\_*\_*) then
		mv $prot.pqrm ../../../structures_NormCoMoDo/$prot.$iteration.pqrm
	else if ($iteration =~ *\_*) then
		mv $prot.pqrm ../../structures_NormCoMoDo/$prot.$iteration.pqrm
	else
		mv $prot.pqrm ../structures_NormCoMoDo/$prot.$iteration.pqrm
	endif

# more than 1 domain
else
	echo "Cluster consists of several dynamic domains"

	# run DomainClusterer and create two clusters
	$programPath/DomainClusterer $prot.cov -d 2 > NormCoMoDo.out
	# $prot.domains contains cluster assignment of all residues
	grep nodes NormCoMoDo.out > $prot.domains
	
	# create splitted pqrm files and splitted, renormalized covariance matrices
	$programPath/splitPQRM.pl $prot
	$programPath/splitCov.pl $prot
	$programPath/rescaleCov.pl $prot.1
	$programPath/rescaleCov.pl $prot.2

	# create subdirectory for each cluster and copy the CoMoDo script, covariance matrices and pqrm files
	mkdir Dom1
	mkdir Dom2
	cp $scriptPath/NormCoMoDo_sub.csh Dom1
	cp $scriptPath/NormCoMoDo_sub.csh Dom2
	mv $prot.1.scaled.cov Dom1/$prot.cov
	mv $prot.2.scaled.cov Dom2/$prot.cov
	mv $prot.1.pqrm Dom1/$prot.pqrm
	mv $prot.2.pqrm Dom2/$prot.pqrm

	# repeat procedure for each cluster using the script NormCoMoDo_sub.csh
	echo " "
	echo "Domain 1:"
	cd Dom1
	csh NormCoMoDo_sub.csh $prot $PosCovSegMinSize $iteration\_1 # last number keeps track of iterations
	echo " "
	echo "Domain 2:"
	cd ../Dom2
	csh NormCoMoDo_sub.csh $prot $PosCovSegMinSize $iteration\_2
	cd ..
endif	
