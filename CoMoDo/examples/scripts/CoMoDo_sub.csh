#!/bin/csh
set prot = $1
set PosCovSegMinSize = $2
set iteration = $3
set programPath = $CoMoDo/src
set examplePath = $CoMoDo/examples/data
set scriptPath = $CoMoDo/examples/scripts
echo "iteration: $iteration"

# run TEN
# calculate eigenvalues and -vectors
$TEN/EValVecC/EValVecC > out
# calculate covariance matrix
$TEN/NWAnalyzer/NWAnalyzer > out

# run DomainTester: check if several dynamic domains exist
$programPath/DomainTester $prot\_covsGNM.out $PosCovSegMinSize > $prot.posCovSeg
grep nodes $prot.posCovSeg | awk '{printf("%e %d\n", $6/$2, $11);}' > percentNoSegment.dat 

# protein consists only of one domain if more than half of the nodes don't belong to a positive-covariance segment or if less than two non-overlapping positive-covariance segments were found
set boolDomain = `cat percentNoSegment.dat | awk '{if($1 >= 0.5 || $2 < 2) i=1; else i=0; printf("%d\n", i)}'`

# 1-domain protein
if ( $boolDomain == 1) then
	echo "Cluster is only 1 dynamic domain"
	echo "$iteration" >> $examplePath/$prot/domains.dat
	# store pqrm files in directory structures_CoMoDo and quit
	if ($iteration =~ *\_*\_*\_*) then
		mv $prot.pqrm ../../../../structures_CoMoDo/$prot.$iteration.pqrm
	else if ($iteration =~ *\_*\_*) then
		mv $prot.pqrm ../../../structures_CoMoDo/$prot.$iteration.pqrm
	else if ($iteration =~ *\_*) then
		mv $prot.pqrm ../../structures_CoMoDo/$prot.$iteration.pqrm
	else
		mv $prot.pqrm ../structures_CoMoDo/$prot.$iteration.pqrm
	endif

# more than 1 domain
else
	echo "Cluster consists of several dynamic domains"

	# run DomainClusterer and create two clusters
	$programPath/DomainClusterer $prot\_covsGNM.out -d 2 > CoMoDo.out
	# $prot.domains contains cluster assignment of all residues
	grep nodes CoMoDo.out > $prot.domains

	# create subdirectory for each cluster and copy the CoMoDo script and the settings file for TEN
	mkdir Dom1
	mkdir Dom2
	cp $scriptPath/CoMoDo_sub.csh Dom1
 	cp $scriptPath/CoMoDo_sub.csh Dom2
	cp settings Dom1
	cp settings Dom2

	# create and copy splitted pqrm files to subdirectories
	$programPath/splitPQRM.pl $prot
	mv $prot.1.pqrm Dom1/$prot.pqrm
	mv $prot.2.pqrm Dom2/$prot.pqrm

	# repeat procedure for each cluster using the script CoMoDo_sub.csh
	echo " "
	echo "Domain 1:"
	cd Dom1
	csh CoMoDo_sub.csh $prot $PosCovSegMinSize $iteration\_1 # last number keeps track of iterations
	echo " "
	echo "Domain 2:"
	cd ../Dom2
	csh CoMoDo_sub.csh $prot $PosCovSegMinSize $iteration\_2
	cd ..
endif	
