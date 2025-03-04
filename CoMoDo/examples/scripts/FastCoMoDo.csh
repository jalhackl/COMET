#!/bin/csh
#The covariance files of the example proteins are given.
#They are generated using a Gaussian Network Model with a covalent force-constant of 10 kcal/(mol A^2), non-covalent force constant of 5 kcal/(mol A^2) and a cutoff radius of 7 A.
#DomainClusterer stops clustering when the largest intercluster covariance is negative.
#The *.pqrm files can be loaded in VMD by selecting "Determine file type: PDB".

setenv CoMoDo $HOME/CoMoDoPlus/programs/CoMoDo # must be changed to your path!!!
set programPath = $CoMoDo/src
set examplePath = $CoMoDo/examples/data
set scriptPath = $CoMoDo/examples/scripts

# DomainTester setup
set PosCovSegMinSize = 40  # minimal size of positive-covariance segments

# calculate dynamic domains of all proteins
foreach prot (`cat names.dat`)
	cd $examplePath/$prot
	echo $prot
	rm -r structures_FastCoMoDo

	# run DomainTester: check if several dynamic domains exist
	$programPath/DomainTester $prot.cov $PosCovSegMinSize > $prot.posCovSeg
	grep nodes $prot.posCovSeg | awk '{printf("%e %d\n", $6/$2, $11);}' > percentNoSegment.dat
 
	# protein consists only of one domain if more than half of the nodes don't belong to a positive-covariance segment or if less than two non-overlapping positive-covariance segments were found
	set boolDomain = `cat percentNoSegment.dat | awk '{if($1 >= 0.5 || $2 < 2) i=1; else i=0; printf("%d\n", i)}'`
	# 1-domain protein
	if ( $boolDomain == 1) then
		echo "Protein consists of only 1 dynamic domain"
		
	# more than 1 domain
	else
		echo "Protein consists of several dynamic domains"

		# run DomainClusterer and stop when largest intercluster covariance is negative
		$programPath/DomainClusterer $prot.cov -c 0 -v > FastCoMoDo.out
		grep Final FastCoMoDo.out
		mkdir structures_FastCoMoDo
		# $prot.domains contains cluster assignment of all residues
		grep nodes FastCoMoDo.out > $prot.domains

		# create splitted pqrm files and store them in structures_FastCoMoDo
		$programPath/splitPQRM.pl $prot
		mv $prot.*.pqrm structures_FastCoMoDo
	endif	
	echo " "
	
end
