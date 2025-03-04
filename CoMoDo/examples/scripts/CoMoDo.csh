#!/bin/csh
#The covariance files are calculated by TEN (Tools for Elastic Networks).
#They are generated using a Gaussian Network Model with a covalent force-constant of 10 kcal/(mol A^2), non-covalent force constant of 5 kcal/(mol A^2) and a cutoff radius of 7 A.
#If no covariance matrices of the single dynamic domains can be generated, one can alternatively use FastCoMoDo (script FastCoMoDo.csh) or rescale the splitted covariance matrices instead of recalculating them (script NormCoMoDo.csh).
#The *.pqrm files can be loaded in VMD by selecting "Determine file type: PDB".

setenv CoMoDo $HOME/CoMoDoPlus/programs/CoMoDo # must be changed to your path!!!
setenv TEN $HOME/programs/TEN/test/TEN # must be changed to your path!!!
set programPath = $CoMoDo/src
set examplePath = $CoMoDo/examples/data
set scriptPath = $CoMoDo/examples/scripts

# DomainTester setup
set PosCovSegMinSize = 40  # minimal size of positive-covariance segments

# TEN setup (Elastic Network Model)
set kcov = 10 # covalent force constant
set kncov = 5 # non-covalent force constant
set rcut = 7 # cutoff distance

# calculate dynamic domains of all proteins
foreach prot (`cat names.dat`)
	cd $examplePath/$prot
	echo "============================================"
	echo $prot
	echo "============================================"
	rm -r structures_CoMoDo
	rm -r Dom1
	rm -r Dom2

	# prepare setup file for TEN
	echo "names $prot" > settings
	echo "kcovG $kcov" >> settings
	echo "kncovG $kncov" >> settings
	echo "rcutG $rcut" >> settings
	echo "models GNM" >> settings
	echo "bvalAlphaOnly 0" >> settings

	# run TEN
	# calculate eigenvalues and -vectors
	$TEN/EValVecC/EValVecC > out
	# calculate covariance matrix
	$TEN/NWAnalyzer/NWAnalyzer > out
	
	# run DomainTester: check if several dynamic domains exist
	$programPath/DomainTester $prot\_covsGNM.out $PosCovSegMinSize > $prot.posCovSeg
	grep nodes $prot.posCovSeg | awk '{printf("%e %d\n", $6/$2, $11);}' > percentNoSegment.dat 
	
	# protein consists only of one domain if more than half of the nodes don't belong to a positive-covariance segment or if less than two non-overlapping positive-covariance segments were found
	set boolDomain = `cat percentNoSegment.dat | awk '{if($1 >= 50 || $2 < 2) i=1; else i=0; printf("%d\n", i)}'`

	# 1-domain protein
	if ( $boolDomain == 1) then
		echo "Protein consists of only 1 dynamic domain"
		echo "1" > domains.dat
	
	# more than 1 domain
	else
		echo "Protein consists of several dynamic domains"
		rm domains.dat
		mkdir structures_CoMoDo # pqrm files for all domains will be copied to structures_CoMoDo
		
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
		
		csh CoMoDo_sub.csh $prot $PosCovSegMinSize 1 # last number keeps track of iterations
		echo " "
		echo "Domain 2:"
		cd ../Dom2
		csh CoMoDo_sub.csh $prot $PosCovSegMinSize 2
		cd ..
	endif	
	echo " "
	# determine number of dynamic domains
	wc -l domains.dat | awk '{printf("Dynamic domains: %d\n", $1)}'
	echo " "
	
end
