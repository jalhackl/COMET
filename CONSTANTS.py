# Path for Ubuntu VM
#path_prefix = '/home/kloss/data/'
#path_prefix = '/Users/steffi/Documents/LMU/Master/6_FS/MA/data/'
path_prefix = 'trajdata/'


tmp_prefix = path_prefix[0:(len(path_prefix)-5)] + 'Code/tmp/'
pickle_prefix = path_prefix + 'pickle/'
test_prefix = path_prefix[0:(len(path_prefix)-5)] + 'test/data/'
geostas_prefix = path_prefix[0:(len(path_prefix)-5)] + 'geostas/'

 #[os.path.join(protease_path, "prod_r1_nojump_prot.xtc"),os.path.join(protease_path, "prod_r1_pbc_fit_prot_last.pdb"),'',-1,4,None,'protease_dcd']
prodr1 = ['prod_r1_nojump_prot.xtc','prod_r1_pbc_fit_prot_last.gro','',40001,16,'prod_r1_geostas_16k.csv','TLL']
prodr2 = ['prod_r2_nojump_prot.xtc','prod_r1_pbc_fit_prot_last.gro','',40001,16,'prod_r2_geostas_16k.csv','TLL']
prodr3 = ['prod_r3_nojump_prot.xtc','prod_r1_pbc_fit_prot_last.gro','',40001,16,'prod_r3_geostas_16k.csv','TLL']

#ergaenzung des path
prodr1 = ['prod_r1_nojump_prot.xtc','prod_r1_pbc_fit_prot_last.gro','ProtNo2/',40001,16,'prod_r1_geostas_16k.csv','TLL']
prodr2 = ['prod_r2_nojump_prot.xtc','prod_r1_pbc_fit_prot_last.gro','ProtNo2/',40001,16,'prod_r2_geostas_16k.csv','TLL']
prodr3 = ['prod_r3_nojump_prot.xtc','prod_r1_pbc_fit_prot_last.gro','ProtNo2/',40001,16,'prod_r3_geostas_16k.csv','TLL']


sav1 = ['savinase_1.xtc','savinase.pdb','savinase/',1481,16,'savinase_1_geostas_16k.csv','Savinase']
sav2 = ['savinase_2.xtc','savinase.pdb','savinase/',1481,16,'savinase_2_geostas_16k.csv','Savinase']
mcg1 = ['trajectory-1.xtc','fs-peptide.pdb','McGibbon/',10000,4,'mcgib_traj1_geostas_4k.csv','Ace']
mcg2 = ['trajectory-2.xtc','fs-peptide.pdb','McGibbon/',10000,4,'mcgib_traj2_geostas_4k.csv','Ace']
mcg3 = ['trajectory-3.xtc','fs-peptide.pdb','McGibbon/',10000,4,'mcgib_traj3_geostas_4k.csv','Ace']
mcg4 = ['trajectory-4.xtc','fs-peptide.pdb','McGibbon/',10000,4,'mcgib_traj4_geostas_4k.csv','Ace']
mcg5 = ['trajectory-5.xtc','fs-peptide.pdb','McGibbon/',10000,4,'mcgib_traj5_geostas_4k.csv','Ace']
mcg6 = ['trajectory-6.xtc','fs-peptide.pdb','McGibbon/',10000,4,'mcgib_traj6_geostas_4k.csv','Ace']
mcg7 = ['trajectory-7.xtc','fs-peptide.pdb','McGibbon/',10000,4,'mcgib_traj7_geostas_4k.csv','Ace']
mcg8 = ['trajectory-8.xtc','fs-peptide.pdb','McGibbon/',10000,4,'mcgib_traj8_geostas_4k.csv','Ace']
mcg9 = ['trajectory-9.xtc','fs-peptide.pdb','McGibbon/',10000,4,'mcgib_traj9_geostas_4k.csv','Ace']
mcg10 = ['trajectory-10.xtc','fs-peptide.pdb','McGibbon/',10000,4,'mcgib_traj10_geostas_4k.csv','Ace']
mcg11 = ['trajectory-11.xtc','fs-peptide.pdb','McGibbon/',10000,4,'mcgib_traj11_geostas_4k.csv','Ace']
mcg12 = ['trajectory-12.xtc','fs-peptide.pdb','McGibbon/',10000,4,'mcgib_traj12_geostas_4k.csv','Ace']
mcg13 = ['trajectory-13.xtc','fs-peptide.pdb','McGibbon/',10000,4,'mcgib_traj13_geostas_4k.csv','Ace']
mcg14 = ['trajectory-14.xtc','fs-peptide.pdb','McGibbon/',10000,4,'mcgib_traj14_geostas_4k.csv','Ace']
mcg15 = ['trajectory-15.xtc','fs-peptide.pdb','McGibbon/',10000,4,'mcgib_traj15_geostas_4k.csv','Ace']
mcg16 = ['trajectory-16.xtc','fs-peptide.pdb','McGibbon/',10000,4,'mcgib_traj16_geostas_4k.csv','Ace']
mcg17 = ['trajectory-17.xtc','fs-peptide.pdb','McGibbon/',10000,4,'mcgib_traj17_geostas_4k.csv','Ace']
mcg18 = ['trajectory-18.xtc','fs-peptide.pdb','McGibbon/',10000,4,'mcgib_traj18_geostas_4k.csv','Ace']
mcg19 = ['trajectory-19.xtc','fs-peptide.pdb','McGibbon/',10000,4,'mcgib_traj19_geostas_4k.csv','Ace']
mcg20 = ['trajectory-20.xtc','fs-peptide.pdb','McGibbon/',10000,4,'mcgib_traj20_geostas_4k.csv','Ace']
mcg21 = ['trajectory-21.xtc','fs-peptide.pdb','McGibbon/',10000,4,'mcgib_traj21_geostas_4k.csv','Ace']
mcg22 = ['trajectory-22.xtc','fs-peptide.pdb','McGibbon/',10000,4,'mcgib_traj22_geostas_4k.csv','Ace']
mcg23 = ['trajectory-23.xtc','fs-peptide.pdb','McGibbon/',10000,4,'mcgib_traj23_geostas_4k.csv','Ace']
mcg24 = ['trajectory-24.xtc','fs-peptide.pdb','McGibbon/',10000,4,'mcgib_traj24_geostas_4k.csv','Ace']
mcg25 = ['trajectory-25.xtc','fs-peptide.pdb','McGibbon/',10000,4,'mcgib_traj25_geostas_4k.csv','Ace']
mcg26 = ['trajectory-26.xtc','fs-peptide.pdb','McGibbon/',10000,4,'mcgib_traj26_geostas_4k.csv','Ace']
mcg27 = ['trajectory-27.xtc','fs-peptide.pdb','McGibbon/',10000,4,'mcgib_traj27_geostas_4k.csv','Ace']
mcg28 = ['trajectory-28.xtc','fs-peptide.pdb','McGibbon/',10000,4,'mcgib_traj28_geostas_4k.csv','Ace']
H140 = ['abeta_1_40.dcd','abeta_1_40.pdb','Harvard/',100000,6,'abeta_1_40_geostas_6k.csv','Abeta']
H22G = ['abeta_E22G_1_40.dcd','abeta_E22G_1_40.pdb','Harvard/',100000,6,'abeta_E22G_1_40_geostas_6k.csv','Abeta']
tr0 = ['tr0.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr1 = ['tr1.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr2 = ['tr2.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr3 = ['tr3_unfolded.xtc','2f4k.gro','2F4K/',10000,5,'tr3_unfolded_geostas_5k.csv','2F4K']
tr4 = ['tr4.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr5 = ['tr5.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr6 = ['tr6.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr7 = ['tr7.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr8 = ['tr8_folded.xtc','2f4k.gro','2F4K/',10000,5,'tr8_folded_geostas_5k.csv','2F4K']
tr9 = ['tr9.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr10 = ['tr10.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr11 = ['tr11.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr12 = ['tr12.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr13 = ['tr13.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr14 = ['tr14.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr15 = ['tr15.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr16 = ['tr16.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr17 = ['tr17.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr18 = ['tr18.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr19 = ['tr19.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr20 = ['tr20.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr21 = ['tr21.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr22 = ['tr22.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr23 = ['tr23.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr24 = ['tr24.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr25 = ['tr25.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr26 = ['tr26.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr27 = ['tr27.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr28 = ['tr28.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr29 = ['tr29.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr30 = ['tr30.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr31 = ['tr31.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr32 = ['tr32.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr33 = ['tr33.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr34 = ['tr34.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr35 = ['tr35.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr36 = ['tr36.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr37 = ['tr37.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr38 = ['tr38.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr39 = ['tr39.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr40 = ['tr40.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr41 = ['tr41.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr42 = ['tr42.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr43 = ['tr43.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr44 = ['tr44.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr45 = ['tr45.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr46 = ['tr46.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr47 = ['tr47.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr48 = ['tr48.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr49 = ['tr49.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr50 = ['tr50.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr51 = ['tr51.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr52 = ['tr52.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr53 = ['tr53.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr54 = ['tr54.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr55 = ['tr55.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr56 = ['tr56.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr57 = ['tr57.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr58 = ['tr58.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr59 = ['tr59.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr60 = ['tr60.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr61 = ['tr61.dcd','2f4k.gro','2F4K/',10000,5,None,'2F4K']
tr62 = ['tr62.dcd','2f4k.gro','2F4K/',7907,5,None,'2F4K']


abeta = [H140,H22G]
sav = [sav1,sav2]
r1 = [prodr1,prodr2,prodr3]
#r1 == TLL !!!


tr = [tr0,tr1,tr2,tr3,tr4,tr5,tr6,tr7,tr8,tr9,tr10,tr11,tr12,tr13,tr14,tr15,tr16,tr17,tr18,tr19,tr20,tr21,tr22,tr23,tr24,tr25,tr26,tr27,tr28,tr29,tr30,tr31,tr32,tr33,tr34,tr35,tr36,tr37,tr38,tr39,tr40,tr41,tr42,tr43,tr44,tr45,tr46,tr47,tr48,tr49,tr50,tr51,tr52,tr53,tr54,tr55,tr56,tr57,tr58,tr59,tr60,tr61]
mcg = [mcg1,mcg2,mcg3,mcg4,mcg5,mcg6,mcg7,mcg8,mcg9,mcg10,mcg11,mcg12,mcg13,mcg14,mcg15,mcg16,mcg17,mcg18,mcg19,mcg20,mcg21,mcg22,mcg23,mcg24,mcg25,mcg26,mcg27,mcg28]


chig1 = ['nvt_prod_1.xtc','0new.pdb','Chignolin/both/',10000,4,'mcgib_traj1_geostas_4k.csv','Chignolin']
chig2 = ['nvt_prod_2.xtc','0new.pdb','Chignolin/both/',10000,4,'mcgib_traj2_geostas_4k.csv','Chignolin']
chig3 = ['nvt_prod_3.xtc','0new.pdb','Chignolin/both/',10000,4,'mcgib_traj3_geostas_4k.csv','Chignolin']
chig4 = ['nvt_prod_4.xtc','0new.pdb','Chignolin/both/',10000,4,'mcgib_traj4_geostas_4k.csv','Chignolin']
chig5 = ['nvt_prod_5.xtc','0new.pdb','Chignolin/both/',10000,4,'mcgib_traj5_geostas_4k.csv','Chignolin']
chig = [chig1, chig2, chig3, chig4, chig5]

forsch1 = ["5m3avmd.pdb",None,'',10000,4,'mcgib_traj1_geostas_4k.csv','5m3a']
forsch = [forsch1]


forsch1_11 = ["5m3avmd.pdb",None,'',10000,11,'mcgib_traj1_geostas_4k.csv','5m3a']




forsch_11 = [forsch1_11]




a3d0 = ['A3D-0-c-alpha-000.dcd','alpha3_top.pdb','shaw/',-1,5,None,'a3d']
a3d1 = ['A3D-0-c-alpha-001.dcd','alpha3_top.pdb','shaw/',-1,5,None,'a3d']
a3d2 = ['A3D-0-c-alpha-002.dcd','alpha3_top.pdb','shaw/',-1,5,None,'a3d']
a3d3 = ['A3D-0-c-alpha-003.dcd','alpha3_top.pdb','shaw/',-1,5,None,'a3d']
a3d4 = ['A3D-0-c-alpha-004.dcd','alpha3_top.pdb','shaw/',-1,5,None,'a3d']
a3d5 = ['A3D-0-c-alpha-005.dcd','alpha3_top.pdb','shaw/',-1,5,None,'a3d']
a3d6 = ['A3D-0-c-alpha-006.dcd','alpha3_top.pdb','shaw/',-1,5,None,'a3d']
a3d7 = ['A3D-0-c-alpha-007.dcd','alpha3_top.pdb','shaw/',-1,5,None,'a3d']
a3d8 = ['A3D-0-c-alpha-008.dcd','alpha3_top.pdb','shaw/',-1,5,None,'a3d']
a3d9 = ['A3D-0-c-alpha-009.dcd','alpha3_top.pdb','shaw/',-1,5,None,'a3d']
a3d10 = ['A3D-0-c-alpha-010.dcd','alpha3_top.pdb','shaw/',-1,5,None,'a3d']
a3d11 = ['A3D-0-c-alpha-011.dcd','alpha3_top.pdb','shaw/',-1,5,None,'a3d']
a3d12 = ['A3D-0-c-alpha-012.dcd','alpha3_top.pdb','shaw/',-1,5,None,'a3d']
a3d13 = ['A3D-0-c-alpha-013.dcd','alpha3_top.pdb','shaw/',-1,5,None,'a3d']
a3d14 = ['A3D-0-c-alpha-014.dcd','alpha3_top.pdb','shaw/',-1,5,None,'a3d']
a3d15 = ['A3D-0-c-alpha-015.dcd','alpha3_top.pdb','shaw/',-1,5,None,'a3d']
a3d16 = ['A3D-0-c-alpha-0016.dcd','alpha3_top.pdb','shaw/',-1,5,None,'a3d']

a3ds = [a3d0, a3d1, a3d2, a3d3, a3d4, a3d5, a3d6, a3d7, a3d8, a3d9, a3d10, a3d11, a3d12, a3d13, a3d14, a3d15, a3d16]



a3d0_noc = ['A3D-0-c-alpha-000.dcd','alpha3_top.pdb','shaw/',-1,None,None,'a3d']
a3d1_noc = ['A3D-0-c-alpha-001.dcd','alpha3_top.pdb','shaw/',-1,None,None,'a3d']
a3d2_noc = ['A3D-0-c-alpha-002.dcd','alpha3_top.pdb','shaw/',-1,None,None,'a3d']
a3d3_noc = ['A3D-0-c-alpha-003.dcd','alpha3_top.pdb','shaw/',-1,None,None,'a3d']
a3d4_noc = ['A3D-0-c-alpha-004.dcd','alpha3_top.pdb','shaw/',-1,None,None,'a3d']
a3d5_noc = ['A3D-0-c-alpha-005.dcd','alpha3_top.pdb','shaw/',-1,None,None,'a3d']
a3d6_noc = ['A3D-0-c-alpha-006.dcd','alpha3_top.pdb','shaw/',-1,None,None,'a3d']
a3d7_noc = ['A3D-0-c-alpha-007.dcd','alpha3_top.pdb','shaw/',-1,None,None,'a3d']
a3d8_noc = ['A3D-0-c-alpha-008.dcd','alpha3_top.pdb','shaw/',-1,None,None,'a3d']
a3d9_noc = ['A3D-0-c-alpha-009.dcd','alpha3_top.pdb','shaw/',-1,None,None,'a3d']
a3d10_noc = ['A3D-0-c-alpha-010.dcd','alpha3_top.pdb','shaw/',-1,None,None,'a3d']
a3d11_noc = ['A3D-0-c-alpha-011.dcd','alpha3_top.pdb','shaw/',-1,None,None,'a3d']
a3d12_noc = ['A3D-0-c-alpha-012.dcd','alpha3_top.pdb','shaw/',-1,None,None,'a3d']
a3d13_noc = ['A3D-0-c-alpha-013.dcd','alpha3_top.pdb','shaw/',-1,None,None,'a3d']
a3d14_noc = ['A3D-0-c-alpha-014.dcd','alpha3_top.pdb','shaw/',-1,None,None,'a3d']
a3d15_noc = ['A3D-0-c-alpha-015.dcd','alpha3_top.pdb','shaw/',-1,None,None,'a3d']
#a3d16_noc = ['A3D-0-c-alpha-0016.dcd','alpha3_top.pdb','shaw/',-1,None,None,'a3d']

a3ds_noc = [a3d0_noc, a3d1_noc, a3d2_noc, a3d3_noc, a3d4_noc, a3d5_noc, a3d6_noc, a3d7_noc, a3d8_noc, a3d9_noc, a3d10_noc, a3d11_noc, a3d12_noc, a3d13_noc, a3d14_noc, a3d15_noc]



lambda0 = ['lambda-0-c-alpha-000.dcd','lambda_top.pdb','shaw/',-1,5,None,'lambda']
lambda1 = ['lambda-0-c-alpha-001.dcd','lambda_top.pdb','shaw/',-1,5,None,'lambda']
lambda2 = ['lambda-0-c-alpha-002.dcd','lambda_top.pdb','shaw/',-1,5,None,'lambda']
lambda3 = ['lambda-0-c-alpha-003.dcd','lambda_top.pdb','shaw/',-1,5,None,'lambda']
lambda4 = ['lambda-0-c-alpha-004.dcd','lambda_top.pdb','shaw/',-1,5,None,'lambda']
lambda5 = ['lambda-0-c-alpha-005.dcd','lambda_top.pdb','shaw/',-1,5,None,'lambda']
lambda6 = ['lambda-0-c-alpha-006.dcd','lambda_top.pdb','shaw/',-1,5,None,'lambda']
lambda7 = ['lambda-0-c-alpha-007.dcd','lambda_top.pdb','shaw/',-1,5,None,'lambda']

lambdas = [lambda0, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7]

lambda0_noc = ['lambda-0-c-alpha-000.dcd','lambda_top.pdb','shaw/',-1,None,None,'lambda']
lambda1_noc = ['lambda-0-c-alpha-001.dcd','lambda_top.pdb','shaw/',-1,None,None,'lambda']
lambda2_noc = ['lambda-0-c-alpha-002.dcd','lambda_top.pdb','shaw/',-1,None,None,'lambda']
lambda3_noc = ['lambda-0-c-alpha-003.dcd','lambda_top.pdb','shaw/',-1,None,None,'lambda']
lambda4_noc = ['lambda-0-c-alpha-004.dcd','lambda_top.pdb','shaw/',-1,None,None,'lambda']
lambda5_noc = ['lambda-0-c-alpha-005.dcd','lambda_top.pdb','shaw/',-1,None,None,'lambda']
lambda6_noc = ['lambda-0-c-alpha-006.dcd','lambda_top.pdb','shaw/',-1,None,None,'lambda']
lambda7_noc = ['lambda-0-c-alpha-007.dcd','lambda_top.pdb','shaw/',-1,None,None,'lambda']

lambdas_noc = [lambda0_noc, lambda1_noc, lambda2_noc, lambda3_noc, lambda4_noc, lambda5_noc, lambda6_noc, lambda7_noc]



p2f4k_0 = ['2F4K-0-protein-000.dcd','2f4k.pdb','2f4k/',-1,None,None,'2f4k']
p2f4k_1 = ['2F4K-0-protein-001.dcd','2f4k.pdb','2f4k/',-1,None,None,'2f4k']
p2f4k_2 = ['2F4K-0-protein-002.dcd','2f4k.pdb','2f4k/',-1,None,None,'2f4k']
p2f4k_3 = ['2F4K-0-protein-003.dcd','2f4k.pdb','2f4k/',-1,None,None,'2f4k']
p2f4k_4 = ['2F4K-0-protein-004.dcd','2f4k.pdb','2f4k/',-1,None,None,'2f4k']
p2f4k_5 = ['2F4K-0-protein-005.dcd','2f4k.pdb','2f4k/',-1,None,None,'2f4k']
p2f4k_6 = ['2F4K-0-protein-006.dcd','2f4k.pdb','2f4k/',-1,None,None,'2f4k']
p2f4k_7 = ['2F4K-0-protein-007.dcd','2f4k.pdb','2f4k/',-1,None,None,'2f4k']
p2f4k_8 = ['2F4K-0-protein-008.dcd','2f4k.pdb','2f4k/',-1,None,None,'2f4k']
p2f4k_9 = ['2F4K-0-protein-009.dcd','2f4k.pdb','2f4k/',-1,None,None,'2f4k']
p2f4k_10 = ['2F4K-0-protein-010.dcd','2f4k.pdb','2f4k/',-1,None,None,'2f4k']
p2f4k_11 = ['2F4K-0-protein-011.dcd','2f4k.pdb','2f4k/',-1,None,None,'2f4k']
p2f4k_12 = ['2F4K-0-protein-012.dcd','2f4k.pdb','2f4k/',-1,None,None,'2f4k']
p2f4k_13 = ['2F4K-0-protein-013.dcd','2f4k.pdb','2f4k/',-1,None,None,'2f4k']
p2f4k_14 = ['2F4K-0-protein-014.dcd','2f4k.pdb','2f4k/',-1,None,None,'2f4k']
p2f4k_15 = ['2F4K-0-protein-015.dcd','2f4k.pdb','2f4k/',-1,None,None,'2f4k']
p2f4k_16 = ['2F4K-0-protein-016.dcd','2f4k.pdb','2f4k/',-1,None,None,'2f4k']
p2f4k_17 = ['2F4K-0-protein-017.dcd','2f4k.pdb','2f4k/',-1,None,None,'2f4k']
p2f4k_18 = ['2F4K-0-protein-018.dcd','2f4k.pdb','2f4k/',-1,None,None,'2f4k']
p2f4k_19 = ['2F4K-0-protein-019.dcd','2f4k.pdb','2f4k/',-1,None,None,'2f4k']
p2f4k_20 = ['2F4K-0-protein-020.dcd','2f4k.pdb','2f4k/',-1,None,None,'2f4k']
p2f4k_21 = ['2F4K-0-protein-021.dcd','2f4k.pdb','2f4k/',-1,None,None,'2f4k']
p2f4k_22 = ['2F4K-0-protein-022.dcd','2f4k.pdb','2f4k/',-1,None,None,'2f4k']
p2f4k_23 = ['2F4K-0-protein-023.dcd','2f4k.pdb','2f4k/',-1,None,None,'2f4k']
p2f4k_24 = ['2F4K-0-protein-024.dcd','2f4k.pdb','2f4k/',-1,None,None,'2f4k']
p2f4k_25 = ['2F4K-0-protein-025.dcd','2f4k.pdb','2f4k/',-1,None,None,'2f4k']
p2f4k_26 = ['2F4K-0-protein-026.dcd','2f4k.pdb','2f4k/',-1,None,None,'2f4k']


p2f4ks = [p2f4k_0, p2f4k_1,p2f4k_2,p2f4k_3,p2f4k_4,p2f4k_5,p2f4k_6,p2f4k_7, 
p2f4k_8 
,p2f4k_9 
,p2f4k_10 
,p2f4k_11 
,p2f4k_12 
,p2f4k_13 
,p2f4k_14 
,p2f4k_15 
,p2f4k_16 
,p2f4k_17 
,p2f4k_18 
,p2f4k_19 
,p2f4k_20
,p2f4k_21
,p2f4k_22
,p2f4k_23
,p2f4k_24
,p2f4k_25
,p2f4k_26]