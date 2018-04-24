Code is programed under R version 3.4.3

DBSCAN: library required:
=============================================================================================================
	library(fpc)
	library(dbscan)
	library(factoextra)
	library(cluster)

	Run code: 
------------------------------------------------------------------------------------------------------------
	several key points need to be noticed, you can following the comments to see the result of each step.
	#1 Find optimal eps for original space 
	#2 See result of DBSCAN
	#3 Perform PCA on data space, find best PC dimensions
	#4 See data on principle componets space 
	#5 Find optimal eps for PCA data
	#6 See result of PCA DBSCAN

K-means: library required:
==============================================================================================================
	library(fpc)
	library(factoextra)
	
	Run code:
--------------------------------------------------------------------------------------------------------------
	#1 Find optimal K
	#2 See clustering in pair of dimensions
	#3 Plot clustering result K=8
	#4 Compared ground truth with clustered result
	#5 Run as optimal K
	#6 Plot clustering result k=2

Hierachical: library required:
==============================================================================================================
	library(colorspace)
	library(gplots)
	library(dendextend)
	library(factoextra)
	library(cluster)

	Run Code£º
--------------------------------------------------------------------------------------------------------------
	#1 Create cluster heat map and tree structure 
	#2 K=8 clustering result

Some pacakges may not avaible by install.package("") command in certain platform, please run install.package("dev_tools") before and using github_install("") command install directly from 
Git hub branch.


