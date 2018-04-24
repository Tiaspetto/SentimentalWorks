library(fpc)
library(dbscan)
library(factoextra)
library(cluster)
orgData <- read.csv("Clustering/ai2013_papers.csv")
y<-orgData[,c("type")]
x<-orgData[,c(2,3,4,5,6,7,8,9,10,11,12)]

normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}

x<-as.data.frame(lapply(x, normalize))
x<-as.matrix(x)

#1 Find optimal eps for original space 
dbscan::kNNdistplot(x, k =  5)
abline(h = 0.25, lty = 2)

#2 See result of DBSCAN
set.seed(123)
ds_xx <- fpc::dbscan(x,0.25, 5) 
summary(as.factor(ds_xx$cluster))
print(ds_xx)
fviz_cluster(ds_xx, x, geom = "point")

#3 Perform PCA on data space, find best PC dimensions
pc <- prcomp(x)
plot(pc, type='l')

#4 See data on principle componets space 
comp <- data.frame(pc$x[,1:3])
plot(comp, pch=16, col=rgb(0,0,0,0.5))

dbscan::kNNdistplot(comp, k =  2)
abline(h = 0.08, lty = 2)

ds_xx <- fpc::dbscan(comp,0.08, 2) 
summary(as.factor(ds_xx$cluster))
print(ds_xx)
fviz_cluster(ds_xx, comp, geom = "point")
