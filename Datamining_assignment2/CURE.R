orgData <- read.csv("ai2013_papers.csv")
y<-orgData[,c("type")]
x<-orgData[,c(2,3,4,5,6,7,8,9,10,11,12)]

library(rPython)
python.exec("from pyclustering.cluster.cure import cure;")
xx<-python.call("cure",x,8)