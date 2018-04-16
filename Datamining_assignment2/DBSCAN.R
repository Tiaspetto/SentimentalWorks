libaray(fpc)
orgData <- read.csv("ai2013_papers.csv")
y<-orgData[,c("type")]
x<-orgData[,c(2,3,4,5,6,7,8,9,10,11,12)]

normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}

x<-as.data.frame(lapply(x, normalize))

ds_xx <- dbscan(x,1,10)
summary(x)
summary(as.factor(ds_xx$cluster))