library(rPython)
orgData <- read.csv("Clustering/ai2013_papers.csv")
y<-orgData[,c("type")]
x<-orgData[,c(2,3,4,5,6,7,8,9,10,11,12)]

normalize <- function(x){
  return((x-min(x))/(max(x)-min(x)))
}

x <- as.data.frame(lapply(x, normalize))

write.csv(x, "Clustering/samples.csv",row.names = FALSE)
file_name = "Clustering/CURE_R_Python.py"
python.load(file_name)

result <- python.get("result")
head(result)


