#Read data from disk and 
orgData <- read.csv("Clustering/ai2013_papers.csv")
orgData <- orgData[sample(nrow(orgData)), ]
#Divided data into samples and labels
y<-orgData[,c("type")]
x<-orgData[,c(2,3,4,5,6,7,8,9,10,11,12)]

#View data
summary(x)

#nolmalize data
normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}

x<-as.data.frame(lapply(x, normalize))


#Euclidean distance function
Euclidean_distance <- function(sample1, sample2){
  sum_squared_distance <- 0
  for(i in (1:length(sample1))){
    sum_squared_distance = sum_squared_distance+(sample1[[i]] - sample2[[i]])^2
  }
  return(sum_squared_distance^(1/2))
}

#kmeans function
mykmeans <- function(k, data){
  random_init <- round(runif(1,min=1,max=nrow(data)))
  print(random_init)
  clusters_centroid <-matrix(0,k,ncol(data))
  clusters_centroid[1,] <-as.matrix(data[random_init, ])
  for(i in (2:k)){
    max_distance = 0 
    max_distance_index = 0
    for(j in (1:nrow(data))){
      if(!(j %in% clusters_centroid)){
        distance = Euclidean_distance(clusters_centroid[i,], data[j,])
        if(distance > max_distance){
          max_distance = distance
          max_distance_index = j
        }
      }
    }
    clusters_centroid[i,] = as.matrix(data[max_distance_index,])
  }
  converged = FALSE
  while(!converged){
    temp_centroid  <- matrix(0,k,ncol(data))
    temp_centroid_count <- matrix(0,k,1)
    for(i in (1:nrow(data))){
      min_distance = 10000000
      min_distance_index = 0
      for(j in (1:k)){
        distance = Euclidean_distance(clusters_centroid[j,], data[i,])
        if(distance < min_distance){
          min_distance = distance
          min_distance_index = j
        }
      }
      temp_centroid[min_distance_index,] = temp_centroid[min_distance_index,]+as.matrix(data[i,])
      temp_centroid_count[min_distance_index,1] = temp_centroid_count[min_distance_index,1]+1
    }
    for(j in (1:k)){
      if(temp_centroid_count[j,1]!=0){
          temp_centroid[j,] = temp_centroid[j,]/temp_centroid_count[j,1]
      }
    }

    if(identical(temp_centroid, clusters_centroid)){
      converged = TRUE
    }
    else{
      clusters_centroid = temp_centroid
    }
  }
  classfied_type <- matrix(0,nrow(data),1)
  for(i in (1:nrow(data))){
    min_distance = 1000000000
    min_distance_index = 0
    for(j in (1:k)){
      distance = Euclidean_distance(clusters_centroid[j,], data[i,])
      if(distance < min_distance){
        min_distance = distance
        min_distance_index = j
      }
    }
    classfied_type[i,1] = min_distance_index
  }
  return(as.factor(classfied_type))
}

#run my kmeans
#my_xx<-mykmeans(8,x)
#summary(my_xx)

#1 Find optimal K
fviz_nbclust(x, kmeans, method = "silhouette")+
  labs(subtitle = "Silhouette method")

#2 See clustering in pair of dimensions
xx <-kmeans(x,8)
with(orgData, pairs(x, col=c(1:8)[xx$cluster]))
library(fpc)
library(factoextra)
stats=cluster.stats(dist(x), xx$cluster)
silhoute <- stats$avg.silwidth
print(silhoute)
#3 Plot clustering result K=8
fviz_cluster(xx, x, geom = "point", ellipse.type = "norm")

#4 Compared ground truth with clustered result
summary(as.factor(xx$cluster))
summary(y)

# Run as optimal K
xx <-kmeans(x,2)
table(y,xx$cluster)
stats=cluster.stats(dist(x), xx$cluster)
silhoute <- stats$avg.silwidth
print(silhoute)
fviz_cluster(xx, x, geom = "point", ellipse.type = "norm")
#compared ground truth with clustered result
summary(as.factor(xx$cluster))


