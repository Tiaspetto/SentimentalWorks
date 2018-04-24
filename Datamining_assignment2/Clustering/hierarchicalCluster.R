#Load Data
orgData <- read.csv("Clustering/ai2013_papers.csv")
y <- rev(levels(orgData[,c("type")]))
x<-orgData[,c(2,3,4,5,6,7,8,9,10,11,12)]

#normalize <- function(x) { 
#  return((x - min(x)) / (max(x) - min(x)))
#}

#x<-as.data.frame(lapply(x, normalize))
#x<-as.matrix(x)
#load library
library(colorspace)
library(gplots)
library(dendextend)
library(factoextra)
library(cluster)

d_x = dist(x, method = "euclidean")
# clustering witg hclust
hc_x <- hclust(d_x, method = "ward.D2")
dend <- as.dendrogram(hc_x)
dend <- rotate(dend, 1:1290)
dend <- color_branches(dend, k=8)

labels_colors(dend) <-
  rainbow_hcl(8)[sort_levels_values(
    as.numeric(orgData[,c("type")])[order.dendrogram(dend)]
  )]

labels(dend) <- paste(as.character(orgData[,c("type")])[order.dendrogram(dend)],
                      "(",labels(dend),")", 
                      sep = "")

dend <- hang.dendrogram(dend,hang_height=0.1)
dend <- set(dend, "labels_cex", 0.5)

some_col_func <- function(n) rev(colorspace::heat_hcl(n, c = c(80, 30), l = c(30, 90), power = c(1/5, 1.5)))

#1 Create cluster heat map and tree structure 
gplots::heatmap.2(as.matrix(x), 
                  main = "Heatmap for the AIpapers data set",
                  srtCol = 20,
                  dendrogram = "row",
                  Rowv = dend,
                  Colv = "NA", # this to make sure the columns are not ordered
                  trace="none",          
                  margins =c(5,0.1),      
                  key.xlab = "Cm",
                  denscol = "grey",
                  density.info = "density",
                  RowSideColors = rev(labels_colors(dend)), # to add nice colored strips        
                  col = some_col_func
)

#2 K=8 clustering result
groups <- cutree(hc_x, k=8) 
summary(silhouette(cutree(hc_x,k=8),d_x))$avg.width
sub_grp <- cutree(hc_x, k = 8)
fviz_cluster(list(data = d_x, cluster = sub_grp))



