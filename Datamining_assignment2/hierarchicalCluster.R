orgData <- read.csv("ai2013_papers.csv")
y <- rev(levels(orgData[,c("type")]))
x<-orgData[,c(2,3,4,5,6,7,8,9,10,11,12)]
random_Data <- orgData[sample(nrow(orgData)), ]
random_x <- random_Data[c(1:1290),c(2,3,4,5,6,7,8,9,10,11,12)]
random_type <- random_Data[c(1:1290),c("type")]
randon_y <- rev(levels(random_type))

d_x = dist(random_x)
hc_x <- hclust(d_x, method = "complete")
dend <- as.dendrogram(hc_x)
dend <- rotate(dend, 1:1290)
dend <- color_branches(dend, k=8)

labels_colors(dend) <-
  rainbow_hcl(8)[sort_levels_values(
    as.numeric(random_type)[order.dendrogram(dend)]
  )]

labels(dend) <- paste(as.character(random_type)[order.dendrogram(dend)],
                      "(",labels(dend),")", 
                      sep = "")

dend <- hang.dendrogram(dend,hang_height=0.1)
dend <- set(dend, "labels_cex", 0.5)

some_col_func <- function(n) rev(colorspace::heat_hcl(n, c = c(80, 30), l = c(30, 90), power = c(1/5, 1.5)))

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

