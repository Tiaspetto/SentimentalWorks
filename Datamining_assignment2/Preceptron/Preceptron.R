Iris = iris
binary_iris = subset(Iris, Species=="versicolor" | Species =="setosa")
random_iris = binary_iris[sample(nrow(binary_iris)), ]

#Preprcess data from original dataset
preprocess<-function(dataAndlabels)
{
  data <- as.matrix(dataAndlabels[,1:4])
  labels <- as.matrix(dataAndlabels[,5])
  num_labels <- matrix(1, nrow(dataAndlabels),1)
  for(i in 1:nrow(labels)){
    if(labels[i,1]=="setosa"){
      num_labels[i,1]=-1
    }
  }  
  return(list(data,num_labels))
}


#preceptron learning function
perceptron <-function(data, labels, b=0, learning_rate = 0.02)
{
  paramters <-matrix(0,4,1)
  max_loop = 0
  
  converged <- FALSE
  loop_times <- 0
  while(converged == FALSE){
    mis_class <- 0
    for(i in 1:nrow(data)){
      predict = data[i,1:4]%*%paramters+b
      if(labels[i,1]*predict<=0){
        paramters <- paramters+learning_rate*data[i,1:4]*labels[i,1]
        mis_class = mis_class+1
      }
    }
    if(mis_class == 0){
      converged = TRUE
    }
    else{
      max_loop = max_loop +1
      if(max_loop>10000){
        print(paste(b, learning_rate,"can't converage after 10000 iterarion", sep=" "))
        return(list(converged,paramters))
      }
    }
    
  }
  
  print(paste(b,learning_rate,"converaged and return",sep=" "))
  
  return(list(converged,paramters))
}

#grid_search method , for different bias value and learning rate
grid_search <- function(data)
{
  b_array <- seq(-1,1,0.2)
  learning_rate <- seq(0.01,20,0.01)
  ori_data <- preprocess(data)
  count_search = 0
  count_not_converged = 0
  for(b in b_array){
    for(l in learning_rate){
      result = perceptron(ori_data[[1]],ori_data[[2]], b, l)
      count_search = count_search + 1
      if(result[[1]]==FALSE){
        count_not_converged = count_not_converged
      }
    }
  }
  print(paste("searched",count_search,"times","not converged",count_not_converged,"times", sep=" "))
}

grid_search(random_iris)

      




