#Load Data
chd <- read.csv("C:/Users/rajiv/OneDrive/Desktop/Summer Sem 2020/Project 2/SAheart.data")   #Select the data file
head(chd)
str(chd)


#Normalize the predictor
chd['norm_pred'] = scale(chd$ldl)

#Taking first 362 rows as training set
X = chd$norm_pred[1:362]
X = cbind(rep(1,length(X)),X)#Add ones to Predictor variables matrix
Y = chd$chd[1:362]

#Defining Objective function for Logistic Regression

#Sigmoid Function
sigmoid <- function(z)
{
  g <- 1/(1+exp(-z))
  return(g)
}

#Likelihood Function
MLE <- function(theta)
{
  m <- nrow(X)
  g <- sigmoid(X%*%theta)
  J <- sum((Y*log(g)) + ((1-Y)*log(1-g)))
  return(J)
}


# Define learning rate and tolerance limit
alpha <- 0.0001
s <- 0.0001

# initialize coefficients
beta <- c(0,0)
obj = c(MLE(beta),0,0)

# gradient ascent
for (i in 1:1000000) {
  error <- (Y - sigmoid(X%*%beta))
  delta <- (t(X) %*% error)
  beta <- beta + (alpha*delta)
  obj[i+1]=MLE(beta)

  if(abs(obj[i+1]-obj[i])<s){
    break
    }
}
beta

plot(obj,type = 'l', main = 'Convergence of Objective function')

#sigmoid(t(c(1,0.294399840))%*%beta)
head(obj,200)

test = tail(chd,100)
x = test$norm_pred
prob = c()

for (i in x){
  p = sigmoid(t(c(1,i))%*%beta)
  prob=c(prob,p)
}


test['pred_prob'] = prob
test['pred_chd'] = ifelse(prob>0.5,1,0)

#Confusion Matrix
xtabs(~chd+pred_chd, data = test)
