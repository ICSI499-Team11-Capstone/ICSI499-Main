
truncated_sampling <- function(mean, covariance, lower, num_samples){
  library(tmvtnorm)

  library(corpcor)
  library(gmm)
  print('r1')
  mean_vector = c(mean)
  mean_vector = sapply(mean_vector, as.numeric)
  mean_vector = as.vector(mean_vector)
  print('r2')
  covariance_matrix = as.matrix(sapply(covariance, as.numeric))
  covariance_matrix = t(covariance_matrix) %*% covariance_matrix #https://mathworld.wolfram.com/SymmetricMatrix.html
  print('r3')
  covariance_matrix = make.positive.definite(covariance_matrix, tol=1e-3) #https://mathworld.wolfram.com/PositiveDefiniteMatrix.html
  print('r4')
  lower_bound = c(lower)
  lower_bound = sapply(lower, as.numeric)
  lower_bound = as.vector(lower_bound)
  print(lower_bound)
  print('r5')
  x <- rtmvnorm(n=num_samples, mean=mean_vector, sigma=covariance_matrix, upper = lower_bound)
  print('r6')
  df = data.frame(x)
  print('r7')
  return(df)
}
