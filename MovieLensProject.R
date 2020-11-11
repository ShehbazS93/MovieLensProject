##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#############################Attempt#############################
library(dslabs)
data("movielens")

#############################Using the average of all movies to see the RMSE and setting that as the base#############################
set.seed(1, sample.kind = "Rounding")
mu <- mean(edx$rating)                                                  #Getting the average of all movies
mu_rmse <- RMSE(mu, validation$rating)                                  #RMSE of mu
rmse_results <- data.frame(method = "Just the Average", RMSE = mu_rmse) #Making table to see progress, organized appearance. Was in notes/lectures

#############################Finding the bi(movie effect) now#############################
movie_avgs <- edx %>%                                                  #Will store the bi of all movies. Will be added onto validation later
  group_by(movieId) %>%                                                #MovieId is used since title is not unique.
  summarize(bi = mean(rating- mu) , n = n(), .groups = 'keep')                           #bi is what's left after we remove the average(mu) from the rating of the movie
predicted_bi_rating <- validation %>%                                  #Similar to predict using the test set and what was in train
  left_join(movie_avgs, by='movieId') %>%                              #Adds bi to the validation set according to movieId
  .$bi + mu 
bi_rmse <- RMSE(predicted_bi_rating, validation$rating)         #Finds out and stores the rmse of the movie effects algorithm
rmse_results <- bind_rows(rmse_results,                         #Adds the movies effect algorithm and it's rmse to the table
                          data_frame(method = "Movies Algorithm",
                                     RMSE = bi_rmse))
rmse_results #prints out the rmse_result to see how it worked compared to mu

#Seeing which movies are highest rating according to bi
movie_titles <- edx %>%                                         #Making a dataset of the number of times each movie was viewed and their bi
  select(movieId, title) %>%
  group_by(movieId, title) %>%
  summarize(n = n(), .groups = 'keep') %>%
  distinct()
validation %>%                                                  #Highest rated movies according to bi
  left_join(movie_avgs, by = "movieId") %>% 
  select(title, bi, n) %>%
  arrange(desc(bi)) %>% 
  distinct() %>%
  top_n(10, bi) 
validation %>%                                                  #Lowest rated movies according to bi
  left_join(movie_avgs, by = "movieId") %>% 
  select(title, bi, n) %>%
  arrange(bi) %>% 
  distinct() %>%
  top_n(10, bi) 
validation %>%                                                  #Find movies with biggest difference between their rating and the prediction, and their number of reviews
  left_join(movie_avgs, by = "movieId") %>% 
  select(movieId,rating,  title, bi, n) %>%
  group_by(title) %>%
  mutate(difference = rating - mu - bi) %>%
  arrange(desc(abs(difference))) %>% 
  distinct() %>%
  select(title, difference, n)



lambda <- 3                                                     #Using 3 because it was the number used in the class example
reg_movie_avg_atmpt <- edx %>%                                   #Getting the movie effect again, but regularizing it using a lambda of 3. 
  group_by(movieId) %>% 
  summarize(reg_bi = sum(rating - mu)/(n() + lambda), n = n(), .groups = 'keep')  #Can't use mean, so sum the top part and then divide by lambda + the count
predicted_reg_bi_rating <- validation %>%                       #Using the regularized bi and mu to predict the rating
  left_join(reg_movie_avg_atmpt, by = "movieId") %>%
  mutate(pred = mu + reg_bi) %>%
  .$pred
reg_bi_rmse <- RMSE(predicted_reg_bi_rating, validation$rating) #RMSE of the regularized movie effect
rmse_results <- bind_rows(rmse_results,                         #Adding the regularized movie effect to the table
                          data_frame(method = "Regularized Movie Algorithm, l = 3",
                                     RMSE = reg_bi_rmse))
rmse_results

#Optimizing lambda
lambdas <- seq(0,10,0.1)                           #Different values of lambda to use
just_the_sum <- edx %>%                            #Saw this used in the lecture notes so decided to try it. Gets the top half of the 'mean'
  group_by(movieId) %>%                            #Attachs the sum to movieIds so it can be added to another calculation
  summarize(s = sum(rating - mu), n_i  = n(), .groups = 'keep') 
reg_movie_rmse <- sapply(lambdas, function(l){     #Need sapply to go through the lambdas
  preds <- validation %>%  
    group_by(movieId) %>%                          #Grouping by movieIds since this is for movie effects
    left_join(just_the_sum, by = "movieId") %>%    #Adds the sums to validation data set via movieIds
    mutate(bi = s/(n_i + l)) %>%                   #Saves it a calculation I guess
    mutate(pred = mu + bi) %>%  
    .$pred
  return(RMSE(preds,validation$rating))
})
plot(lambdas, reg_movie_rmse, xlab = "Lambda", ylab = "RMSE", main = "Regularized Movie Algorithm RMSE")   #3 was kinda close to optimal actually, guess it mught not decrease the rmse much then
lambda <- lambdas[which.min(reg_movie_rmse)]  #Saves the optimal lambda
rmse_results <- bind_rows(rmse_results,
                          data_frame(method = paste("Regularized Movie Alogrithm, l = ", lambda), #Written this way so that the optimal lambda is
                                     RMSE = min(reg_movie_rmse)))
rmse_results

#Seeing which movies are highest rating according to bi after using regularization
opt_reg_movie_avgs <- edx %>%                     #Making a dataset of number of times each movie was reviewed along with the regularized bi
  group_by(movieId) %>%
  mutate(bi = sum(rating-mu)/(n() + lambda), n = n()) %>% 
  select(movieId, title, bi, n) %>%
  arrange(movieId) %>% 
  unique()
validation %>%                                    #Find movies with highest bi and number of ratings after optimizing the regularization for movie effects
  left_join(opt_reg_movie_avgs, by = "movieId") %>%
  select(title = title.x, bi, n) %>% 
  arrange(desc(bi)) %>% 
  distinct() %>%
  top_n(10, bi)
validation %>%                                    #Find movies with lowest bi and number of ratings after optimizing the regularization for movie effects
  left_join(opt_reg_movie_avgs, by = "movieId") %>%
  select(title = title.x, bi, n) %>% 
  arrange(bi) %>% 
  distinct() %>%
  top_n(-10, bi)
validation %>%                                    #Movies with the 10 biggest differences between rating and prediction, and their number of reviews
  left_join(opt_reg_movie_avgs, by = "movieId") %>% 
  select(movieId,rating,  title= title.x, bi, n) %>%
  group_by(title) %>%
  mutate(difference = rating - mu - bi) %>%
  arrange(desc(abs(difference))) %>% 
  distinct() %>%
  select(title, difference, n)
#Seems better, lots of low number movies aren't as high



#############################Adding user effects to the movie model#############################
user_avgs <- edx %>%                        #Making averages for the user effect
  left_join(movie_avgs, by="movieId") %>%   #Need to add in movie effects to subract it from the rating and mu to find the user effect
  group_by(userId) %>%                      #Grouping by userId to find the user effect 
  summarize(bu = mean(rating - mu - bi), .groups = 'keep')
predicted_bu_bi_rating <- validation %>% 
  left_join(movie_avgs, by = "movieId") %>%  #Adding movie effects
  left_join(user_avgs, by = "userId") %>%    #Adding user effects
  mutate(pred = mu + bi + bu) %>% 
  .$pred
model_bu_bi_rmse <- RMSE(predicted_bu_bi_rating, validation$rating)  #Getting the RMSE the movie + user effects
rmse_results <- bind_rows(rmse_results,                              #Adding the movie & user RMSE to the table
                          data_frame(method = "Movies and Users Algorithm", 
                                     RMSE = model_bu_bi_rmse))
rmse_results


#Regularizing both user and movie effects
reg_rmse <- sapply(lambdas, function(l){   #Using the same lambdas as before
  bi <- edx %>%                            #Getting the movie effect
    group_by(movieId) %>% 
    summarize(bi = sum(rating - mu)/(n()+l), .groups = 'keep') 
  bu <- edx %>%                            #Getting the user effect
    left_join(bi, by = "movieId") %>%
    group_by(userId) %>%
    summarize(bu = sum(rating - (mu + bi))/(n()+l), .groups = 'keep')
  pred <- validation %>%                   
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    mutate(pred = mu + bi + bu) %>% 
    .$pred
  return(RMSE(pred, validation$rating))
})
plot(lambdas, reg_rmse, xlab = "Lambda", ylab = "RMSE", main = "Regularized Movie and User Algorithm RMSE") #Plotting to see how well lamda lowers RMSE
lambda <- lambdas[which.min(reg_rmse)]  #Saving the optimal lamdba to use in the table
rmse_results <- bind_rows(rmse_results, 
                          data_frame(method = paste("Regularized Movie and User Algorithm, l = ", lambda),
                                     RMSE = min(reg_rmse)))
rmse_results

#############################Genres#############################
movie_genres <- edx %>%  #Making a list of genres and how many have them. Not seperating each into seperate individual genres
  group_by(genres) %>% 
  summarize(n = n(), .groups = 'keep') %>%  
  distinct()
genre_avgs <- edx %>%                                                #Calculating the genre effect
  left_join(movie_avgs, by = "movieId") %>%                          #Adding the movie effect via movieId
  select(userId, movieId, rating, timestamp, title, genres,bi,n) %>% #Removing some unneccesary info
  left_join(user_avgs, by = "userId") %>%                            #Adding the user effect via userId
  group_by(genres) %>% 
  summarize(bg = mean(rating - mu - bu - bi), .groups = 'keep')
predicted_iug <- validation %>% 
  left_join(movie_avgs, by = "movieId") %>%                      #Adding the movie effect/bias 
  select(userId, movieId, timestamp, title, genres, bi, n) %>%   #Making sure just the necessary stuff is here
  left_join(user_avgs, by = "userId") %>%                        #Adding the user effect/bias
  left_join(genre_avgs, by  = "genres") %>%                      #Adding the genre effect/bias
  mutate(pred = mu + bi + bu + bg) %>%
  .$pred
model_iug_rmse <- RMSE(predicted_iug, validation$rating)         #RMSE of the non-regularized movie, user and genre effects/biases
rmse_results <- bind_rows(rmse_results,                          #Adding the RMSE to the table
                          data_frame(method = "Movies, Users, and Genres Algorithm",
                                     RMSE = model_iug_rmse))
rmse_results

#Optimize lambda for Genres 
reg_rmse <- sapply(lambdas, function(l){                         #Same lambda as earlier 0->10 by 0.1
  bi <- edx %>%                                                  #Getting the movie bias, same as before
    group_by(movieId) %>% 
    summarize(bi = sum(rating - mu)/(n()+l), .groups = 'keep')
  bu <- edx %>%                                                  #Getting the user bias, same as before
    left_join(bi, by = "movieId") %>%
    group_by(userId) %>%
    summarize(bu = sum(rating - (mu + bi))/(n()+l), .groups = 'keep')
  bg <- edx %>%                                                  #Getting genre effect, similar to how the user and movie biases were gotten
    left_join(bi, by = "movieId") %>% 
    left_join(user_avgs, by = "userId") %>% 
    group_by(genres) %>%
    summarize(bg = sum(rating - mu - bu - bi)/(n() + l), .groups = 'keep')
  pred <- validation %>%                                         #Same prediction as before, but with genre bias added
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    select(movieId, genres, bi, bu) %>%
    left_join(bg, by = "genres") %>%
    mutate(pred = mu + bi + bu + bg) %>% 
    .$pred
  return(RMSE(pred, validation$rating))
})
plot(lambdas, reg_rmse, xlab = "Lambda", ylab = "RMSE", main = "Regularized Movie, User and Genre Algorithm RMSE")
lambda <- lambdas[which.min(reg_rmse)]

rmse_results <- bind_rows(rmse_results,                         #Adding the RMSE of regularized movie, user and genre biases
                          data_frame(method = paste("Regularized Movie, User, and Genre Algorithm, l = ", lambda),
                                     RMSE = min(reg_rmse)))
rmse_results

#############################Review Week#############################
edx_date_week <- edx %>%                                        #Saving the time the review was written as as a date
  mutate(review_date = lubridate::as_datetime(timestamp))  %>%  #Changing the time stamp to date with time
  mutate(review_week = cut(as.Date(review_date), "week"))       #Changing the date the review was written as the first Monday of the week it was written
review_week <-edx_date_week %>%                                 #Saving the number of reviews that were each week
  select(timestamp, review_week) %>%                            #Keeping the timestamp to bind to another set and the week the review was written
  group_by(review_week) %>%                                     
  summarize(n = n(), .groups = 'keep')                                            #Getting the number of reviews that were written each week
review_week_avgs <- edx_date_week %>% group_by(review_week) %>%  #Getting the effect of the week a review was written
  left_join(movie_avgs, by = "movieId") %>%                     #Adding the movie bias to the dataset
  select(userId, movieId, rating, timestamp, title, genres,review_week, bi) %>%  #Removing unnecessary info
  left_join(user_avgs, by = "userId") %>%                       #Adding the user bias to the data set
  left_join(genre_avgs, by = "genres") %>%                      #Adding the genre bias to the data set
  mutate(bd = mean(rating - mu - bi - bu - bg)) %>%             #Calculating the review week bias
  select(review_week, bd) %>%                                   #Leaving just the review week and the bias of that week
  distinct() %>%                                                #Cleaning up the data set to remove duplicates
  arrange(review_week)                                          #Organizing the dataset bu review week

#review_week_avgs
predicted_iugd <- validation %>%                                #Predicted rating using movie, user, genre, and review week bias
  left_join(movie_avgs, by ="movieId") %>%                      #Adding movie bias
  select(userId, movieId, rating, timestamp, title, genres, bi) %>% #Removing some variables from the dataset
  left_join(user_avgs, by = "userId") %>%                       #Adding user bias
  left_join(genre_avgs, by = "genres") %>%                      #Adding genre bias
  mutate(review_week = lubridate::as_datetime(timestamp)) %>%   #Changing the time stamp to dates
  mutate(review_week = cut(as.Date(review_week), "week")) %>%   #Changing the date to week
  select(rating, bi, bu, bg, review_week) %>%                   #Removing unnecessary info besides the biases and review week
  left_join(review_week_avgs, by = "review_week") %>%           #Adding review week bias
  mutate(pred = mu + bi + bu + bg + bd) %>%                     #Calculating the predicted rating
  .$pred
model_iugd_rmse <- RMSE(predicted_iugd, validation$rating)      #Getting and saving the RMSE of mu and the biases
rmse_results <- bind_rows(rmse_results,                         #Saving the RMSE to the table
                          data_frame(method = paste("Movies, Users, Genres, and Review Week Algorithm"),
                                     RMSE = model_iugd_rmse))
rmse_results

#Optimize lambda for Review Week
reg_rmse <- sapply(lambdas, function(l){                        #Using sapply to find the best lambda
  bi <- edx %>%                                                 #Getting the movie bias
    group_by(movieId) %>% 
    summarize(bi = sum(rating - mu)/(n()+l), .groups = 'keep')
  bu <- edx %>%                                                 #Getting the user bias
    left_join(bi, by = "movieId") %>%
    group_by(userId) %>%
    summarize(bu = sum(rating - (mu + bi))/(n()+l), .groups = 'keep')
  bg <- edx %>%                                                 #Getting the genre bias
    left_join(bi, by = "movieId") %>% 
    left_join(user_avgs, by = "userId") %>% 
    group_by(genres) %>%
    summarize(bg = sum(rating - mu - bu - bi)/(n() + l), .groups = 'keep')
  brw <- edx %>%                                                #Getting the review week bias
    left_join(bi, by = "movieId") %>% 
    left_join(bu, by = "userId") %>% 
    left_join(bg, by = "genres") %>% 
    mutate(review_week = lubridate::as_datetime(timestamp)) %>%
    mutate(review_week = cut(as.Date(review_week), "week")) %>%
    group_by(review_week) %>%
    mutate(brw = sum(rating - mu - bu - bi - bg)/(n() + l)) %>%
    select(review_week, brw) %>% 
    distinct(review_week, brw)
  pred <- validation %>%                                       #Getting the regularized RMSE for a lambda
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    left_join(bg, by = "genres") %>%
    mutate(review_week = lubridate::as_datetime(timestamp)) %>%
    mutate(review_week = cut(as.Date(review_week), "week")) %>%
    left_join(brw, by ="review_week") %>% 
    mutate(pred = mu + bi + bu + bg + brw) %>% 
    .$pred
  return(RMSE(pred, validation$rating))
})
plot(lambdas, reg_rmse, xlab = "Lambda", ylab = "RMSE", main = "Regularized Movie, User, Genre, and Review Week Algorithm RMSE")                                        #Plotting the RMSE vs lambda to see how much each lambda lowered RMSE
lambda <- lambdas[which.min(reg_rmse)]                         #Saving the best lambda to lambda
rmse_results <- bind_rows(rmse_results,                        #Adding RMSE to the table
                          data_frame(method = paste("Regularizaed Movie, User, Genre  and Review Week Effect, l = ", lambda),
                                     RMSE = min(reg_rmse)))
rmse_results
