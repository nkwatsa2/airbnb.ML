# ML Task: Can we somehow use our data to guess or predict a reasonable price for a listing?
# Algorithm: KNN
##### Using data set of listings in Washington dc


setwd("/Users/akankwasa/Desktop/R/Airbnb_ML/")

library(dplyr)
library(magrittr)
library(naniar)

listings <- read.csv("listings.csv", header = TRUE, na.strings = c("", "N/A", "NA"), sep = ",")
class(listings)

# There is many columns not needed so extract the columns key to the task
dc_listings <- listings[c("host_response_rate", "host_acceptance_rate", "host_listings_count", 
                          "latitude", "city", "zipcode", "state", "accommodates",
                          "room_type", "bedrooms", "bathrooms", "beds", "price", "cleaning_fee",
                          "security_deposit", "minimum_nights", "maximum_nights", "number_of_reviews"
                          )] #subsetting listings dataframe
class(dc_listings)
is.na(dc_listings) # checking whether data frame contains missing values

# One strategy to determine prices is to look at similar listings´prices. 
# KNN is one algorithm to do this

#-----------------------------------------------------------------------------------------------
#Using Euclidean distance to check similarity.
#Ex.1: A listing which accommodates 3 pple. We use accomodates column

first_row_acc_value <- dc_listings$accommodates[1] # first space in accommodates row in our dataset
our_acc_value <- 3 
first_distance <- abs(our_acc_value - first_row_acc_value) #abs() function to calculate absolute distance

# calculating euclidean distance btn our listing and every listing in accommodates column
dc_listings <- dc_listings %>% mutate(euclidean_distance = abs(our_acc_value - accommodates))

# looking at diff (distinct) values in euclidean_distance column
euc_distance <- pull(dc_listings, euclidean_distance) 
table(euc_distance)

# using accomodates is not the only fcator to consider when checking for similarity with our listing.
# randomly pick any 3 of the CLOSEST listings (listings with minimum distance i.e euclidean_distance = 0)
set.seed(1)
zero_distance_indices <- which(dc_listings$euclidean_distance == 0)
random_three_indices <- sample(zero_distance_indices, 3, replace = FALSE) # replace paramenter for whether sampling with/ without replacement

class(dc_listings$price)
# price column is of factor class, and contains comas and dollar signs. 
# to make it more usable, convert entire column to double data type. 
# to covert to double, FIRST convert to CHARACTER and then to DOUBLE

str(dc_listings$price) # shows tsructure of column
dc_listings$price <- as.character(dc_listings$price)
#removing dollar signs and asign to new column
dc_listings <- dc_listings %>% mutate (
  tidied_price = gsub("[\\$,]", "", price)) # gsub() function used for replacement operations
dc_listings$tidied_price <- as.numeric(dc_listings$tidied_price) # convert to double

#calcualte average price of our three listings from our sample. Average price will be price for our listings
avg_price_of_sample <- mean(random_three_indices)

#----------------------------------------------------------------------------------------------


# above we outlined knn step by step using one listing whose price we wanted to predict.

#-----------------------------------------------------------------------------------------------
# below we will use knn using caret library, evaluate well how the model predicts outcomes of data it has not seen before.
# form of model perfomance evaluation (cross valiadation) used will be hold out validation.

library(caret)
################# STEP 1: split dataset into 2; training(majority of data ), test sets #############

set.seed(12345)
training_set_indices <- createDataPartition(
  y = dc_listings[["tidied_price"]],
  p = 0.8,
  list = FALSE) # list is false in order to return vector of indices (instead of list) w/c is more natural to pass into a tibble to split the data
test_dc_set <- dc_listings[-training_set_indices, ]
training_dc_set<- dc_listings[training_set_indices, ]
# train the algorithm and evalaute it against test set.

# trainConrol() to set parameters for validation process before we start training. specifies how training will be done
train_control_dc_listings <- trainControl(method = "none") # we specify this first before training the algorithm
# method = "none"  done to avoid cross validation

######### STEP 2 : Train the algorithm using data from training set #########
dc_listings_knn_model <- train(tidied_price ~ accommodates + maximum_nights,
                               data = training_dc_set,
                               method = "knn",
                               tr_control = train_control_dc_listings)
# outcome of train() is a list that essentially contains the trained ML model


######### STEP 3: holdout valiadation process; For each listing in the test set, we will calculate the average price of its nearest neighbors. These prices become the model's predictions. ##########
dc_listings_predictions <- predict(dc_listings_knn_model, newdata = test_dc_set)
dc_listings_predictions


######## STEP 4: Compare these predictions against the actual values of the listings as given in the tidiedprice column. ######

# using error to quantify difference
test_dc_set <- test_dc_set %>% mutate(prediction_error = tidied_price - dc_listings_predictions)

# so many errors so use single summary metric. (I used RMSE)
squared_error <- test_dc_set$prediction_error ** 2 #square all the errors first
rmse_metric <- sqrt(mean(squared_error))
# our RMSE is ~ $117.83.
