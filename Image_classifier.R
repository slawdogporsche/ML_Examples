library(keras)

## Importing Data ##
fashion_mnist <- dataset_fashion_mnist()

## Separating Testing and Training Set ## 
c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

## Setting labels for classes (since they're 1-9 in data) ##
class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat', 
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')

## Checking Dimensions of Data ##
dim(train_images)
dim(train_labels)
train_labels[1:20]
dim(test_images)
dim(test_labels)

library(tidyr)
library(ggplot2)

## Setting up image for visualization ## 
image_1 <- as.data.frame(train_images[1, , ])
colnames(image_1) <- seq_len(ncol(image_1))
image_1$y <- seq_len(nrow(image_1))
image_1 <- gather(image_1, "x", "value", -y)
image_1$x <- as.integer(image_1$x)

## Visualization ##
ggplot(image_1, aes(x = x, y = y, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "black", na.value = NA) +
  scale_y_reverse() +
  theme_minimal() +
  theme(panel.grid = element_blank())   +
  theme(aspect.ratio = 1) +
  xlab("") +
  ylab("")

## Pre Processing Data ##
train_images <- train_images / 255
test_images <- test_images / 255

## Display first 25 images in training set and display class ##
par(mfcol=c(5,5))
par(mar=c(0, 0,1.5, 0), xaxs ='i', yaxs='i')
for (i in 1:25) {
  img <- train_images[i, ,]
  img <- t(apply(img, 2, rev)) 
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', 
        yaxt ='n', main = paste(class_names[train_labels[i] + 1]))
}

## Building a keras model ## 
## First layer (layer_flatten) transforms format of images 
## from a 2d array of 28x28 pixels to a 1d array of 28x28=784 pixels
## After pixels are flattened, neural network has two dense layers
## These are densely connected neural layers
## First layer has 128 nodes
## The second is a 10 node softmax layer, returning an array of 
## 10 probability scores that sum to 1
## Each node contains a score that indicates the prob that the 
## image belongs to one of the 10 classes ## 
model <- keras_model_sequential()
model %>%
  layer_flatten(input_shape = c(28,28)) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = "softmax")

## Before compiling the model, it needs a few settings ## 
## These are added during the compile step ##
## Loss function - Measures how accurate model is during training 
## We want to minimize this function to steer the model 
## Optimizer function - This is how the model is updated based on
## data it sees and loss function
## Metrics - Used to monitor the training and testing steps. 
## Following example uses accuracy, the fraction of images
## that are correctly classified 
model %>% compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = c('accuracy')) ##

## Training the model requires the following steps ##
## Feeding training data to model (train_images, train labels) ##
## Association of images and labels ##
## Asking model to make predictions about test set (test_images)

## To train, call the fit method ## 
model %>% fit(train_images, train_labels, epochs = 5, verbose = 2)

## Evaluating accuracy ##
score <- model %>% evaluate(test_images, test_labels, verbose = 0)
cat('Test loss:', score$loss, "\n")
cat('Test accuracy:', score$acc, "\n")

## Accuracy on test is worse than training ##
## This is due to overfitting ##

## With the model trained, we can use it to make predictions ##
predictions <- model %>% predict(test_images)

## This describes the confidence of the model that it corresponds
## to each of the 10 articles of clothing ##
predictions[1,]

## To find highest confidence value ##
which.max(predictions[1, ])

## We can also directly get class prediction ##
class_pred <- model %>% predict_classes(test_images)
class_pred[1:20]

test_labels[1]

## Plotting several images with their predictions ##
par(mfcol=c(5,5))
par(mar=c(0,0,1.5,0), xaxs='i', yaxs='i')
for (i in 1:25) {
  img <- test_images[i, ,]
  img <- t(apply(img, 2, rev))
# Subtract 1 a labels go from 0 to 9
  predicted_label <- which.max(predictions[i, ]) - 1
  true_label <- test_labels[i]
  if (predicted_label == true_label) {
    color <- '#008800'
  } else {
    color <- '#bb0000'
  }
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt ='n',
        main = paste0(class_names[predicted_label + 1], " (", 
                      class_names[true_label + 1], ")"),
        col.main = color)
}

## Making a prediction about a single image ##
## Keep the batch dimension, as this is expected by model 
img <- test_images[1, , , drop = FALSE]
dim(img)

predictions <- model %>% predict(img)
predictions

## Predict returns a list of lists, one for each image 
## Grab predictons for only image in batch
## Subtract 1 as labels are 0 based
prediction <- predictions[1,] - 1
which.max(prediction)

class_pred <- model %>% predict_classes(img)
class_pred
