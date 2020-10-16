Project 2
================
Shelnes Elie
10/13/2020

# *Introduction*

The data in this project represents the online news popularity. This
dataset summarizes a heterogeneous set of features about articles
published by Mashable in a period of two years. The goal is to predict
the number of shares in social networks (popularity).Some of the
variables used are:

  - weekday\_is/is\_weekend = represents what day of the week or is it
    the weekend.  
  - shares = \# of shares/target  
  - LDA = Close to LDA topic  
  - n\_tokens\_content = Number of words in the content  
  - data\_channel\_is\_lifestyle = Is data channel ‘Lifestyle’?  
  - data\_channel\_is\_entertainment = Is data channel
    ‘Entertainment’?  
  - data\_channel\_is\_bus = Is data channel ‘Business’?  
  - data\_channel\_is\_socmed = Is data channel ‘Social Media’?  
  - data\_channel\_is\_tech = Is data channel ‘Tech’?  
  - data\_channel\_is\_world = Is data channel ‘World’?

# *Summarization*

The below summary describes the measure of center for shares.

    ## [1] "weekday_is_friday"

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##      22     974    1500    3285    2700  233400

# *Graphs*

## Barchart

The below barchart shows the count for each data channel when it equals
1 or is active. As you can see, world, tech, and entertainment’s
frequency are very high while lifestyle and socmed are very low.

``` r
# Plots

ggplot(channels, aes(data_channel)) +
  geom_bar()
```

![](Project02_files/figure-gfm/plot1-1.png)<!-- -->

## Boxplot

Below displays boxplots for shares broken by data channel.We can see
that there’s outliers for the shares. All the data channels seem to have
condensed measure of center, especially bus.This should make predicting
shares easier for some data\_channels.

``` r
ggplot(channels, aes(x = data_channel, y = shares))+ 
  geom_jitter(aes(color = data_channel)) + ggtitle("Boxplot for Shares") + 
  geom_boxplot() 
```

![](Project02_files/figure-gfm/plot2-1.png)<!-- -->

## Scatterplot

The scatterplots below represent the total words in the content vs
shares broken down by each data\_channel. We can see that all the
data\_channels try to the words below 2000 except for some outliers.
This must be to maintain the attention of the readers as the shares are
the highest in this area.

``` r
ggplot(channels, aes(x = n_tokens_content, y = shares, group = data_channel)) + geom_point(aes(color = data_channel)) + geom_smooth(method = 'lm', color = 'green') +
  ggtitle("Number of Words in Content vs Shares") + facet_wrap(~data_channel)
```

    ## `geom_smooth()` using formula 'y ~ x'

![](Project02_files/figure-gfm/plot3-1.png)<!-- -->

# *Modeling*

Goal: Create models for predicting the shares variable.

## Regression Tree

A (not ensemble) tree-based model chosen using leave one out cross
validation  
Root MSE is used on the test data set to see how well the model is.

``` r
# a (not ensemble) tree-based model chosen using leave one out cross validation
# Regression Tree
treeFit <- tree(shares ~ . ,data = newsDataTrain)
plot(treeFit); text(treeFit)
```

![](Project02_files/figure-gfm/regressiontree-1.png)<!-- -->

``` r
cvTree <- cv.tree(treeFit); cvTree
```

    ## $size
    ## [1] 7 1
    ## 
    ## $dev
    ## [1] 313238696061 314329327976
    ## 
    ## $k
    ## [1]       -Inf 6573268322
    ## 
    ## $method
    ## [1] "deviance"
    ## 
    ## attr(,"class")
    ## [1] "prune"         "tree.sequence"

``` r
# Visual of How CV Error Changes w/# of Nodes
plot(cvTree$size, cvTree$dev, type="b")
```

![](Project02_files/figure-gfm/regressiontree-2.png)<!-- -->

``` r
# Predictions usingpredict
pred <- predict(treeFit, newdata=dplyr::select(newsDataTest,-shares))

# Root MSE
sqrt(mean((pred-newsDataTest$shares)^2))
```

    ## [1] 7170.225

## Boosted Tree

Model chosen using cross-validation  
Utilizing misclassification rate to evaluate the model.

``` r
boostTree <- train(shares ~ ., data = newsDataTrain, method = "gbm",
trControl = trainControl(method = "repeatedcv", number = 4, repeats = 2),
preProcess = c("center", "scale"), verbose = FALSE)
boostTree
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 3990 samples
    ##   59 predictor
    ## 
    ## Pre-processing: centered (59), scaled (59) 
    ## Resampling: Cross-Validated (4 fold, repeated 2 times) 
    ## Summary of sample sizes: 2991, 2994, 2993, 2992, 2993, 2993, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE      Rsquared    MAE     
    ##   1                   50      8387.399  0.01015314  2994.939
    ##   1                  100      8396.802  0.01254114  3007.555
    ##   1                  150      8397.652  0.01458175  3006.163
    ##   2                   50      8399.464  0.01379720  3021.425
    ##   2                  100      8407.442  0.01844978  3019.269
    ##   2                  150      8412.326  0.02125119  3022.108
    ##   3                   50      8430.503  0.01418638  3012.672
    ##   3                  100      8458.653  0.01869861  3038.396
    ##   3                  150      8498.559  0.01962867  3076.585
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## 
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 50, interaction.depth =
    ##  1, shrinkage = 0.1 and n.minobsinnode = 10.

``` r
boostTbl <- table(data.frame(pred = predict(boostTree, newsDataTest), true = newsDataTest$shares))


#misclassificatoon rate to evaluate model
1-c(boost = sum(diag(boostTbl)/sum(boostTbl)))
```

    ## boost 
    ##     1

``` r
render_html_fun <- function(hub_in){
  rmarkdown::render('Project-02.Rproj',
                    output_file = paste0('Project2', hub_in, "_", '.html'),
                    params = list(hub = hub_in, set_title = hub))}


hubs_input <- c("weekday_is_monday", "weekday_is_tuesday", "weekday_is_wednesday", 
                "weekday_is_thursday", "weekday_is_friday", "weekday_is_saturday", 
                "weekday_is_sunday")


walk(hubs_input, render_html_fun)
```
