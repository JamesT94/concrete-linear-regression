# Predicting the compressive strength of concrete
For a full project overview with detailed discussion please see the full-project notebook [here](https://github.com/JamesT94/concrete-compressive-strength/blob/master/full-project-notebook.ipynb)

## Introduction
Concrete quality is typically defined by it's compressive strength. Civil engineers will carry out rigorous testing using varying combinations of raw materials and curing time. The process of testing concrete compressive strength can be found here. With curing time taking up to 91 days in some cases, the whole process is very time consuming and labour intensive.

There is a clear opportunity for digital simulation to reduce wait time and total number of combinations. With the data set acquired we can learn about the relations between variables and develop a predictive model. Highlighting potentially optimal combinations to then be used in physical testing will provide enormous benefit and significantly reduce labour and testing costs.

## Data
In this project we will be using the [Concrete Compressive Strength data set](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength) from the UCI Machine Learning Repository.

The data contains over 1,000 instances of concrete each with 9 variables (including compressive strength).

* Cement
* Blast Furnace Slag
* Fly Ash
* Water
* Superplasticizer
* Coarse Aggregate
* Fine Aggregate
* Age
* Concrete Compressive Strength

## Exploratory Data Analysis
The first, and arguably most important, step in a data science project is to explore the data in an attempt to gain insights that will guide the rest of the project. EDA generally includes ensuring the data is clean and usable, visualising features and their relationships, observing distributions, etc.

First of all, let's check the disribution of our target variable and take a closer look at the input features and try to identify any correlations.

![str_dist](imgs/strength_dist.png)

![pair_plot](imgs/pair_plot.png)

At first glance, there doesn't seem to be any high correlation between any 2 features. Although Cement and Compressive Strength look like they may have some correlation. Let's determine the Pearson Correlation coefficients to get a numerical value of the strength of the correlations.

![r_heat](imgs/r_heatmap.png)

As expected, the highest positive correlations is between **Compressive Strength** and **Cement**. 

**Age** and **Superplasticizer** also have a positive impact on the **Compressive Strength**.  

By sorting for the largest magnitude values (both positive and negative) we can focus on the strong correlations with the target variable, and the strong correlations between other features.

As expected, there aren't many high correlations with Compressive Strength (CS). The only significant ones are the following 3 top features.

|Feature|Correlation w/ CS|
|:---|:---|
|Cement|0.50|
|Superplasticizer|0.37|
|Age|0.33|

Looking at the top 3 with the highest negative correlation, perhaps water could be a useful variable, Fine Aggregate and Age are unlikely to be helpful in isolation.  

|Feature|Correlation w/ CS|
|:---|:---|
|Water|-0.29|
|Fine Aggregate|-0.17|
|Age|-0.16|

Some other notable strong correlations include the following.

|Features|Correlation|
|:---|:---|
|Superplasticizer / Water|-0.66|
|Fine Aggregate / Water|-0.45|
|Cement / Fly Ash|-0.40|
|Fly Ash / Superplasticizer|0.38|

Already we can determine the 3 most postive and negative influences on Compressive Strength.

Additionally, there are strong correlations between some of the feature variables. We can see a highly negative correlation between **Superplasticizer** and **Water**, but a positive correlation between **Superplasticizer** and **Fly Ash**.

Perhaps plotting these relationships in a visual way will help to gain more insight.

![feat1](imgs/feature_compare1.png)

From this plot we can make some sensible observations on the relationships between these variables and compressive strength:  

* **Compressive Strength** correlates positively with **Cement**
* **Compressive Strength** correlates positively with **Age**, though less than **Cement**
* Older **Cement** tends to require more **Water**, as shown by the larger green data points
* **Compressive Strength** correlates negatively with **Water**
* High **Compressive Strength** with a low **Age** requires more **Cement**

![feat2](imgs/feature_compare2.png)

From this plot we can make further observations on the relationships between this second set of variables and compressive strength:  

* **Compressive strength** correlates negatively with **Fly Ash**
* **Compressive strength** correlates positively with **Superplasticizer**

Through some very simple charts we have discovered relationships between ingredients that allow us to make predictions on what our future model will value when seeking a high compressive strength.  

It is likely that the ideal concrete mixture (when prioritising compressive strength) will consist of:

* Large quantity of **Cement**
* Potentially a long aging process however this comes at the cost of adding **Water**, which negatively impacts the strength
* Large quantity of **Superplasticizer**

## Data Preparation
Prior to fitting a machine learning model, steps must be taken to prepare the data to get the most out of the models.

* Splitting data into train and test sets ensures that we evaluate our model on unseen data.
* Scaling the features to have a mean residing at 0 and a standard deviation of 1 helps to bring a consistency to our variables. This stops the model putting higher weights on greater values, and vice versa.

## Modelling
Now that the data is prepared, we can fit different models on the training data and compare their performance on predicting the test data. To evaluate the models we'll use a variety of metrics suitable to a regression problem like this.

This diagram from the [scikit-learn website](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html) can be helpful tool when it comes to deciding which models to initially compare.

![Machine Learning Map](imgs/ml_map.png)

Since we "predicting a quantity" we are solving a regression problem. We also have less than 100k samples so **Lasso** and **Elastic Net** are the recommended estimators. Just to cover the basics and completeness, we'll also throw in basic **Linear Regression** and **Ridge**

## Linear Regression
In this section we will evaluate 4 different linear regression models:
* Simple
* Ridge
* Lasso 
* Elastic Net 

### Simple (Ordinary Least Squares)
As the hallmark of any regression problem, linear regression is a simple but effective way of making predictions when you expect your data to have mostly linear relationships.

Sometimes the Ordinary Least Squares method can suffer from a low bias and a consequently high variance. To remedy some of this, we can introduce regularisation in the form of Ridge and Lasso regression.

### Ridge
Ridge regression introduces a penalty on the size of coefficients which can address some of the problems faced when using simple regression. In simple terms, the introduction of a term lambda is multiplied by the coefficient and added to the overall error, therefore encouraging a lower coefficient that is less sensitive to change.

To identify the best value of lambda we will use 10-fold cross validation.

### Lasso
Lasso regression is very similar to ridge regression. They key difference is that instead of lambda being multiplied by the coefficient squared, it is instead multiplied by the magnitude. This importantly means that instead of less valuable features asymptotically reaching 0 with ridge regression, they can be completely nullified to 0 with lasso regression.

Once again we will use 10-fold CV to identify the best value for lambda.

In conclusion, ridge performs well when most variables are useful, and lasso performs well when you have a lot of useless variables. It is unclear which approach will work best in our problem since we have a rather small amount of variables, but some of them seem to have little impact on the compressive strength.

### Elastic Net
If we find that the positives of ridge and lasso are both worthwhile, then perhaps elastic net regression is the way to go. This method combines the positives of the 2 previously mentioned into a nice package.

With this model we'll need to identify to individual values of lambda, lambda and lambda_ with cross validation.

![feat_coefs](imgs/feature_coefs.png)

![coefs_table](imgs/coefs_table.png)

As expected the behaviour between these models is similar since they are all linear regression models. However, some of our previous expectations of coefficients are displayed. The ridge model attempts to reduce the coefficients where possible. The lasso reduces them even further and even brings some down to 0 when necessary. Then finally, the elastic net lands somewhere in the middle.

Its time for the moment of truth, let's see the results of each model's predictions on the test set.

![res_table](imgs/results_table.png)

Although all of our models have performed similarly, it depends on which scoring metric you look at the determine the overall winner. Let's explore the different scoring metrics.

#### Root Mean Square Error (RMSE)
RMSE is the square root of the error function that the regression algorithms are trying to reduce. It is an absolute measure of how well the model fits the data.

#### Mean Absolute Error (MAE)
Similarly to RMSE, MAE looks at the sum of the value of errors. However, since we are not squaring the value and instead taking the absolute value, it is more forgiving to large prediction errors.

#### R Square
R Square measures how much variability in a dependent variable can be explained by the model. It is a good metric to determine the fit on dependent variables. But, it does not take into consideration overfitting.

## Conclusion
Using machine learning models we have simulated the compressive strength of concrete using countless combinations of ingredients and aging process.

Through EDA we have discovered that cement, age and superplasticizer have a positive impact on the overall strength. Water is found to have a negative impact, but is often present in older concrete due to the curing processes.

After utilising linear regression models it has become evident that blast furnace slag is much more important than previously thought, and superplasticizer the opposite. These insights are not intuitive and could be profound and beneficial in a business context.

All of our linear regression models performed similarly in the scoring metrics and would all be suitable in a production environment for making further predictions and testing.

To improve, perhaps decision trees and ensemble methods could be used to increase the prediction score.  

## Deployment
As an extra task I decided to create a web app that uses the best model to test your own concrete samples. For this I used [Streamlit](streamlit.io).

You can access the app [here](https://share.streamlit.io/jamest94/concrete-compressive-strength/package-and-deploy.py) and see below for a screenshot. The sliders allow the user to try different amounts of each ingedient and hit the predict button to see the compressive strength.

![Streamlit Web App](imgs/streamlit.png)

### References

Data: https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength  
Concrete Testing: http://www.civilengineeringforum.me/compressive-strength-test-of-concrete/  
Scikit-learn: https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html  
Streamlit: https://towardsdatascience.com/pycaret-and-streamlit-how-to-create-and-deploy-data-science-web-app-273d205271a3
