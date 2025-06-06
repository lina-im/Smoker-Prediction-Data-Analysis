# Smoker-Prediction-Data-Analysis
Using the “Healthcare Insurance.csv” dataset, apply and compare the three data analysis methods, logistic regression, shallow neural network, and decision trees on the dataset to predict the outcome variable. 

Link: https://www.kaggle.com/datasets/willianoliveiragibin/healthcare-insurance/data

# I. Description
The selected dataset was taken from Kaggle.com, and the file consists of 
7 columns: Age, sex, BMI, children, smoker, and region. While “age,” “bmi”, “children”, 
and “charges” are numerical, “sex,” “smoker,” and “region” are categorical values. The 
“age” column lists the insured person’s age. The “sex” column lists the gender of the 
insured individual. The “bmi” (Body Mass Index) column lists the measure of body fat 
based on height and weight. The “children” column lists the number of dependents 
covered. The “region” column lists the geographic area of coverage 
(southwest/southeast, northwest/northeast). The “charges” column lists the medical 
insurance costs incurred by the insured person. Here, the “smoker” column lists 
whether the insured is a smoker (yes/no), which serves as the output variable to be 
used to measure the accuracy of the prediction. 

# II. Brief Description of the Three Methods Used 
**Logistic regression** is one of the methods used, and it is a classification model that 
predicts binary outcomes (yes/no). It applies a sigmoid function to estimate the 
probability of an event falling into a specific category (between 0 and 1) by fitting a 
linear equation to the dataset. Values above the threshold indicate 1 (yes) and values 
below the threshold indicate 0 (no). This probability demonstrates the likelihood of an 
event belonging to a specific category (yes/no).  

**Shallow neural network** is another method used, and this is an artificial neural 
network with typically one or very few hidden layers for simple tasks such as 
classification. The input layer receives data, the hidden layer(s) processes it through 
activation functions, such as ReLU or sigmoid, and the output layer produces the final 
classification (yes/no).  

**Decision tree** is the third method used, and this acts as a series of if-then conditions. 
Based on a decision rule through the Gini Impurity calculation, branches are created to 
represent possible outcomes with the leaves representing the final classification 
(yes/no).

# III. Experimental Results  
_Logistic Regression Results:_

![1](https://github.com/user-attachments/assets/add80650-9a4e-452e-b24f-1879ddb635a5)

**Figure 1:** The confusion matrix after performing logistic regression shows that the 
model predicted 208 correct “no” (non-smokers) and 51 correct “yes” (smokers) as the 
final classification, achieving a model accuracy of **96.64%**.

_Shallow Neural Network (SNN) Results:_  

![2](https://github.com/user-attachments/assets/b2ebde32-8d72-476b-bf59-cf1ff58b7727)

**Figure 2:** The confusion matrix after performing SNN shows that the model predicted 
207 correct “no” (non-smokers) and 53 correct “yes” (smokers) as the final 
classification, achieving a model accuracy of **97.01%**. 

_Decision Tree Results:_ 

![3](https://github.com/user-attachments/assets/5902bafc-9c20-4059-b7ed-3ee34321c1d0)

**Figure 3:** The confusion matrix after performing decision tree shows that the model 
predicted 206 correct “no” (non-smokers) and 51 correct “yes” (smokers) as the final 
classification, achieving a model accuracy of **95.90%**. 

# IV. Discussion of Results 
When comparing the model accuracy with the three different methods, the model used 
with SNN displayed the highest accuracy of 97.01%. The second highest accuracy was 
the model used with logistic regression with an accuracy of 96.64%, and the lowest 
accuracy was the model used with a decision tree with an accuracy of 95.90%. 

When using logistic regression model, the target variable, “smoker,” was converted into 
binary, with “yes” representing “1” and “no” representing “0”. Then, the dataset was 
split into training (80%) and testing (20%) sets before the logistic regression model was 
applied for training. In my experimental results, it was found that the logistic regression 
model displayed a high accuracy. This indicates the model accurately predicted the 
binary outcome (yes/no) based on the 6 input variables. This could be due to the model 
showing a linear relationship or a pattern between the input variables and the output 
variable, which is ideal for a logistic regression model. To improve this model, k-fold 
cross-validation could be implemented to allow a data point in the training set to 
appear more than once to prevent overfitting in data. This implementation could also 
provide a more reliable evaluation by taking the average accuracy across all k
iterations.  

When using SNN model, I used one hidden layer with 20 neurons after standardizing the 
input features and dividing the dataset into training (80%) and testing (20%) sets. 
Although training this model took more time than the logistic regression model, this 
model showed a higher accuracy by 0.0037. The SNN model can process more complex 
and non-linear relationships between the input and target variables, so it can learn the 
behavior of the dataset more flexibly which yields to higher accuracy. To improve this 
model, dropout regularization can be implemented to reduce overfitting. By temporarily 
disabling a random fraction of neurons during each training iteration, the model is 
prevented from relying on specific neurons. This prevents the model to “memorize” 
certain patterns in data that could contribute to overfitting. 

When using decision tree model, I used Gini Impurity for splitting nodes within the tree 
after standardizing and splitting the dataset into training (80%) and testing (20%) sets. 
The Gini Impurity was used to grow the tree until all nodes were “pure” (1 label: yes/no) 
or the node contains a low number of data points. The training dataset begins at the 
root node and splits into nodes recursively that minimizes the Gini Impurity calculation. 
Based on the experimental results, this model achieved the lowest accuracy. This could 
indicate the decision tree grew deep, leading to higher time complexity, and overfitting. 
To improve this model, pruning can be implemented to reduce the depths of decision 
tree to further simplify the model. By removing subtrees that contribute to overfitting 
and decreases the model’s accuracy, the model’s variance is reduced which makes it 
more generalizable. 
