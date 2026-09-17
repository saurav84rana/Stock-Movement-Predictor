# Model choice explanation

# Chosen models:

## Logistic Regression: Linear model
Logistic Regression (with default of L2 regularisation) was chosen because it is a simple linear and interpretable model. It uses negative log likelihood loss to get the best weights (coefficients) whiles having the smallest error during training between its predictions and real results. 

$F(x) = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n$\, where:

$F_n(x)$: the whole model,\
$w_0$: the intercept,\
$w_n$: weight for feature n,\
$x_n$: feature n

Then, a sigmoid function is used to squash its prediction in range 0-1, $sigmoid(F(x))$.\
Default, the threshold is:\
class = 0, if sigmoid() < 0.5, and \
class = 1, if sigmoid() >= 0.5

And, the model can be interpreted fairly easily, doing `.coef_` will show all weights, which can be mapped to actual features, to see which features are being considered the most by the model.

## Gradient boosting: Non linear model
Gradient boosting sequentially adds shallow trees, where each tree is trained on the residual (for classification, this is the pseudo residual, the negative gradient of log loss) of the previouses (ensemble) instead of the target itself. The predction of the first tree, $F_0$, is simply a constant (log odds of class prior of the training data), where trees thereafter will train on the pseudo residual.

$F_n(x) = F_{n-1}(x) + \mu h_n(x)$, where:

$F_n(x)$: the whole model at tree n,\
$F_{n-1}(x)$: the whole model at tree n-1,\
$\mu$: step size, e.g. 0.1,\
$h_n(x)$: $n^{th}$ tree

This lets us map non linear features which logistic regression would otherwise not be able to capture.


# Other popular models, and why they weren't selected:

## Naive Bayes
Naive Bayes finds $P(class = target|X)$ via bayes theorem ("given dataset X, how likely is it for its class to be `target`?), and fundamentally relies on the assumption that all features are conditionally independent of eachother. But the features I used are not.

The features I engineered include lookback period, for example, `return_*d`. If `return_20d` has massive returns, then theres likely some relationship with `return_10d` or `return_5d`, because they share the same or related properties of price history.

Also, adding `stoch_k_14d`, `rsi_14d` into the example, and these are all momentum indicators, capturing the same signal from different angles, which means they are correlated by design.

In other use cases, it is still used despite features never being completely independent, and Naive Bayes still works quite well. But for the case of stock movement predictor, it is not a strong suite especially given upfront we know features are correlated.

## SVM (Support Vector mechanism)
SVM finds a decision boundary that seperates a class whiles making the margin as wide as possible.\
However a linear SVM decision boundary can look similar to that of logistic regression's despite having different optimisation goal.

You could use kernel SVM, which can map non linear behaviour, but that includes hyperparameter tuning (`C`/`gamma`), and that it can be very sensitive to them, which may require a seperate focused section. 

Additionally, at around 47k rows, for 5 folds, times however many hyperparemeters combination to train, and that sklearn's SVC's `predict_proba()` performs additional cross validations, means each fit can get slow enough the workflow is not as smooth anymore.

Considering there are other faster yet powerful non linear models, like decision tree based ones, such as random forest and gradient boosting, others were favoured over SVM.


## RF (Random Forest)
A set of decision trees running in parallel, each with bootstrap samples, meaning each tree is different. Then for prediction, run that sample through all of the trees, and get a majority vote of what class it should be. \
About 37% of the samples are mathematically never chosen during bagging (used during training/construction of trees), which means you can train using the whole dataset and use that 37% OOB (out of bag) data to test/validate.

This is an excellent rival with GD (gradient boosting), because they can both capture non linear relationships of the features. 

However I chose to prefer GD over RF. GD's sequential and error correcting structure, tends to extract slightly more signal when they are faint, compared to RF's bagging parallel averaging with no knowledge of eachother's errors, especially in large data structures. This is useful in the stock universe, where we know real signal is faint and noise is high.
