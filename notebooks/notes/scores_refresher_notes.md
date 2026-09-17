# Evaluation of models predictions

## The fundamentals
| | Actually 1 | Actually 0 |
| --- | :---: | :---: |
| Predicted 1 | TP | FP |
| Predicted 0 | FN | TN |

TP: True positive, "The 1 you predicted, is correct"\
FP: False positive, "The 1 you predicted, isn't correct"\
FN: False negative, "The 0 you predicted, isn't correct"\
TN: True negative, "The 0 you predicted, is correct"

Precision: look at row Predicted 1\
Recall: look at column Actually 1\
Specificity: look at column Actually 0\
Accuracy: everything

## Accuracy:
Accuracy = $\frac{TP+TN}{TP+TN+FP+FN}$

or $\frac{\text{Correct guesses}}{\text{All guesses/dataset size (row)}}$

or "Of all my guesses, what proportion was guessed correctly?"

But a useless model that predicts everything `1`, on dataset where only 1% is `0`, will score 99% accuracy. That's awful


## Precision: 
"When I say yes, am I usually right?"
$Precision = \frac{TP}{TP+FP}$

or $\frac{\text{Correctly guessed 1s}}{\text{All guessed 1s}}$

or "Of all the `1`s I predicted, how many were actually guessed correctly?"

For high precision you are trying to minimise FP (predicted spam but is actually legit)

E.g. spam detection, high precision = don't make many false positive. 

## Recall/TPR (True positive rate): 
"Did I find most of the actual yes cases?"

$recall = \frac{TP}{TP+FN}$

or $\frac{\text{Correctly guessed 1s}}{\text{All actual 1s}}$

or "Of all the actual `1`s, how many did I find/how many of them did I correctly guess as `1`?"

For high recall you are trying to minimise FN (predicted not cancer, but is actually cancer)

E.g. medical screening, a patient with cancer, but model says they dont have cancer, can be catastrophic. High recall = dont make false negatives (dont miss out ill patients)


## Specificity/TNR (True negative rate): 
"Did I find most of the actual no cases?"

$specificity = \frac{TN}{TN+FP}$

or $\frac{\text{Correctly guessed 0s}}{\text{All actual 0s}}$

Mathematically you are trying to minimise FP (predicted up but is actually down)

"Of all the actual `0`s, how many did i correctly reject/correctly guess as `0`?"

Also, False Positive Rate = 1 - Specificity

## Example: fraud detector
Model A: Recall, Precision = 10%, 95%\
Model B: Recall, Precision = 95%, 40%

Model A almost never catches actual fraudulent transactions, but it also almost never false accuses innocent customers

Model B catches most fraudulent transactions, but it also flags a lot of legit transactions 

There is no best, it only depends on what you want.


## F1 score
We want both precision and recall to be high (harmonic mean of precision and recall)

$F1 = 2 \cdot \frac{\text{precision} \times \text{recall}}{\text{precision + recall}}$

if precision and recall is high, F1 is high   (high)/(very high)\
if precision and recall is low, F1 is low  (very low)/(low-medium)

if precision is high, recall is low, then F1 is low, (very low)/(high)
same for vice versa


## F2 score
Like F1, but gives more importance to recall. "Did I find most of the actual yes cases?"

$F\beta$ = $\frac{(1+\beta^2) \times \text{precision} \times \text{recall}}{(\beta^2 \times \text{precision}) + \text{recall}}$

"I will sacrifice precision if that means catching more actual positives"

For F1:
$\beta = 1$, turns into normal F1 formula\
For F2:
$\beta = 2$

## F0.5 score
$\beta = 0.5$
Like F1, but gives more importance to precision

"I will sacrifice recall if that means im more precise (less wrong `1` guesses)"


## sklearn metrics
it goes class by class (support = num items in that class)

macro avg: equally weighted average of all performance metrics\
weighted avg: if class 0 is 90% of data, and class 1 is 10%, then times that by class' performance before adding and dividing for mean, giving most weight (0.9) to class 0 because its got more candidates.


## ROC curve
"How likely is the model's score on a randomly selected positive example > score on randomly selected negative example?"

`model.predict_proba(X_test)` gives
```txt
Email A: 0.97 spam
Email B: 0.82 spam
Email C: 0.31 spam
Email D: 0.02 spam
```
The default threshold = 0.5, \
\>= threshold -> spam\
<  threshold -> not spam

You can vary that threshold. It changes precision, recall, false positive rates, etc.

ROC curve plots what happens as you vary that threshold, it plots TPR vs FPR:

Very high threshold: predict almost everything negative.
Very low threshold: predict almost everything positive.

True Positive Rate = Recall (TP/TP+FN).\
Can't have TP if you dont let it predict positive (0/FN), and can't have FN if you dont let it predict negative (TP/TP+0).

False Positive Rate = 1 - Specificity, or 1-(TN/TN+FP), so 1-1=0 at threshold=1, 1-0=1 at threshold=0


## ROC-AUC (area under the curve)
In range 0-1:\
<0.5: the model is backwards\
0.5 = random guessing\
0.8 = good\
1.0 = perfect ranking on evaluated data


## PR-AUC (Precision-Recall AUC)
For catching rare events like fraud transactions detection, model can have very good ROC-AUC while still having high false positives. Not our area of interest.
