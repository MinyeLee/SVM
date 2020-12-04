## Text Categorization with Support Vector Machines : Learning with Many Relevant Features

본 논문은 SVM 을 텍스트 분류 Task 에 적용한 논문으로 1998년에 발표된 논문이다. SVM이 1992년도에 나왔으므로, 그 당시에 새로운 학습 기법을 적용한 것이다. 본 논문에서는 Text Categorization 이라고 말하는데 Text Classification 과 동일한 문제이다. 문서를 정해진 분류와 매칭하는 것이다. 

본 논문에서 Text Categorization Task 에 SVM 을 적용한 이유는 다음과 같다.

- High Dimensional Input Space
- Few Irrelevant Features
- Document Vectors are Sparse
- Most Text Categorization Problems are linearly separable



```python
from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
```

```python
twenty_train.target_names #prints all the categories
print("\n".join(twenty_train.data[0].split("\n")[:3])) #prints first line of the first data file
```

```python
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape
```

```python
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
```

```python
from sklearn.pipeline import Pipeline
```

```python
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)

```

```python
import numpy as np
```

```python
from sklearn.linear_model import SGDClassifier

text_clf_svm = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, random_state=42)),
])

_ = text_clf_svm.fit(twenty_train.data, twenty_train.target)
predicted_svm = text_clf_svm.predict(twenty_test.data)
np.mean(predicted_svm == twenty_test.target)
```

```python

```
