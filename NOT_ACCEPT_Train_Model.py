import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import OneClassSVM
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
import joblib
# Example Yageo part numbers
df=pd.read_csv(r"500K_Sample_Vishay_CM_Parts.tsv",sep='\t',encoding='ISO-8859-1')
df.head()
#Vishay_parts_sample=df.sample(10000)
Vishay_parts = df["COM_PARTNUM"].to_list()

# Vectorize the part numbers using TF-IDF
vectorizer = TfidfVectorizer(analyzer='char',ngram_range=(3,3),lowercase=True)
print("before_vectorizer")
X_train = vectorizer.fit_transform(Vishay_parts)
print(X_train.shape)
print("after_vectorizer")
# Create and train the One-Class SVM model
oc_svm = OneClassSVM(kernel='linear', gamma='auto', nu=0.1,verbose=True)
start_time=time.time()
oc_svm.fit(X_train)
print(f"Elapsed Time is {time.time()-start_time} Seconds")
joblib.dump(oc_svm,"SVM_NOTACCEPT_MODEL.pkl")
# Example new part numbers
new_parts = [
    'R234-5678-90', 'B123-4567-89', 'C890-9876-54', 
    'Z123-4567-89', 'R456-7890-12'
]

# Vectorize the new part numbers
X_test = vectorizer.transform(new_parts)

# Predict using the trained One-Class SVM model
predictions = oc_svm.predict(X_test)

# Interpret the predictions
for part, pred in zip(new_parts, predictions):
    if pred == 1:
        print(f"{part} related to Vishay.")
    else:
        print(f"{part} NOT related to Vishay.")