'''Author: Rita Lin 2022/04/07 特別感謝 Sean Lee 協助debug'''

#從Stack Overflow 提取問題並找到最佳解答, 讓我們來看看如何在NLP上使用CNN模型? (這是個好問題)
from bs4 import BeautifulSoup
from urllib.request import urlopen

#要爬的網址
url = "https://stackoverflow.com/questions/62964559/"
html = urlopen(url).read()
soupified = BeautifulSoup(html,"html.parser")

#question 一開始結果有"-", 利用"-"符號做切割成三個部分
question = soupified.find("title").text.split("-")[1].strip()

#希望印出Title並加上Title內容
print("title:", question)


#這邊檢查網頁原始碼發現 Question和Answer的div後面那串長的一樣! [0]是Question [1]是Answer
text_2 = soupified.find_all("div",{"class":"s-prose js-post-body"})

questiontext = text_2[0]

#希望印出Question字樣並加上Question內容
print("Question: \n", questiontext.text.strip())

print('------------------------------------------------------\n') #我是分隔線


#希望印出Answer字樣並加上Answer內容
answer = text_2[1]
print("Answer: \n", answer.text.strip())

'''
result>>>
title: How to fit NLP in CNN model?
Question: 
 I am doing research on using CNN machine learning model with NLP (multi-label classification)
I read some papers that mentioned getting good results in applying CNN for multi-label classification
I am trying to test this model on Python.
I read many articles about how to work with NLP an Neural Networks.
I have this code that is not working and giving me many errors ( every time I fix the error I get another error )
I ended seeking paid FreeLancers to help me fix the code, I hired 5 guys but non of them was able to fix the code !
you are my last hope.
I hope someone can helpe me fix this code and get it working.
First this is my dataset (100 record sample, just to make sure that code is working, I know it is not enogh for good accuracy. I will tweak and enhance model later)
http://shrinx.it/data100.zip
at the time being I just want this code to work. yet tips on how to enhance accuracy are really welcomed.
Some of the errors I got
InvalidArgumentError: indices[1] = [0,13] is out of order. Many sparse ops require sorted indices.
    Use `tf.sparse.reorder` to create a correctly ordered copy.

and
ValueError: Input 0 of layer sequential_8 is incompatible with the layer: expected ndim=3, found ndim=2. Full shape received: [None, 18644]

here is my code
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import *



# Load Dataset

df_text = pd.read_csv("J:\\__DataSets\\__Samples\\Test\\data100\\text100.csv")
df_results = pd.read_csv("J:\\__DataSets\\__Samples\\Test\\data100\\results100.csv")

df = pd.merge(df_text,df_results, on="ID")


#Prepare multi-label
Labels = [] 

for i in df['Code']: 
  Labels.append(i.split(",")) 


df['Labels'] = Labels



multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(df['Labels'])

y = multilabel_binarizer.transform(df['Labels'])
X = df['Text'].values

#TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=1000)


xtrain, xval, ytrain, yval = train_test_split(X, y, test_size=0.2, random_state=9)

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=1000)

# create TF-IDF features
X_train_count = tfidf_vectorizer.fit_transform(xtrain)
X_test_count = tfidf_vectorizer.transform(xval)


#Prepare Model

input_dim = X_train_count.shape[1]  # Number of features
output_dim=len(df['Labels'].explode().unique())


sequence_length = input_dim
vocabulary_size = X_train_count.shape[0]
embedding_dim = output_dim
filter_sizes = [3,4,5]
num_filters = 512
drop = 0.5

epochs = 100
batch_size = 30



#CNN Model

inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
output = Dense(units=2, activation='softmax')(dropout)


# this creates a model that includes
model = Model(inputs=inputs, outputs=output)


#Compile

checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
print("Traning Model...")
model.summary()


#Fit
model.fit(X_train_count, ytrain, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=(X_test_count, yval))  # starts training



#Accuracy
loss, accuracy = model.evaluate(X_train_count, ytrain, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test_count, yval, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

a sample of my dataset
text100.csv
ID  Text
1   Allergies to Drugs  Attending:[**First Name3 (LF) 1**] Chief Complaint: headache and neck stiffne
2   Complaint: fever, chills, rigors  Major Surgical or Invasive Procedure: Arterial l
3   Complaint: Febrile, unresponsive--> GBS meningitis and bacteremia  Major Surgi
4   Allergies to Drugs  Attending:[**First Name3 (LF) 45**] Chief Complaint: PEA arrest .   Major Sur
5   Admitted to an outside hospital with chest pain and ruled in for myocardial infarction.  She was tr
6   Known Allergies to Drugs  Attending:[**First Name3 (LF) 78**] Chief Complaint: Progressive lethargy 
7   Complaint: hypernatremia, unresponsiveness  Major Surgical or Invasive Procedure: PEG/tra
8   Chief Complaint: cough, SOB  Major Surgical or Invasive Procedure: RIJ placed Hemod

Results100.csv
ID  Code
1   A32,D50,G00,I50,I82,K51,M85,R09,R18,T82,Z51
2   418,475,905,921,A41,C50,D70,E86,F32,F41,J18,R11,R50,Z00,Z51,Z93,Z95
3   136,304,320,418,475,921,998,A40,B37,G00,G35,I10,J15,J38,J69,L27,L89,T81,T85
4   D64,D69,E87,I10,I44,N17
5   E11,I10,I21,I25,I47
6   905,C61,C91,E87,G91,I60,M47,M79,R50,S43
7   304,320,355,E11,E86,E87,F06,I10,I50,I63,I69,J15,J69,L89,L97,M81,N17,Z91
------------------------------------------------------

Answer: 
 I don’t have anything concrete to add at the moment, but I found the following two debugging strategies to be useful for me:

Distill your bugs into different sections. For e.g which errors are related to compiling models and which related to training? There could be errors before the model. For the errors that you showed, when did they first raise? Its kind of hard to see without line number and etc.

This step is useful personally as sometimes later errors are manifestation of earlier ones, so sometimes 50 errors might be just 1-2 at the beginning stage.

For a good library, typically their error messages are helpful. Have you tried what the error messages suggest and how did that go?
'''

#Rita Lin 2022/04/07 特別感謝Sean Lee
