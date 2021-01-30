import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import threading
import nltk
import nltk as nlp
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn import model_selection
from sklearn import preprocessing, ensemble
from sklearn.feature_extraction.text import TfidfVectorizer

# =============================================================================
# Section 1 : FirstVariables
# =============================================================================
class firstVariable:
    headers_param = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36"}
    r = requests.get(
        "https://www.hepsiburada.com/xiaomi-redmi-airdots-tws-bluetooth-5-0-kulaklik-p-HBV00000K41EM-yorumlari?sayfa=1&sadeceonayli=evet",
        headers=headers_param)
    soup = BeautifulSoup(r.content, "lxml")
    reviews = soup.find_all("div", attrs={"class": "hermes-ReviewCard-module-34AJ_"})
    column_names = ["Reviews", "Rating"]
    dataSet = pd.DataFrame(columns=column_names)
    clearCounter = 0
    pageSize = int(
        soup.find("ul", attrs={"class": "hermes-PaginationBar-module-3qhrm hermes-PaginationBar-module-1ujWo"}).select(
            "li:nth-of-type(9) > span")[0].text)
    lemma = nlp.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('turkish')
    stopwords.append("fakat")
    stopwords.append("herkes")
# urun kelimesi bir stop word olmamasina karsi ekledim cunku her cumlede fazlası ile geçiyor ve yanlış sonuç oluşturuyor
    stopwords.append("urun")
    tr2Eng = str.maketrans("çğıöşü", "cgiosu")

# =============================================================================
# Section 2 : WebScraping
# =============================================================================
def webScraping(first, last):
    for i in range(first, last):
        try:
            url = "https://www.hepsiburada.com/xiaomi-redmi-airdots-tws-bluetooth-5-0-kulaklik-p-HBV00000K41EM-yorumlari?sayfa=" + str(
                i) + "&sadeceonayli=evet"
        except:
            print(str(i) + " sayfa numaralı url'ye erişilemiyor")
        try:
            r = requests.get(url, headers=firstVariable.headers_param)
        except:
            print("Url'ye İstek Atarken Bir Hata Meydana Geldi")
        try:
            soup = BeautifulSoup(r.content, "lxml")
            firstVariable.reviews = firstVariable.reviews + soup.find_all("div", attrs={
                "class": "hermes-ReviewCard-module-34AJ_"})
            print(str(i) + " numaralı sayfa başarı ile parçalandı")
        except:
            print("Aranılan http modülüne ulaşılamıyor")

# =============================================================================
# Section 3: Threading for WebScraping
# =============================================================================
perLoopForThread = int(firstVariable.pageSize / 8)

t1 = threading.Thread(target=webScraping, args=(0, perLoopForThread))
t2 = threading.Thread(target=webScraping, args=(perLoopForThread, 2 * perLoopForThread))
t3 = threading.Thread(target=webScraping, args=(2 * perLoopForThread, 3 * perLoopForThread))
t4 = threading.Thread(target=webScraping, args=(3 * perLoopForThread, 4 * perLoopForThread))
t5 = threading.Thread(target=webScraping, args=(4 * perLoopForThread, 5 * perLoopForThread))
t6 = threading.Thread(target=webScraping, args=(5 * perLoopForThread, 6 * perLoopForThread))
t7 = threading.Thread(target=webScraping, args=(6 * perLoopForThread, 7 * perLoopForThread))
t8 = threading.Thread(target=webScraping, args=(7 * perLoopForThread, firstVariable.pageSize))

t1.start()
t2.start()
t3.start()
t4.start()
t5.start()
t6.start()
t7.start()
t8.start()

t1.join()
t2.join()
t3.join()
t4.join()
t5.join()
t6.join()
t7.join()
t8.join()

# =============================================================================
# Section 4: Get reviews to list and add to dataSet
# =============================================================================
for review in firstVariable.reviews:
    try:
        reviewText = review.find("span", attrs={"itemprop": "description"}).text
        tempRate = len(review.find_all("path", attrs={"fill": "#f28b00"}))
        if tempRate >= 3:
            reviewRate = 1
        elif tempRate < 3:
            reviewRate = 0
    except:
        reviewText = "NULL"
        reviewRate = "NULL"
    firstVariable.dataSet = firstVariable.dataSet.append({'Reviews': reviewText, 'Rating': reviewRate},
                                                         ignore_index=True)

firstVariable.dataSet = firstVariable.dataSet[firstVariable.dataSet.Reviews != 'NULL']
firstVariable.dataSet = firstVariable.dataSet[firstVariable.dataSet.Rating != '3']

# =============================================================================
# Section 5: Clear the data
# =============================================================================
firstVariable.clearCounter += 1
firstVariable.dataSet['Reviews'] = firstVariable.dataSet['Reviews'].apply(
        lambda x: " ".join(word.lower() for word in x.split()))
firstVariable.dataSet['Reviews'] = firstVariable.dataSet['Reviews'].str.translate(firstVariable.tr2Eng)
firstVariable.dataSet['Reviews'] = firstVariable.dataSet['Reviews'].str.replace("[^\w\s]", "")
firstVariable.dataSet['Reviews'] = firstVariable.dataSet['Reviews'].str.replace("\d", "")
firstVariable.dataSet['Reviews'] = firstVariable.dataSet['Reviews'].apply(
        lambda x: " ".join(word for word in x.split() if word not in firstVariable.stopwords))
firstVariable.dataSet['Reviews'] = firstVariable.dataSet['Reviews'].apply(
        lambda x: " ".join(firstVariable.lemma.lemmatize(word) for word in x.split()))

# =============================================================================
# Section 6: Import dataSet to desktop
# =============================================================================

firstVariable.dataSet.to_csv(r'C:\Users\90543\Desktop\export_dataframe.csv', index=False, header=True)

# =============================================================================
# Section 7: Split dataSet as train and test
# =============================================================================
train_x,test_x,train_y,test_y = model_selection.train_test_split(firstVariable.dataSet["Reviews"],firstVariable.dataSet["Rating"],random_state = 1)

encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)

# =============================================================================
# Section 8: Create TF-IDF Vector on NGRAM Level
# =============================================================================

vectorizerTfIdf_ngram = TfidfVectorizer(ngram_range = (1,2))
vectorizerTfIdf_ngram.fit(train_x)

vectorizerTfIdf_ngram.get_feature_names()[0:5]

train_x_tfIdf_ngram = vectorizerTfIdf_ngram.transform(train_x) 
test_x_tfIdf_ngram  =  vectorizerTfIdf_ngram.transform(test_x)  

# =============================================================================
# section 9: Create model with Random Forest Algorithm and get Accuracy
# =============================================================================
rf = ensemble.RandomForestClassifier()
rf_model = rf.fit(train_x_tfIdf_ngram,train_y)
accuracy = model_selection.cross_val_score(rf_model,test_x_tfIdf_ngram,test_y,cv = 10).mean()

print("TF Vectors Accuracy Oranı:",accuracy)

# =============================================================================
# Test Section
# =============================================================================

testVariable = "bu urun kotu olabilir fakat guzel kisimlarida yok degil"
iyi = rf_model.predict(vectorizerTfIdf_ngram.transform([testVariable]))

testVariable = "ben bu urunu tavsiye etmiyorum"
kotu = rf_model.predict(vectorizerTfIdf_ngram.transform([testVariable]))