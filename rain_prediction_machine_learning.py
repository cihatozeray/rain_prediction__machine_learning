# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 21:17:48 2020

@author: Cihat Özeray
"""

import pandas as pd
from sklearn.utils import shuffle
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
#import numpy as np

file_name = "otomatik-meteorolojik-gozlem-istasyon-verileri-akom.xlsx"
# sheet_name=None olarak seçildiği zaman .xlsx dosyası içerisindeki tüm sheetleri ayrı
# birer dataFrame olarak bir dictionary içerisinde sunuyor.
dfs_dict = pd.read_excel(file_name, sheet_name=None)

metadata = dfs_dict["Metadata"]
del dfs_dict["Metadata"]

# # headerları silmek gerekti, birleştirmeden önce bir döngüyle halledildi:
for dataframe in dfs_dict:
    dfs_dict[dataframe].drop(0, inplace=True)

# ölçümlerin nerede yapıldığının öneminin olmadığına kanaat getirildi
# ve tüm veriler birleştirildi
df_appended = pd.concat(dfs_dict, ignore_index=True)
df_appended.info()

# nan yönetimi: ilk başta nan içeren tüm satırların atılması düşünüldü 
# ama satırların 3/4'ünün kaybolacağı farkedildi (22999 to 5753). non içeren hücrelerin 
# Min_Yol_S_Z, "Min_Yol_S", Maks_Yol_S_Z, Maks_Yol_S, Ort_Yol_S kolonlarında 
# genellikle yoğunlaştığı gözlemlendi ve bu kolonların etkisi ihmal edilecek
df_appended.drop(columns=["Min_Yol_S_Z", "Min_Yol_S", "Maks_Yol_S_Z", "Maks_Yol_S", "Ort_Yol_S"], \
                inplace=True)


# zaman içeren tüm kolonların droplanmasına karar verildi:
# yine aynı şekilde ActuelIB datasına da anlam verilemedi ve droplanacak:
df_appended.drop(columns=df_appended.columns[[0, 2, 4, 7, 9, 12, 14, \
                    18, 19, 21, 22, 20, 24, 26, 29, 31]], inplace=True)
# gün / ay kolonundan ay' belki tahmin' i iyileştirmek için kullanılabilir
# temizlenip eklenecek: 
df_appended[df_appended.columns[0]] = df_appended[df_appended.columns[0]].str.replace('\d+', '')

df_dummy = pd.get_dummies(df_appended[df_appended.columns[0]])
df_appended.drop(columns=df_appended.columns[0], axis=1, inplace=True)
df_appended = pd.concat([df_appended, df_dummy], axis=1)


# # 2020 verilerinde kayma tespit edildi ve silinecek komple (22999 to 21363):
# df_appended = df_appended[~df_appended["Verinin Yılı"].isin(["2020" ])]
# Yıl kolonunun komple silinmesine karar verildi
# # nan yönetimi: artık nan içeren tüm satırlar atılabilir (21363 to 19747):
df_appended.dropna(how="any", inplace=True) 
# # veri temizliği: "///" içeren hücrelerin bulunduğu satırlar komple atılmalı (19747 to 17421):
for col in df_appended.columns:
    df_appended = df_appended[~df_appended[col].isin(["///"])]

# # kaymalardan dolayı datetime olarak kalanlar olmuş ve ayrı bir döngüyle atılacak:
# # kolaylık olsun diye str dönüşümü yapıldı    
df_appended = df_appended.astype(str) 
# #son kalan datetime'lar da atıldı (17421 to 17420) (sadece 1 satırda kalmış)
for col in df_appended.columns:
    df_appended = df_appended[~df_appended[col].str.contains(":")]

# # ML işlemleri için veri tipi dönüşümü yapıldı 
df_appended = df_appended.astype(float)


# kontrol edildiğinde metadata'da da belirtilen -9999 v.b. anlamsız veriler bulundu
# ve aşama aşama düzeltilecek, filter nan olarak bıraktığı için tekrar nan'lar atıldı
# 17420 to 15798 (final number of rows (around 60% I think))
filter1 = df_appended["Min_Hava_S"]>-100
filter2 = df_appended["Min_Nem"]>-100
filter3 = df_appended["Min_Toprak_S"]>-50
filter4 = df_appended["Min_Toprak_Nem"]>-50
df_appended.where(filter1 & filter2 & filter3 & filter4, inplace = True)
df_appended.dropna(how="any", inplace=True)
desc = df_appended.describe()

# Aşağıdaki sadeleştirmeyi tahmin kalitesini yada en azından hızı arttırmak
# için kullandım ama skor yarı yarıya azaldı ve kullanmaktan vazgeçtim
# df_appended.drop(columns=df_appended.columns[[1, 2, 3, 5, 6, 8, 12, 13, \
#                     14, 16]], inplace=True)


# Aşağıdaki sadeleştirme ile sadece yağış olup olmadığının kontrolü amaçlandı
# Bu işlem sonucu skor iki katına çıktı
df_appended['Top_Yagis'].values[df_appended['Top_Yagis'].values > 0] = 1



#### ML Section ##########################################################
# X, y ayrılması yapıldı


y = df_appended[["Top_Yagis"]]
X = df_appended.drop(["Top_Yagis"], axis=1)

# from sklearn.utils import shuffle

X, y = shuffle(X, y, random_state=7)

# X 'i integer olarak girmek denendi ancak pek bir etkisi olmadı
# for col in X.columns:
#     X[col] = X[col].apply(np.floor)

#offset::
satir_sayisi = y.shape[0]
offset = int(satir_sayisi * 0.7)

X_train = X[:offset]
X_test = X[offset:]
y_train = y[:offset]
y_test = y[offset:]

#model, fit::

# from sklearn.linear_model import LinearRegression
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

coefs = model.coef_
coefs = preprocessing.normalize(coefs)*100

#prediction: y_pred hesaplanacak
y_pred = model.predict(X_test)

# from sklearn import metrics
score = metrics.r2_score(y_test, y_pred)

score_out = "SCORE is  " + str(score)
print(score_out)

# Not: ML bölmesinde derste yazılan kodlar kullanıldı
# ML bölmesi ayrı bir hücre olarak çalıştırılarak zamandan kazanılabilir
# ve farklı regression çeşitleri denenebilir
# (benim denediklerimin sonuca fazla olumlu bir etkisi olmadı)
# Ekim ayının sonuca etkisi çok yüksek gözüküyor. Yağışın yüksek olduğu
# aylar yağış miktarını yönlendiriyor olabilir. Yağışın 0 dan büyük olma durumunu
# 1 olarak kabul ettiğimiz zaman skor iki katına çıkıyor ama hala 0.20 seviyelerinde
