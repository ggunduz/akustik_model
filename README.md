#### Kurulum
Gerekli olan tensorflow ve tensorflow-io kutuphaneleri kurulmalidir
```
pip install -r requirements.txt
```
gerekli kutuphaneler kurulduktan sonra
```
python cnn_inference.py
```
komutu ile `./data/` klasoru altindaki 6 adet ornek ses dosyasi icin tahminleme yapilir
___
#### Cikti
['./data/dog_1.wav', 'hayvan']
['./data/dog_2.wav', 'hayvan']
['./data/human_english.wav', 'insan']
['./data/human_english2.wav', 'insan']
['./data/engine_1.wav', 'arac']
['./data/engine_2.wav', 'arac']

___

