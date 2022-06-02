#!/usr/bin/python3
import requests,json
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
#http_proxy  = "http://127.0.0.1:10809"
#https_proxy = "https://127.0.0.1:10809"#请打开v2ray
#proxyDict = { 
              #"http"  : http_proxy, 
              #"https" : https_proxy, 
           # }
url="https://api.coincap.io/v2/assets/bitcoin/history?interval=m30"
#history=requests.get(url, proxies=proxyDict)
history=requests.get(url)
history=history.text
print(history)
data=json.loads(history)
y=[]
dataprice=data['data']
for i in dataprice:
    priceUSD=i['priceUsd']#每个比特币价格
    #print(priceUSD)
    priceUSD=json.loads(priceUSD)
    y.append(int(priceUSD))
#y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
x = list(range(1,len(y)+1,1))
mymodel = numpy.poly1d(numpy.polyfit(x, y, 15))
myline = numpy.linspace(1, len(y)+3, 10000)
#print("拟合度:",r2_score(y, mymodel(x)))
plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
biaozhuncha = numpy.std(y)
print("标准差:",biaozhuncha)
fangcha=numpy.var(y)
print("方差:",fangcha)
print("这个值应该大于,",len(x))
print("最近一次统计的数值是:",y[-1])
ask=int(input("请输入预测的值"))
speed = mymodel(ask)
print("根据统计的",len(y),"个数据,我们预估第",ask,"大概是",speed,"左右")
print("多项式回归的拟合度为:",r2_score(y, mymodel(x)))
if  speed>y[-1]:
    bili=(speed/y[-1])
    print("可能会涨",bili)
else:
    bili=(1-(speed/y[-1]))
    print("可能会跌",bili)
plt.show()
