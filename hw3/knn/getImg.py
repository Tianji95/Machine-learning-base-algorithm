import urllib2,urllib,time


aimUrl = "http://jwbinfosys.zju.edu.cn/CheckCode.aspx"
def saveImg(imageURL,fileName):
    
     u = urllib.urlopen(imageURL)
     data = u.read()
     f = open(fileName, 'wb')
     f.write(data)
     f.close()

for i in range(300):
    print saveImg(aimUrl, "images/"+str(i+1)+".png")
    time.sleep(0.1)