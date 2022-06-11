

import requests
from bs4 import BeautifulSoup  #pip install beautifulsoup4
import time
import re
import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from tkinter import *
import pandas as pd 
import threading
from google_play_scraper import app
appNameList=[]
url=[]
class MyWindow:
    def __init__(self, win):
        self.appNameList=[]
        self.url=[]
        self.lbl1=Label(win,text='Application Name')
        self.lbl1['fg']="blue"
        self.lbl2=Label(win ,text='Application Url')
        self.lbl2['fg']="blue"
        self.lbl3=Label(win,text='RESULT')
        self.lbl4=Label(win,text='')
        # self.f=Frame(win,30)
        # self.f.pack_propagate(0) # don't shrink
   		# self.f.place(x=x,y=y)
   		# self.label=Label(f,*args,**kwargs)
        # self.label.pak(fill=BOTH,expand=1)
        # self.lbl3=self.make_label(self,win,30,40,10,30,text='xxx',background='white')
        self.t1=Entry(bd=3,width=80)
        self.t2=Entry(width=80)
        self.t3=Entry()
        #self.r=win
        #self.r.geometry("400x400")
        self.t3=Text(win,height=50,width=50)
        self.t3.pack()
        self.btn1=Button(win,text='Analysis')
    	#self.btn2=Button(win,text='subtract')
        self.lbl1.place(x=100,y=50)
        self.t1.place(x=240,y=50)
        self.lbl2.place(x=100,y=100)
        self.t2.place(x=240,y=100)
        self.b1=Button(win,text='Analysis', command = self.processinBack)
        self.appNameSpanClassName="AHFaub"
        self.appReviewDateSpanClassName="bAhLNe kx8XBd"
        self.numberOfRatingsCountSpanClassName="AYi5wd TBRnV"
        self.ratingsCountSpanClassName="BHMmbe"
        self.installSpanClassName="htlgb"
        self.appNameSpanClassName="AHFaub"
        self.categorySpanClassName="T32cc UAO9ie"
        self.b1.place(x=500, y=150)
        #self.b2.place(x=200, y=150)
        self.lbl3.place(x=500, y=200)
        self.lbl4.place(x=600, y=200)
        self.t3.place(x=250, y=250)
    def processinBack(self):
        if (str(self.t1.get())!="" or str(self.t1.get())!=""):
            #print("if")
            self.lbl4['text']=""
            self.lbl4['fg']="white"
            download_thread=threading.Thread(target=self.Process)
            download_thread.start()
        else:
            print("else")
            self.lbl4['text']="*Please Enter All Fields"
            self.lbl4['fg']="red"
            
    def Process(self):
        self.b1['state']="disabled"
        self.t3.delete(0.0, 'end')
        self.appNameList.append(str(self.t1.get()))
        self.url.append(str(self.t2.get()))
        totalNumberOfURLs=len(self.url)
        startingPoint=0
        totalURLs=totalNumberOfURLs-startingPoint
        #self.t3.insert(END,("Starting from index: %d | App: %s\n"))
        #(startingPoint, appNameList[startingPoint])
        self.t3.insert(END,("Started Analysing the Training Model\n"))
        PosrevCount=0
        NegrevCount=0
        index=startingPoint-1
        finReviews=[]
        while index<totalNumberOfURLs-1:
            self.t3.insert(END,("Current App: %s\n" %(self.appNameList[index])))
            self.t3.insert(END,("Started getting Reviews\n"))
            index+=1
            startTime=time.time()
            html_doc=requests.get(self.url[index])
            soup=BeautifulSoup(html_doc.content, 'html.parser')
            findId=self.url[index].find('id=')

            url1=self.url[index][findId+3:]
            file = open("cc.txt", "w",encoding='utf-8')
            file.write(str(app(
            url1,
            lang='en', # defaults to 'en'
            country='us' # defaults to 'us'
            )))
            file.close()
            myfile=[]
            with open("cc.txt",encoding='utf8') as mydata:
	            for data in mydata:
		            myfile.append(data)
            start=myfile[0].find('comments')
            end=myfile[0].find('editorsChoice')
            c=data[start:end]
            #print(c)
            #print(soup.prettify())
            x=c
            x=x.split(',')
            print(x)
            k=index+1
            #print("Now Running: App Number - ",k)
        finReviews=x
        timeTaken=time.time()-startTime
        remainingTime=(timeTaken*(totalNumberOfURLs-index))/60
        numberofAppsProcessed=index-startingPoint+1
        self.t3.insert(END,("\n"))
        self.t3.insert(END,("Time Taken to train:%0.1fs| Remaining time:%0.1fm" %(timeTaken,remainingTime)))
        self.t3.insert(END,("\n"))
        self.t3.insert(END,("Completed getting the reviews\n"))
        self.t3.insert(END,("Started given Application Comparing against the Training Model\n"))
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import CountVectorizer
        import pickle
        df=pd.read_csv('training.csv')
        df.head()
        def preprocess_data(df):
            #Remove package name as it's not relevant
            df=df.drop('package_name',axis=1)
            #Convert text to lowercase
            df['review']=df['review'].str.strip().str.lower()
            return df
        df=preprocess_data(df)
        x=df['review']
        y=df['polarity']
        x,x_test,y,y_test=train_test_split(x,y,stratify=y,test_size=0.25,random_state=42)
        vec = CountVectorizer(stop_words='english')
        x = vec.fit_transform(x).toarray()
        x_test = vec.transform(x_test).toarray()
        #Using naive bayes to train model and classification
        from sklearn.naive_bayes import MultinomialNB
        model = MultinomialNB()
        model.fit(x, y)
        model.score(x_test , y_test)
        #print(finReviews)
        #print(model.predict(vec.transform(["Great app. Love the simplicity."])))
        fop=model.predict(vec.transform(finReviews))
        self.t3.insert(END,("\n"))
        self.t3.insert(END,("Completed Analysis"))
        self.t3.insert(END,( "\n"  ) )
        self.t3.insert(END,(fop))
        PosrevCount=0
        NegrevCount=0
        for i in fop:
            if ( i ==1) :
                PosrevCount+=1
            else :
                NegrevCount+=1
        self.t3.insert(END,("\n"))
        self.t3.insert(END,("Negitive Reviews count : "+str(NegrevCount)))
        self.t3.insert(END,( "\n " ))
        self.t3.insert(END,("Positive Reviews count : "+str(PosrevCount)))
        self.t3.insert(END,( "\n " ) )
        self.t3.insert(END,("Negitive Reviews percent : "+str((NegrevCount/(
        NegrevCount+PosrevCount ) *100) ) ) )
        self.t3.insert(END,( " \n " ) )
        self.t3.insert(END,("Positive Reviews percent : "+str((PosrevCount/(
        NegrevCount+PosrevCount ) *100) ) ) )
        self.t3.insert(END,( " \n " ) )
        fig = plt.figure ()
        if PosrevCount >=NegrevCount :
            fig.suptitle(self.appNameList[startingPoint]+" Verdict: This is a good APP")
        else :
            fig.suptitle(self.appNameList[startingPoint]+"Verdict: This is a Fraud/Faulty APP")
        ax = fig.add_axes([0,0,1,1])
        #ax. axis ( ' equal ')
        langs = ['Positive reviews', 'Negetive reviews ']
        students = [ PosrevCount , NegrevCount ]
        ax.pie(students , labels = langs ,autopct='%1.2f%%')
        plt.show()
        self.b1.config( state="normal")
window=Tk ( )
mywin=MyWindow(window)
window . title( 'Detect Fraud App in Google Play Store by Sentiment Analysis')
window . geometry ( "800x600+10+10" )
window . mainloop ( )
    # print (”Please Give Aplication Name:”)
    # appNameList.append(str(input()))
    # print(”Please Give Aplication URL from PLayStore:”)
    # url.append(str(input()))
    # The for loop below runs for each link in the URL list            