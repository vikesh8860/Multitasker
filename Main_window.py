import os
import re
import cv2
import sys
import vlc
import data
import time
import math
import shutil
import smtplib
import threading
import numpy as np
import language_check
import matplotlib.pyplot as plt
import speech_recognition as sr
from PyQt4 import Qt
from gtts import gTTS
from pygame import mixer
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from threading import Thread
from autocorrect import spell
from PyQt4 import QtGui,QtCore
from collections import Counter
from email.mime.text import MIMEText
from sklearn.externals import joblib
from PIL import Image,ExifTags,ImageOps
from PyPDF2 import PdfFileMerger, PdfFileReader

sys.setrecursionlimit(10000)

####################################################################################################        
#####   This is the main class that control all the subclass and used for layout switching     #####        
####################################################################################################
class App_Window(QtGui.QMainWindow):

    #---------------------------------------------------------------------
    # Init method for our App_Window class
    def __init__(self, parent=None):    
        super(App_Window, self).__init__(parent)
        self.setWindowTitle(data.app_name)                     #App Title
        self.setWindowIcon(QtGui.QIcon(data.app_logo_image))   #App Icon
        
        ##Set the background of window as an image
        back_img = Image.open(data.background_image)
        width , height = back_img.size
        self.setFixedSize(width,height)
        palette 	= QtGui.QPalette()
        palette.setBrush(QtGui.QPalette.Background,QtGui.QBrush(QtGui.QPixmap(data.background_image)))
        self.setPalette(palette)


        self.central_widget = QtGui.QStackedWidget()
        self.setCentralWidget(self.central_widget)
        main_window_widget = Main_window(self)
        self.central_widget.addWidget(main_window_widget)




    #----------------------------------------------------------------------
    # This method is called when back button is pressed in any other window
    def back_button(self):
        #The below line will set the user interface to the main window
        self.central_widget.setCurrentIndex(0)



        
    #----------------------------------------------------------------------
    #Processing of the photo ocr operation
    def photo_ocr(self):
        photo_ocr_widget = Photo_ocr_class(self)
        self.central_widget.addWidget(photo_ocr_widget)
        self.central_widget.setCurrentWidget(photo_ocr_widget)
        



    #----------------------------------------------------------------------
    #Processing of the pdf_scanner operation
    def pdf_scanner(self):
        pdf_scanner_widget = Pdf_scanner_class(self)
        self.central_widget.addWidget(pdf_scanner_widget)
        self.central_widget.setCurrentWidget(pdf_scanner_widget)




    #----------------------------------------------------------------------
    #Processing of the speech_to_text operation
    def speech_to_text(self):
        speech_to_text_widget = Speech_to_text_class(self)
        self.central_widget.addWidget(speech_to_text_widget)
        self.central_widget.setCurrentWidget(speech_to_text_widget)




    #----------------------------------------------------------------------
    #Processing of the text_to_speech operation
    def text_to_speech(self):
        text_to_speech_widget = Text_to_speech_class(self)
        self.central_widget.addWidget(text_to_speech_widget)
        self.central_widget.setCurrentWidget(text_to_speech_widget)



    
    #----------------------------------------------------------------------
    #Information about the developers
    def about_us(self):
        about_us_widget = About_us_class(self)
        self.central_widget.addWidget(about_us_widget)
        self.central_widget.setCurrentWidget(about_us_widget)




    #----------------------------------------------------------------------
    ## Dialog Box to check whether the user really wants to quit
    def closeEvent(self,event):
        reply = QtGui.QMessageBox.question(self,data.app_name,'Are you sure to quit  ' +data.app_name +' ?',QtGui.QMessageBox.Yes|QtGui.QMessageBox.No,QtGui.QMessageBox.No)
        if reply == QtGui.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
            

####################################################################################################        
#########                      Class App_Window ends here                             ##############        
#################################################################################################### 





            
####################################################################################################        
#####         Class containing  the functionality of Photo_ocr option                 ##############        
####################################################################################################
class Photo_ocr_class(QtGui.QWidget):



    image_name      = ''       #Stores the name of image to be loaded
    open_cv_image   = ''       #Used to store the image that will be shown in the QLabel
    original_image  = ''       #Used for reset purpose
    temp_image      = ''       #For syncing purpose between adaptive threshold and simple threshold
    adaptive_thresh_value = 25
    normal_thresh_value = 70
    flag = 0              #Indicate that image has been rotated
    inner_flag  = 0       #Indicate that image has been thresholded
    rotation  = 0
    left_rot = 0
    right_rot = 0
    im_copy2  = 0
    cc2 = 0
    rr1 = 0
    rr2 = 0
    
    def __init__(self, parent=None):
        super(Photo_ocr_class, self).__init__(parent)
        mainWindow = QtGui.QWidget()
        self.initUI_Photo_ocr_class()

    def initUI_Photo_ocr_class(self):

        #Font for the text in the Pdf_scanner window
        newFont = QtGui.QFont(data.font_helvetica,data.font_size,QtGui.QFont.Normal)


        #--------------------------------------------------------------------------------------
        self.back_button = PicButton(QtGui.QPixmap(data.back_button_unpressed_image),QtGui.QPixmap(data.back_button_hover_image),QtGui.QPixmap(data.back_button_pressed_image),self)
        self.back_button.setFixedSize(72,72)
        #--------------------------------------------------------------------------------------


        #--------------------------------------------------------------------------------------
        self.photo_label = QtGui.QLabel(self)
        self.photo_label.setFont(newFont)
        self.photo_label.setFixedSize(650,600)
        self.photo_label.setAlignment(QtCore.Qt.AlignCenter)
        self.photo_label.setSizePolicy( QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored )
        self.photo_label.setStyleSheet('color: white')
        #--------------------------------------------------------------------------------------

        
        #--------------------------------------------------------------------------------------
        self.open_file = PicButton(QtGui.QPixmap(data.insert_photo_unpressed_image),QtGui.QPixmap(data.insert_photo_hover_image),QtGui.QPixmap(data.insert_photo_pressed_image),self)
        self.open_file.setFixedSize(72,72)
        #--------------------------------------------------------------------------------------

        
        #--------------------------------------------------------------------------------------
        self.open_label = QtGui.QLabel(self)
        self.open_label.setFont(newFont)
        self.open_label.setText('Open Image')
        self.open_label.setStyleSheet('color: white')        
        #--------------------------------------------------------------------------------------

        
        #--------------------------------------------------------------------------------------
        self.enhance_image = QtGui.QLabel(self)
        self.enhance_image.setFont(newFont)
        self.enhance_image.setText('Enhance Image')
        self.enhance_image.setStyleSheet('color: white')        
        #--------------------------------------------------------------------------------------

        #--------------------------------------------------------------------------------------
        self.reset_button = PicButton(QtGui.QPixmap(data.reset_unpressed),QtGui.QPixmap(data.reset_unpressed),QtGui.QPixmap(data.reset_pressed),self)
        self.reset_button.setFixedSize(72,72)
        self.reset_button.setEnabled(False)
        #--------------------------------------------------------------------------------------

        
        #--------------------------------------------------------------------------------------
        self.coarse_slider_label = QtGui.QLabel(self)
        self.coarse_slider_label.setFont(QtGui.QFont(data.font_helvetica,12,QtGui.QFont.Normal))
        self.coarse_slider_label.setText('Coarse')
        self.coarse_slider_label.setStyleSheet('color: white')        
        #--------------------------------------------------------------------------------------

        
        #--------------------------------------------------------------------------------------
        self.coarse_slider = QtGui.QSlider(Qt.Horizontal)
        self.coarse_slider.setMinimum(0)
        self.coarse_slider.setMaximum(50)
        self.coarse_slider.setValue(25)
        self.coarse_slider.setTickPosition(QSlider.TicksBelow)
        self.coarse_slider.setTickInterval(10)
        self.coarse_slider.valueChanged.connect(lambda: self.valuechanged(1))
        #--------------------------------------------------------------------------------------

        
        #--------------------------------------------------------------------------------------
        self.fine_slider_label = QtGui.QLabel(self)
        self.fine_slider_label.setFont(QtGui.QFont(data.font_helvetica,12,QtGui.QFont.Normal))
        self.fine_slider_label.setText('Fine')
        self.fine_slider_label.setStyleSheet('color: white')        
        #--------------------------------------------------------------------------------------


        #--------------------------------------------------------------------------------------
        self.fine_slider = QtGui.QSlider(Qt.Horizontal)
        self.fine_slider.setMinimum(0)
        self.fine_slider.setMaximum(255)
        self.fine_slider.setValue(70)
        self.fine_slider.setTickPosition(QSlider.TicksBelow)
        self.fine_slider.setTickInterval(20)
        self.fine_slider.valueChanged.connect(lambda: self.valuechanged(2))
        #--------------------------------------------------------------------------------------



        
        #--------------------------------------------------------------------------------------
        self.rotate_slider_label = QtGui.QLabel(self)
        self.rotate_slider_label.setFont(QtGui.QFont(data.font_helvetica,12,QtGui.QFont.Normal))
        self.rotate_slider_label.setText('Rotate')
        self.rotate_slider_label.setStyleSheet('color: white')
        #--------------------------------------------------------------------------------------


        #--------------------------------------------------------------------------------------
        self.left_rotate = PicButton(QtGui.QPixmap(data.left_unpressed),QtGui.QPixmap(data.left_unpressed),QtGui.QPixmap(data.left_pressed),self)
        self.left_rotate.setFixedSize(72,72)
        self.left_rotate.setEnabled(False)
        #--------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------
        self.right_rotate = PicButton(QtGui.QPixmap(data.right_unpressed),QtGui.QPixmap(data.right_unpressed),QtGui.QPixmap(data.right_pressed),self)
        self.right_rotate.setFixedSize(72,72)
        self.right_rotate.setEnabled(False)
        #--------------------------------------------------------------------------------------

        
        #--------------------------------------------------------------------------------------
        self.ocr_button = PicButton(QtGui.QPixmap(data.ocr_unpressed),QtGui.QPixmap(data.ocr_hover),QtGui.QPixmap(data.ocr_pressed),self)
        self.ocr_button.setFixedSize(72,72)
        self.ocr_button.setEnabled(False)
        #--------------------------------------------------------------------------------------

        #--------------------------------------------------------------------------------------
        self.ocr_label = QtGui.QLabel(self)
        self.ocr_label.setFont(newFont)
        self.ocr_label.setText('OCR it')
        self.ocr_label.setStyleSheet('color: white')
        #--------------------------------------------------------------------------------------


        
        
        #--------------------------------------------------------------------------------------
        self.save_button = PicButton(QtGui.QPixmap(data.save_button_unpressed_image),QtGui.QPixmap(data.save_button_hover_image),QtGui.QPixmap(data.save_button_pressed_image),self)
        self.save_button.setFixedSize(72,72)
        self.save_button.setEnabled(False)
        #--------------------------------------------------------------------------------------

        #--------------------------------------------------------------------------------------
        self.save_label = QtGui.QLabel(self)
        self.save_label.setFont(newFont)
        self.save_label.setText('Save File')
        self.save_label.setStyleSheet('color: white')
        #--------------------------------------------------------------------------------------

        

        #--------------------------------------------------------------------------------------
        self.progress_bar = QtGui.QProgressBar(self)
        self.progress_bar.hide()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)
        self.progress_bar.setFixedSize(1100,30)
        self.progress_bar.setVisible(False)
        #--------------------------------------------------------------------------------------


        #--------------------------------------------------------------------------------------
        horz_top_Layout = QtGui.QHBoxLayout()
        horz_top_Layout.addWidget(self.back_button)
        horz_top_Layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))  #To add spacing in the layout
        #--------------------------------------------------------------------------------------

        #--------------------------------------------------------------------------------------
        vert_left_Layout = QtGui.QVBoxLayout()
        vert_left_Layout.addWidget(self.photo_label)

        #--------------------------------------------------------------------------------------

        
        #--------------------------------------------------------------------------------------
        horz1_Layout = QtGui.QHBoxLayout()
        horz1_Layout.addWidget(self.open_file)
        horz1_Layout.addWidget(self.open_label)
        #--------------------------------------------------------------------------------------

        #--------------------------------------------------------------------------------------
        horz2_Layout = QtGui.QHBoxLayout()
        horz2_Layout.addWidget(self.enhance_image)
        horz2_Layout.addWidget(self.reset_button)
        horz2_Layout.setContentsMargins(0,50,0,0)
        #--------------------------------------------------------------------------------------

        #--------------------------------------------------------------------------------------
        horz3_Layout = QtGui.QHBoxLayout()
        horz3_Layout.addWidget(self.coarse_slider_label)
        #--------------------------------------------------------------------------------------

        #--------------------------------------------------------------------------------------
        horz4_Layout = QtGui.QHBoxLayout()
        horz4_Layout.addWidget(self.coarse_slider)
        #--------------------------------------------------------------------------------------

        #--------------------------------------------------------------------------------------
        horz5_Layout = QtGui.QHBoxLayout()
        horz5_Layout.addWidget(self.fine_slider_label)
        #--------------------------------------------------------------------------------------

        #--------------------------------------------------------------------------------------
        horz6_Layout = QtGui.QHBoxLayout()
        horz6_Layout.addWidget(self.fine_slider)
        #--------------------------------------------------------------------------------------

        #--------------------------------------------------------------------------------------
        horz7_Layout = QtGui.QHBoxLayout()
        horz7_Layout.addWidget(self.rotate_slider_label)
        #--------------------------------------------------------------------------------------
        
        #--------------------------------------------------------------------------------------
        horz8_Layout = QtGui.QHBoxLayout()
        horz8_Layout.addWidget(self.left_rotate)
        horz8_Layout.addWidget(self.right_rotate)
        horz8_Layout.setContentsMargins(0,0,0,40)
        #--------------------------------------------------------------------------------------

        #--------------------------------------------------------------------------------------
        horz9_Layout = QtGui.QHBoxLayout()
        horz9_Layout.addWidget(self.ocr_button)
        horz9_Layout.addWidget(self.ocr_label)
        #--------------------------------------------------------------------------------------

        #--------------------------------------------------------------------------------------
        horz10_Layout = QtGui.QHBoxLayout()
        horz10_Layout.addWidget(self.save_button)
        horz10_Layout.addWidget(self.save_label)
        #--------------------------------------------------------------------------------------

        #--------------------------------------------------------------------------------------
        vert_right_Layout = QtGui.QVBoxLayout()
        vert_right_Layout.addLayout(horz1_Layout)
        vert_right_Layout.addLayout(horz2_Layout)
        vert_right_Layout.addLayout(horz3_Layout)
        vert_right_Layout.addLayout(horz4_Layout)
        vert_right_Layout.addLayout(horz5_Layout)
        vert_right_Layout.addLayout(horz6_Layout)
        vert_right_Layout.addLayout(horz7_Layout)
        vert_right_Layout.addLayout(horz8_Layout)
        vert_right_Layout.addLayout(horz9_Layout)
        vert_right_Layout.addLayout(horz10_Layout)
        vert_right_Layout.setContentsMargins(50,0,0,0)
        #--------------------------------------------------------------------------------------


        #--------------------------------------------------------------------------------------
        horz_medium_Layout = QtGui.QHBoxLayout()
        horz_medium_Layout.addLayout(vert_left_Layout)
        horz_medium_Layout.addLayout(vert_right_Layout)
        horz_medium_Layout.setContentsMargins(50,0,100,50)
        #--------------------------------------------------------------------------------------
        
        #--------------------------------------------------------------------------------------
        horz_bottom_Layout = QtGui.QHBoxLayout()
        horz_bottom_Layout.addWidget(self.progress_bar)
        horz_bottom_Layout.setContentsMargins(0,0,0,20)
        #--------------------------------------------------------------------------------------


        #--------------------------------------------------------------------------------------
        vert_main_Layout = QtGui.QVBoxLayout()
        vert_main_Layout.addLayout(horz_top_Layout)
        vert_main_Layout.addLayout(horz_medium_Layout)
        vert_main_Layout.addLayout(horz_bottom_Layout)
        self.setLayout(vert_main_Layout)
        #--------------------------------------------------------------------------------------

        self.reset_button.clicked.connect(self.reset_method)
        self.left_rotate.clicked.connect(lambda: self.rotate_method(1))
        self.right_rotate.clicked.connect(lambda: self.rotate_method(2))
        self.open_file.clicked.connect(self.open_image_method)
        self.ocr_button.clicked.connect(self.do_operation)
        self.save_button.clicked.connect(self.save_method)
        
        self.back_button.clicked.connect(self.parent().back_button)
    

    #----------------------------------------------------------------------------------
    def save_method(self):
        name = QtGui.QFileDialog.getSaveFileName(self,'Save File')
        file = open(name,'w')
        temp_file = open("C:\\Github\\Multitasker\\Data\\result.txt",'r')
        text = temp_file.read()
        file.write(text)
        temp_file.close()
        os.remove("C:\\Github\\Multitasker\\Data\\result.txt")
        file.close()

    #----------------------------------------------------------------------------------  
        
        
    #----------------------------------------------------------------------------------
    def reset_method(self):
        self.coarse_slider.setValue(25)
        self.fine_slider.setValue(70)
        self.open_cv_image = self.original_image
        self.show_pixmap(self.original_image)
    
    #----------------------------------------------------------------------------------
    def rotate_method(self,direction):
        
        if direction==1 :       #left rotate
            angle = -90
            self.left_rot +=1
            if self.right_rot >0:
                self.right_rot -=1
        else:
            angle = 90         #Right rotate
            self.right_rot +=1
            if self.left_rot >0:
                self.left_rot -=1
            
        (h, w) = self.open_cv_image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # grab the rotation matrix (applying the negative of the angle to rotate clockwise), then grab the sine and cosine (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX , cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        self.open_cv_image = cv2.warpAffine(self.open_cv_image, M, (nW, nH),cv2.INTER_LANCZOS4)
        
        self.flag = 1
        
        if self.left_rot == 4 or self.right_rot == 4:
            self.open_cv_image = self.original_image
            self.left_rot = self.right_rot = 0
            self.open_cv_image = self.original_image
            self.open_cv_image = cv2.adaptiveThreshold(self.open_cv_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,self.adaptive_thresh_value)
            ret,self.open_cv_image  = cv2.threshold(self.open_cv_image,self.normal_thresh_value,255,cv2.THRESH_BINARY)


        #show the opencv image in the Qlabel
        self.show_pixmap(self.open_cv_image)
        
            
     
    #----------------------------------------------------------------------------------
    def open_image_method(self):
        self.image_name = QtGui.QFileDialog.getOpenFileName(self,'Select Image','','Images(*.png *.jpg *.jpeg)')
        
        if len(self.image_name) >0:         #Check whether user has choose an image or not
            try:
                self.ocr_button.setEnabled(True)
                self.coarse_slider.setValue(25)
                self.fine_slider.setValue(70)
                self.reset_button.setEnabled(True)
                self.flag = 0
                self.inner_flag = 0
                
                basewidth  = 650    #Desired width of the image
                baseheight = 600    #Desired height of the image
                image = Image.open(self.image_name)
                width,height = image.size
                wpercent = (basewidth/float(image.size[0]))
                hpercent = (baseheight/float(image.size[1]))

                hsize = int((float(image.size[1])*float(wpercent)))
                wsize = int((float(image.size[0])*float(hpercent)))
                if width > height:
                    size1 = (basewidth,hsize)
                else :
                    size1 = (wsize,baseheight)
                image.thumbnail(size1, Image.ANTIALIAS)

                
                #Convert from PIL to opencv image
                temp_image = image.convert('RGB')
                self.open_cv_image = np.array(temp_image)
                self.open_cv_image = self.open_cv_image[:, :, ::-1].copy()
                
                
                self.open_cv_image = cv2.cvtColor(self.open_cv_image,cv2.COLOR_BGR2GRAY)
                self.original_image = self.open_cv_image
                self.temp_image     = self.open_cv_image

                
                #Enable left and right rotate buttons
                self.left_rotate.setEnabled(True)
                self.right_rotate.setEnabled(True)
                #show the image on the  scree
                self.show_pixmap(self.temp_image)    
            except: pass
    #----------------------------------------------------------------------------------

    
    #----------------------------------------------------------------------------------
    def valuechanged(self,option):
        try:
            if option == 1 and len(self.image_name):
                self.adaptive_thresh_value  =  self.coarse_slider.value()
            
            elif option == 2 and len(self.image_name):
                self.normal_thresh_value  = self.fine_slider.value()

            temp_image = self.open_cv_image
            if len(self.image_name)>0:
                if option ==1:
                    temp_image = cv2.adaptiveThreshold(temp_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,self.adaptive_thresh_value)
                    ret,temp_image  = cv2.threshold(temp_image,self.normal_thresh_value,255,cv2.THRESH_BINARY)            
                else:
                    ret,temp_image  = cv2.threshold(temp_image,self.normal_thresh_value,255,cv2.THRESH_BINARY)
                    temp_image = cv2.adaptiveThreshold(temp_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,self.adaptive_thresh_value)
            
            #show the opencv image in the Qlabel
            self.show_pixmap(temp_image)
            

        except: pass        
    #----------------------------------------------------------------------------------


    #----------------------------------------------------------------------------------
    def show_pixmap(self,image):
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        
        #Convert a opencv image to QImage i.e of pixmap format
        height, width,channel = image.shape
        bytesPerLine = 3 * width
        qImg = QtGui.QImage(image, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        self.photo_label.setPixmap(QtGui.QPixmap(qImg))

    #----------------------------------------------------------------------------------
    def words(self,text):
        return re.findall(r'\w+', text.lower())


    #----------------------------------------------------------------------------------
    def show_progress_bar(self):
        self.progress_bar.show()


    #----------------------------------------------------------------------------------
    def do_operation(self):
        #-------------------------------------------------------------
        #Show the progress bar until the execution is not completed
        thread_progress_bar = Thread(target = self.show_progress_bar)
        thread_progress_bar.start()
        #-------------------------------------------------------------


        #-------------------------------------------------------------
        thread_ocr  = Thread(target =self.ocr_it_method)
        thread_ocr.start()
        #-------------------------------------------------------------

        
        
    #----------------------------------------------------------------------------------
    def ocr_it_method(self):
        
        self.save_button.setEnabled(True)


        im  = cv2.imread(self.image_name,0)
        clf = joblib.load(data.pickel_file)
        f   = open(data.text_file,'w')
        
        im_th = cv2.adaptiveThreshold(im,  255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,self.adaptive_thresh_value)
        ret,im_th = cv2.threshold(im_th,self.normal_thresh_value,255,cv2.THRESH_BINARY)

        height,width=im.shape

        row=np.zeros(height,dtype=np.int16) # no. of row = height of page

        #-------------------------------------------------------------------------------
        #count black pixel in reach row
        for i in range(height):
            for j in range(width):
                if(im_th[i][j]==0):
                    row[i]+=1;

        row_true=np.zeros(height,dtype=np.int16) # actual rows
        num_row = 0

        
        #check if height =0 is first row
        if(row[0]>0):
            row_true[0]=0
            num_row+=1
        mean_dist_row=0


        #-------------------------------------------------------------------------------
        # find actual rows
        # row_true -> array having height of actual rows
        # col_true -> array having width of actual columns
        for j in range(1,height,1):
            if((row[j]>0 and row[j-1]==0) or ( row[j]==0 and row[j-1]>0)):
                row_true[num_row]=j
                num_row+=1
                if(num_row%2==0):
                    mean_dist_row+=row_true[num_row-1]-row_true[num_row-2]

        mean_dist_row/=num_row
        mean_dist_row*=2  # as actual rows= num_row/2

        j=1

        #-------------------------------------------------------------------------------
        self.im_copy2 = im_th.copy()
        first_word=1 # help to check if character is first character of word
             # for detection between I and l(small l)

        first_sentence=1 # help to capitalise frist letter in sentence


        #-------------------------------------------------------------------------------
        # dfs function 
        def func(c,r):
            if(c<c1 or c>c2 or r<r1 or r>r2 or self.im_copy2[r][c]!=0):
                return 
            self.im_copy2[r][c]=255
            self.cc2=max(self.cc2,c)
            self.rr1=min(self.rr1,r)
            self.rr2=max(self.rr2,r)
            roi[r-r1][c-c1]=255  
            func(c-1,r-1)
            func(c,r-1)
            func(c+1,r-1)
            func(c-1,r)
            func(c+1,r)
            func(c-1,r+1)
            func(c,r+1)
            func(c+1,r+1)


        #--------------------------------------------------------------------------------
        def words(text):
            return re.findall(r'\w+', text.lower())

        WORDS = Counter(words(open(data.dictionary).read()))


        #--------------------------------------------------------------------------------
        # spell check functions
        def P(word, N=sum(WORDS.values())):
            return WORDS[word] / N
        ##    "Most probable spelling correction for word."


        #--------------------------------------------------------------------------------
        def correction(word): 
            return max(candidates(word), key=P)
        ##    "Generate possible spelling corrections for word."


        #--------------------------------------------------------------------------------
        def candidates(word): 
            return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])
        ##    "The subset of `words` that appear in the dictionary of WORDS."


        #--------------------------------------------------------------------------------
        def known(words): 
            return set(w for w in words if w in WORDS)

        #--------------------------------------------------------------------------------
        ##    "All edits that are one edit away from `word`."
        def edits1(word):
            letters    = "abcdefghijklmnopqrstuvwxyz0123456789!’#$%&‘()*+,-./:;<=>?@\^_`{|}~÷"
            splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        ##    deletes    = [L + R[1:]               for L, R in splits if R]
            transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
            replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        ##    inserts    = [L + c + R               for L, R in splits for c in letters]
        ##    return set(deletes + transposes + replaces + inserts)
            return set(transposes+replaces)


        #--------------------------------------------------------------------------------
        ##    "All edits that are two edits away from `word`."
        def edits2(word): 
            return (e2 for e1 in edits1(word) for e2 in edits1(e1))


        #--------------------------------------------------------------------------------
        word=''
        word1=''
    
        while j<num_row: # for each row

            #--------------------------------------------------------------------------------
            # illegal row removed
            if((2*(row_true[j]-row_true[j-1]))<mean_dist_row): 
                num_row-=1
                for k in range(j,num_row,1):
                    row_true[k]=row_true[k+1]
            #--------------------------------------------------------------------------------
            # else proceed with row
            else:
                if(j>1):
                    # write word and write new line, j>1 so that condition not checked
                    # for first row

                    #--------------------------------------------------------------------------------
                    # up - is first letter of word capital
                    if(len(word)>0):
                        q=len(word)
                        if(word[0].isupper()):
                            up=1
                        else:
                            up=0
                        # correct from first char till either ( end or first special char)    
                        for p in range(len(word)-1,-1,-1):
                           if((word[p]>='a' and word[p]<='z' )or(word[p]>='A' and word[p]<='Z') ):
                               break
                           else:
                               q-=1

                        #--------------------------------------------------------------------------------
                        if(q==0 or word=='I'): # word contain only special character
                            f.write(word)
                        #--------------------------------------------------------------------------------
                        else:
                            for p in range(q):
                                if(word[p]=='1'):
                                    word1+='l'
                                elif(word[p]=='0'):
                                    if(p==0):
                                        word1+='O'
                                    else:
                                        word1+='o'
                                elif(word[p]=='5'):
                                    if(p==0):
                                        word1+='S'
                                    else:
                                        word1+='s'
                                elif(word[p]=='8'):
                                    word1+='a'
                                else:
                                    word1+=word[p]
                            for p in range(q,len(word)-1,1):
                                word1+=word[p]
                            p=len(word)-2
                            if(q<(len(word))):
                                if(word[p+1]=='*'):
                                    word1+='.'
                                else:
                                    word1+=word[p+1]
                            word2 = spell(word1[0:q])
                            word3= correction(word2)
                            
                            if(up==1):
                                f.write(word3[0].capitalize())
                            else:
                                f.write(word3[0])
                            
                            f.write(word3[1:len(word3)])
                            f.write(word1[q:len(word)])
                        #--------------------------------------------------------------------------------
                    word=''
                    word1=''
                    f.write('\n')
                #--------------------------------------------------------------------------------
                col=np.zeros(width+1,dtype=np.int16)

                col_true=np.zeros(width,dtype=np.int16)
                mean_col=0
                num_col=0

                for i in range(width):
                    for k in range(row_true[j-1],row_true[j],1):
                        if(self.im_copy2[k,i]==0):
                            col[i]+=1
                
                for k in range(width):
                    if(col[k]>0):
                        mean_col+=col[k]
                        num_col+=1

                mean_col/=num_col
                num_col=0
                
                if(col[0]>0):
                    col_true[0]=0
                    num_col=1
                flag=0

                #--------------------------------------------------------------------------------
                # count actual columns in row between lines row_true[j] and row_true[j-1]
                for k in range(1,width,1):
                    
                    if(flag==0 and col[k]>0 and col[k-1]==0):
                        col_true[num_col]=k
                        num_col+=1
                        flag=1

                    elif(flag==1 and col[k]==0 and col[k-1]>0):
                        col_true[num_col]=k

                        # remove noise
                        if((col_true[num_col]-col_true[num_col-1])<=(mean_dist_row/20)):
                            num_col-=1
                            flag=0
                            continue
                        
                        num_col+=1
                        flag=0
                k=1
                r1=row_true[j-1]
                r2=row_true[j]-1

                #--------------------------------------------------------------------------------
                while(k<num_col): # for each contour
                    # space and write word
                    if(k>2 and ((col_true[k-1]-col_true[k-2])>((r2-r1+1)/5.25))):
                        if(len(word)>0):
                            q=len(word)
                            if(word[0].isupper()):
                                up=1
                            else:
                                up=0
                            # correct from first char till either ( end or first special char)    
                            for p in range(len(word)-1,-1,-1):
                               if((word[p]>='a' and word[p]<='z' )or(word[p]>='A' and word[p]<='Z') ):
                                   break
                               else:
                                   q-=1

                            if(q==0 or word=='I'): # word contain only special character
                                f.write(word)
                            else:
                                for p in range(q):
                                    if(word[p]=='1'):
                                        word1+='l'
                                    elif(word[p]=='0'):
                                        if(p==0):
                                            word1+='O'
                                        else:
                                            word1+='o'
                                    elif(word[p]=='5'):
                                        if(p==0):
                                            word1+='S'
                                        else:
                                            word1+='s'
                                    elif(word[p]=='8'):
                                        word1+='a'
                                    else:
                                        word1+=word[p]
                                for p in range(q,len(word)-1,1):
                                    word1+=word[p]
                                p=len(word)-2
                                if(q<(len(word))):
                                    if(word[p+1]=='*'):
                                        word1+='.'
                                    else:
                                        word1+=word[p+1]
                                word2 = spell(word1[0:q])
                                word3= correction(word2)
                                
                                if(up==1):
                                    f.write(word3[0].capitalize())
                                else:
                                    f.write(word3[0])
                                f.write(word3[1:len(word3)])
                                f.write(word1[q:len(word)])
                        word=''
                        word1=''
                        first_word=1
                        f.write(' ') # space
                    proi=np.ones((0,0))
                    # pr1,pr2,pc1,pc2 -> proi coordinates 
                    pr1=10000
                    pr2=-1
                    pc1=10000
                    pc2=-1
                    
                    merge=0


                    #--------------------------------------------------------------------------------
                    while(k<num_col): # for multiple character in contour

                        c1=col_true[k]
                        c2=col_true[k]-1
                        flag_c1=1
                        
                        for p in range(col_true[k-1],col_true[k],1):
                            for q in range(r1,r2+1,1):
                                if(self.im_copy2[q][p]==0):
                                    c1=p
                                    flag_c1=0
                                    break
                            if(flag_c1==0):
                                break
                        if(c1>c2):
                            k+=2    #no character now,print last character
                            if(proi.size==0):
                                continue;

                            #proi=cv2.dilate(proi,kernel,iterations = 1)

                            proi = cv2.resize(proi,(28,28),interpolation=cv2.INTER_AREA)
                            proi=proi.ravel()

                            for vi in range(784):
                                if proi[vi]>0:
                                    proi[vi]=1
                                else:
                                    proi[vi]=0
                            proi=proi.reshape(1,-1)
                            nbr = clf.predict(proi)
                            
                            # pr2-pr1=height, pc2-pc1 =width
                            if(nbr==48 or nbr==79 or nbr==111):# 0 o O
                                
                                if((pr2-pr1+1)/(pc2-pc1+1)<0.8 or (pr2-pr1+1)/(pc2-pc1+1)>1.25):
                                    nbr=48
                                elif((pr2-pr1+1)>(0.6*(r2-r1+1))):
                                     nbr=79
                                else:
                                     nbr=111

                            if(nbr==124 or nbr==73 or nbr==108 or nbr==45 or nbr==95 or nbr==46):

                                if((pr2-pr1+1)/(pc2-pc1+1)>0.6 and (pr2-pr1+1)/(pc2-pc1+1)<1.67):
                                    nbr=46 # .
                                elif((pr2-pr1+1)<(pc2-pc1+1)): 
                                    if(pr1>((r2-r1+1)*0.75+r1)): # _
                                       nbr=95
                                    else: # -
                                       nbr=45
                                elif(first_word==1):
                                    nbr=73
                                else:
                                    nbr=108

                            if(nbr==58 or nbr==61):
                                if((pr2-pr1+1)/(pc2-pc1+1)>2):
                                    nbr=58 # :
                                else:
                                    nbr=61 #=
                            # i is merged of 2 segment,l is single segment
                            if(nbr==91 or nbr==93): # ] [ -> i (merge=1) ,l (merge=0)
                                if(merge==1):
                                    nbr=105
                                else:
                                    nbr=108

                            if(nbr==127): 
                                nbr=120     #  × -> x
                                
                            if(nbr==34 or nbr==44): # , "
                                if(pr1<(r1+(r2-r1+1)*0.4)):
                                    word=word+'’'
                                else:
                                    word=word+','
                                    
                            elif nbr==39:
                                    word=word+'‘'
                                    
                            elif nbr<=126:
                                if(first_sentence==1):
                                    if (nbr>96 and nbr<123):
                                        nbr-=32 # capital first letter of sentence
                                elif(nbr>64 and nbr<91):
                                    nbr+=32
                                if((nbr>64 and nbr<91) or( nbr>96 and nbr<123)):
                                    first_sentence=0            
                                word=word+chr(nbr)
                            else:
                                word=word+'÷'
                                
                            if(nbr==33 or nbr==46 or nbr==63): #sentence ends with . ? !
                                first_sentence=1
                                
                            first_word=0
                            break
                        
                        for p in range(r1,r2+1,1):
                            if(self.im_copy2[p][c1]==0):
                                break
                            
                        roi = np.zeros((r2+1-r1,c2+1-c1))

                        for q in range(r2+1-r1):
                            for r in range(c2+1-c1):
                                roi[q][r]=0
                        # rr1,rr2,c1,cc2 ->roi coordinates
                        # r1,r2,c1,c2 ->row coordinates
                        
                        self.rr1=p
                        self.cc2=c1
                        self.rr2=self.rr1
                        
                        # dfs function
                        func(c1,p)
                        
                        roi=roi[self.rr1-r1:self.rr2+1-r1,0:self.cc2+1-c1]

                        # merge roi and proi , proi= previous roi
                        
                        if(pc2>0 and ( pc2>self.cc2 or ((self.cc2-c1+1)*0.25+c1)<pc2 or ((pc2-pc1+1)*0.75+pc1)>c1)):
                            p2r1=min(pr1,self.rr1)
                            p2c1=min(pc1,c1)
                            p2r2=max(pr2,self.rr2)
                            p2c2=max(pc2,self.cc2)
                            proi2=np.zeros((p2r2-p2r1+1,p2c2-p2c1+1))

                            for p in range(self.rr1,self.rr2+1,1):
                                for q in range(c1,self.cc2+1,1):
                                    p2=p-p2r1
                                    q2=q-p2c1
                                    if(roi[p-self.rr1][q-c1]==255):
                                        proi2[p2][q2]=255

                            for p in range(pr1,pr2+1,1):
                                for q in range(pc1,pc2+1,1):
                                    p2=p-p2r1
                                    q2=q-p2c1
                                    if(proi2[p2][q2]==255):
                                        continue
                                    if(proi[p-pr1][q-pc1]==255):
                                        proi2[p2][q2]=255
                            proi=proi2
                            pr1=p2r1
                            pc1=p2c1
                            pr2=p2r2
                            pc2=p2c2
                            # merge help with conversion ] [ -> i or l
                            merge=1
                            continue    #overlapping character like % i
                        else:
                            if(proi.size>0):

                                #proi=cv2.dilate(proi,kernel,iterations = 1)
                                # write proi 
                                proi = cv2.resize(proi,(28,28),interpolation=cv2.INTER_AREA)
                                proi=proi.ravel()
                                
                                for vi in range(784):
                                    if proi[vi]>0:
                                        proi[vi]=1
                                    else:
                                        proi[vi]=0
                                proi=proi.reshape(1,-1)
                                nbr = clf.predict(proi)
                                
                                if(nbr==48 or nbr==79 or nbr==111):# 0 o O
                                    if((pr2-pr1+1)/(pc2-pc1+1)<0.8 or (pr2-pr1+1)/(pc2-pc1+1)>1.25):
                                        nbr=48
                                    elif((pr2-pr1+1)>(0.6*(r2-r1+1))):
                                         nbr=79
                                    else:
                                         nbr=111
                                         
                                if(nbr==124 or nbr==73 or nbr==108 or nbr==45 or nbr==95 or nbr==46):
                                    if((pr2-pr1+1)/(pc2-pc1+1)>0.6 and (pr2-pr1+1)/(pc2-pc1+1)<1.67):
                                        nbr=46 # .
                                    elif((pr2-pr1)<(pc2-pc1+1)): 
                                        if(pr1>((r2-r1+1)*0.75+r1)): # _
                                           nbr=95
                                        else: # -
                                           nbr=45
                                    elif(first_word==1):
                                        nbr=73 # I
                                    else:
                                        nbr=108 # l(small l)
                                         
                                if(nbr==58 or nbr==61):
                                    if((pr2-pr1+1)/(pc2-pc1+1)>2):
                                        nbr=58 # :
                                    else:
                                        nbr=61 #=
                                        
                                if(nbr==91 or nbr==93): # ] [ -> i (merge=1) ,l (merge=0)
                                    if(merge==1):
                                        nbr=105
                                    else:
                                        nbr=108
                                        
                                if(nbr==127): 
                                    nbr=120     #  × -> x
                                    
                                if(nbr==34 or nbr==44): # , "
                                    if(pr1<(r1+(r2-r1+1)*0.4)):
                                        word=word+'’'
                                    else:
                                        word=word+','
                                        
                                elif nbr==39:
                                    word=word+'‘'
                                    
                                elif nbr<=126:
                                    if(first_sentence==1):
                                        if (nbr>96 and nbr<123):
                                            nbr-=32 # capital 1 letter of sentence
                                    elif(nbr>64 and nbr<91):
                                        nbr+=32
                                    if((nbr>64 and nbr<91) or( nbr>96 and nbr<123)):
                                        first_sentence=0            
                                    word=word+chr(nbr)
                                else:
                                    word=word+'÷'
                                    
                                if(nbr==33 or nbr==46 or nbr==63):
                                    first_sentence=1
                                    
                                first_word=0
            
                            proi=roi
                            pr1=self.rr1
                            pc1=c1
                            pr2=self.rr2
                            pc2=self.cc2
                j+=2

        #--------------------------------------------------------------------------------
        # write last word
        if(len(word)>0):
            q=len(word)
            if(word[0].isupper()):
                up=1
            else:
                up=0
            # correct from first char till either ( end or first special char)    
            for p in range(len(word)-1,-1,-1):
               if((word[p]>='a' and word[p]<='z' )or(word[p]>='A' and word[p]<='Z') ):
                   break
               else:
                   
                   q-=1

            if(q==0 or word=='I'): # word contain only special character
                f.write(word)
            else:
                for p in range(q):
                    if(word[p]=='1'):
                        word1+='l'
                    elif(word[p]=='0'):
                        if(p==0):
                            word1+='O'
                        else:
                            word1+='o'
                    elif(word[p]=='5'):
                        if(p==0):
                            word1+='S'
                        else:
                            word1+='s'
                    elif(word[p]=='8'):
                        word1+='a'
                    else:
                        word1+=word[p]
                for p in range(q,len(word)-1,1):
                    word1+=word[p]
                p=len(word)-2
                if(q<(len(word))):
                    if(word[p+1]=='*'):
                        word1+='.'
                    else:
                        word1+=word[p+1]
                    
                word2 = spell(word1[0:q])
                word3= correction(word2)
                
                if(up==1):
                    f.write(word3[0].capitalize())
                else:
                    f.write(word3[0])
                
                f.write(word3[1:len(word3)])
                f.write(word1[q:len(word)])
        f.close()
        print('1')
        self.progress_bar.hide()
        #-----------------------------------------------------------------------------        
           
        
####################################################################################################        
#########                    Class Photo_ocr_class ends here                          ##############        
#################################################################################################### 






####################################################################################################        
#####         Class containing  the functionality of Pdf_scanner  option              ##############        
####################################################################################################
class Pdf_scanner_class(QtGui.QWidget):

    list_l =[]  #List for storing the images loaded using the insert_image button
    merger = PdfFileMerger()

    #---------------------------------------------------------------------------------------
    def __init__(self, parent=None):
        super(Pdf_scanner_class, self).__init__(parent)
        mainWindow = QtGui.QWidget()
        self.initUI_Pdf_scanner_class()
        
    #----------------------------------------------------------------------------------------
    def initUI_Pdf_scanner_class(self):

        #Font for the text in the Pdf_scanner window
        newFont = QtGui.QFont(data.font_helvetica,data.font_size,QtGui.QFont.Normal)

        
        self.back_button = PicButton(QtGui.QPixmap(data.back_button_unpressed_image),QtGui.QPixmap(data.back_button_hover_image),QtGui.QPixmap(data.back_button_pressed_image),self)
        self.back_button.setFixedSize(72,72)
        self.back_button.setShortcut('Esc')
        

        self.insert_image_button = PicButton(QtGui.QPixmap(data.insert_button_unpressed_image),QtGui.QPixmap(data.insert_button_hover_image),QtGui.QPixmap(data.insert_button_pressed_image),self)
        self.insert_image_button.setFixedSize(72,72)
        self.insert_image_button.setShortcut('Ctrl+o')

        self.insert_image_label = QtGui.QLabel(self)
        self.insert_image_label.setFont(newFont)
        self.insert_image_label.setText('Insert Images')
        self.insert_image_label.setStyleSheet('color: white')
        self.insert_image_label.move(1000,215)

        
        
        self.remove_selected_button = PicButton(QtGui.QPixmap(data.remove_button_unpressed_image),QtGui.QPixmap(data.remove_button_hover_image),QtGui.QPixmap(data.remove_button_pressed_image),self)
        self.remove_selected_button.setFixedSize(72,72)
        self.remove_selected_button.setShortcut('Ctrl+Delete')
        self.remove_selected_button.setEnabled(False)

        self.remove_selected_label = QtGui.QLabel(self)
        self.remove_selected_label.setFont(newFont)
        self.remove_selected_label.setText('Remove')
        self.remove_selected_label.setStyleSheet('color: white')
        

        
        self.convert_button = PicButton(QtGui.QPixmap(data.convert_button_unpressed_image),QtGui.QPixmap(data.convert_button_hover_image),QtGui.QPixmap(data.convert_button_pressed_image),self)
        self.convert_button.setFixedSize(72,72)
        self.convert_button.setShortcut('Ctrl+Space')
        self.convert_button.setEnabled(False)

        self.convert_label = QtGui.QLabel(self)
        self.convert_label.setFont(newFont)
        self.convert_label.setText('Convert')
        self.convert_label.setStyleSheet('color: white')

        
        self.save_button = PicButton(QtGui.QPixmap(data.save_button_unpressed_image),QtGui.QPixmap(data.save_button_hover_image),QtGui.QPixmap(data.save_button_pressed_image),self)
        self.save_button.setFixedSize(72,72)
        self.save_button.setShortcut('Ctrl+s')
        self.save_button.setEnabled(False)

        self.save_label = QtGui.QLabel(self)
        self.save_label.setFont(newFont)
        self.save_label.setText('Save Pdf')
        self.save_label.setStyleSheet('color: white')

        
        self.progress_bar = QtGui.QProgressBar(self)
        self.progress_bar.hide()
        self.progress_bar.resize(1100,105)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet('color:white')
        self.progress_bar.setFont(QtGui.QFont(data.font_helvetica,13,QtGui.QFont.Normal))
        
        self.listWidget = QtGui.QListWidget()
        self.listWidget.setViewMode(QtGui.QListWidget.IconMode)
        self.listWidget.hide()
        

        ## Top horizontal layout
        horz_top_Layout = QtGui.QHBoxLayout()
        horz_top_Layout.addWidget(self.back_button)
        horz_top_Layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))  #To add spacing in the layout
        horz_top_Layout.setContentsMargins(0, 0, 0, 40)

        vert_medium_1_Layout = QtGui.QVBoxLayout()
        vert_medium_1_Layout.addWidget(self.listWidget)
        vert_medium_1_Layout.setContentsMargins(40, 0, 20, 0)
        vert_medium_1_Layout.setGeometry(QtCore.QRect(200,200,800,500))

        
        horz_medium_1_1_Layout = QtGui.QVBoxLayout()
        horz_medium_1_1_Layout.addWidget(self.insert_image_button,1)
        horz_medium_1_2_Layout = QtGui.QVBoxLayout()
        horz_medium_1_2_Layout.addWidget(self.insert_image_label,2)

        horz_medium_2_1_Layout = QtGui.QVBoxLayout()
        horz_medium_2_1_Layout.addWidget(self.remove_selected_button,1)
        horz_medium_2_2_Layout = QtGui.QVBoxLayout()
        horz_medium_2_2_Layout.addWidget(self.remove_selected_label,2)
        
        horz_medium_3_1_Layout = QtGui.QVBoxLayout()
        horz_medium_3_1_Layout.addWidget(self.convert_button,1)
        horz_medium_3_2_Layout = QtGui.QVBoxLayout()
        horz_medium_3_2_Layout.addWidget(self.convert_label,2)
    
        horz_medium_4_1_Layout = QtGui.QVBoxLayout()
        horz_medium_4_1_Layout.addWidget(self.save_button,1)
        horz_medium_4_2_Layout = QtGui.QVBoxLayout()
        horz_medium_4_2_Layout.addWidget(self.save_label,2)


        horz_medium_1_Layout = QtGui.QHBoxLayout()
        horz_medium_1_Layout.addLayout(horz_medium_1_1_Layout,1)
        horz_medium_1_Layout.addLayout(horz_medium_1_2_Layout,2)

        
        horz_medium_2_Layout = QtGui.QHBoxLayout()
        horz_medium_2_Layout.addLayout(horz_medium_2_1_Layout,1)
        horz_medium_2_Layout.addLayout(horz_medium_2_2_Layout,2)


        horz_medium_3_Layout = QtGui.QHBoxLayout()
        horz_medium_3_Layout.addLayout(horz_medium_3_1_Layout,1)
        horz_medium_3_Layout.addLayout(horz_medium_3_2_Layout,2)


        horz_medium_4_Layout = QtGui.QHBoxLayout()
        horz_medium_4_Layout.addLayout(horz_medium_4_1_Layout,1)
        horz_medium_4_Layout.addLayout(horz_medium_4_2_Layout,2)

        vert_medium_2_Layout = QtGui.QVBoxLayout()
        vert_medium_2_Layout.addLayout(horz_medium_1_Layout)
        vert_medium_2_Layout.addLayout(horz_medium_2_Layout)
        vert_medium_2_Layout.addLayout(horz_medium_3_Layout)
        vert_medium_2_Layout.addLayout(horz_medium_4_Layout)
        vert_medium_2_Layout.setContentsMargins(40, 10, 20, 40)

        #Middle horizontal layout
        horz_medium_Layout   = QtGui.QHBoxLayout()
        horz_medium_Layout.addLayout(vert_medium_1_Layout,4)
        horz_medium_Layout.addLayout(vert_medium_2_Layout,1)
        horz_medium_Layout.setContentsMargins(40, 10, 20, 40)

        ##Bottom horizontal layout
        horz_bottom_Layout   = QtGui.QHBoxLayout()
        horz_bottom_Layout.addWidget(self.progress_bar)
        horz_bottom_Layout.setContentsMargins(80, 20, 80, 40)
        
        ##Final Vertical layout
        vert_Layout = QtGui.QVBoxLayout()
        vert_Layout.addLayout(horz_top_Layout,1)
        vert_Layout.addLayout(horz_medium_Layout,4)
        vert_Layout.addLayout(horz_bottom_Layout,1)
        self.setLayout(vert_Layout)

            
        self.insert_image_button.clicked.connect(self.insert_image_method)
        self.convert_button.clicked.connect(self.convert_image_method)
        self.remove_selected_button.clicked.connect(self.remove_selectd_method)        
        self.save_button.clicked.connect(self.save_method)
        self.back_button.clicked.connect(self.parent().back_button)
    #------------------------------------------------------------------------------




    #------------------------------------------------------------------------------
    def insert_image_method(self):
        try:
            _fromUtf8 = QtCore.QString.fromUtf8
        except AttributeError:
            _fromUtf8 = lambda s: s

        temp_list = []     #For temporarily storing the valid images            
        #For showing a dialog box to insert images
        for path in QtGui.QFileDialog.getOpenFileNames(self, 'Select Images','','Images(*.png *.jpg *.jpeg)'):
            temp_list.append(path)
        
        
        if len(temp_list):
            counter = 0    # To check how many files are .png or .jpg or .jpeg  , if there is none then we will do nothing
            
            for one_file in temp_list:
                if one_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    counter += 1
                    item = QtGui.QListWidgetItem()
                    #-------------------------------------------------
                    #This ensures the correct orientation of image
                    i = Image.open(one_file)
                    for orientation in ExifTags.TAGS.keys():
                        if ExifTags.TAGS[orientation] =='Orientation':
                            break
                    try:
                        exif = dict(i._getexif().items())
                        if exif[orientation] == 3:
                            i = i.rotate(180,expand=True)
                        elif exif[orientation] == 6:
                            i = i.rotate(270,expand=True)
                        elif exif[orientation] == 8:
                            i = i.rotate(90,expand=True)
                        i.save(one_file)
                        i.close()
                    except: pass
                    #--------------------------------------------------
                    
                    image_name = os.path.basename(one_file) 
                    for i in range(len(image_name)-1,0,-1):  #For removing the .extension from image
                        if image_name[i] is '.':
                            break
                    image_name = image_name[0:i]             #Till this point

                    icon = QtGui.QIcon(one_file)
                    item.setIcon(icon)
                    item = QtGui.QListWidgetItem(image_name)
                    item.setIcon(icon);
                    self.listWidget.addItem(item)
                    self.listWidget.setSpacing(20)   #Spacing between different items in the Widget
                    self.listWidget.setIconSize(QtCore.QSize(320,200))  # to set uniform size to each item
                    self.listWidget.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
            if counter >0 :
                self.listWidget.show()
                self.convert_button.setEnabled(True)
                self.remove_selected_button.setEnabled(True)
                for i in range(len(temp_list)):
                    self.list_l.append(temp_list[i])
    #-------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------
    def remove_selectd_method(self):
        listItems=self.listWidget.selectedItems()
        temp_item = []   #It will contain the text of images to be remove from the list_l so as to keep consisteency with the QListWidget
        if not listItems: return
        for item in listItems:
            temp_item.append(str(item.text()))      
        for item in listItems:
            self.listWidget.takeItem(self.listWidget.row(item))
        temp_list = [] #for storing the absolute name of image with any path or .exteension 
        for i in range(len(self.list_l)):
            x = os.path.basename(self.list_l[i]) 
            for i in range(len(x)-1,0,-1):  #For removing the .extension from image
                        if x[i] is '.':
                            break
            temp_list.append(x[0:i])

        temp_num = []
        #delete the selected items from list also
        for i in range(len(temp_item)):
            for j in range(len(temp_list)):
                if(temp_item[i]==temp_list[j]):
                    temp_num.append(j)
        temp_num.sort(reverse=True)
        for i in temp_num:
            try:
                del self.list_l[i]
            except: pass
        if len(self.list_l) == 0:
            self.remove_selected_button.setEnabled(False)
            self.convert_button.setEnabled(False)
            self.save_button.setEnabled(False)
            self.listWidget.hide()
        else: self.convert_button.setEnabled(True)
    #---------------------------------------------------------------------------------------    
    
    #---------------------------------------------------------------------------------------
    def convert_image_method(self):

        self.progress_bar.show()        #To show the progress bar
        
        directory = 'C:\\qw1resw32cew321232' #A temporary directory
        temp_index = 0
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        intermediate_pdf = 'a_a_x_z_1_2_b_7_1'
        path             = directory + '\\'
        temp_file_name   = path + 'temporary_11.jpg'
        list_pdf_name    = []
        counter = 0  #For imediate pdf number
        
        self.progress_bar.setValue(1)
        i = 1
        for image_name in self.list_l:
            try:
                image             =  cv2.imread(image_name)
                cv2.imwrite(temp_file_name, image)
                image             = Image.open(temp_file_name)
                image_with_border = ImageOps.expand(image,border = 50,fill='white')                
                width , height    = image_with_border.size
                image_with_border.thumbnail((width, height), Image.ANTIALIAS)
                bet_pdf           = path + intermediate_pdf + str(counter)+'.pdf'
                list_pdf_name.append(bet_pdf)
                Image.Image.save(image_with_border, bet_pdf , "PDF" , resoultion=100)
                try:
                    os.remove(temp_file_name)
                except: pass
                counter +=1

                #----------------
                self.progress_bar.setValue(100/(len(self.list_l))*i)
                self.progress_bar.update()
                i += 1
                #----------------
            except : pass
            
    
        temp_merger = PdfFileMerger()
        if len(list_pdf_name) >0:
            ## Now we merge the intermediate files into one pdf whose name is given in name_of_pdf
            for filename in list_pdf_name:
                temp_merger.append(PdfFileReader(filename))

            self.merger = temp_merger
            ## Now delete the intermediate file names
            for filename in list_pdf_name:
                try:
                    os.remove(filename)
                except: pass
        
        self.save_button.setEnabled(True)
        self.convert_button.setEnabled(False)
        shutil.rmtree(directory)
        self.progress_bar.hide()
    #----------------------------------------------------------------------------------------------------

        
        
    #----------------------------------------------------------------------------------------------------
    def save_method(self):

        name = QtGui.QFileDialog.getSaveFileName(self,'Save PDF','','*.pdf')
        try:
            if name == '' or (not name.endswith('.pdf')):
                pass
            else:
                self.merger.write(name)
        except : pass
    #----------------------------------------------------------------------------------------------------

 
####################################################################################################        
#########                Class Pdf_scanner_class ends here                            ##############        
#################################################################################################### 







####################################################################################################        
#####         Class containing  the functionality of Speech_to_text option            ##############        
####################################################################################################
class Speech_to_text_class(QtGui.QWidget):

    text = ''   #For storing the text temporarily
    #-----------------------------------------------------------------------------------
    def __init__(self, parent=None):
        super(Speech_to_text_class, self).__init__(parent)
        mainWindow = QtGui.QWidget()
        self.initUI_Speech_to_text_class()
    #-----------------------------------------------------------------------------------


    
    
    #-----------------------------------------------------------------------------------
    def initUI_Speech_to_text_class(self):
        
        #Font for the text in the Pdf_scanner window
        newFont = QtGui.QFont(data.font_helvetica,data.font_size,QtGui.QFont.Normal)

        
        self.back_button = PicButton(QtGui.QPixmap(data.back_button_unpressed_image),QtGui.QPixmap(data.back_button_hover_image),QtGui.QPixmap(data.back_button_pressed_image),self)
        self.back_button.setFixedSize(72,72)
        self.back_button.setShortcut('Esc')
        self.back_button.move(10,10)



        self.listening_button = PicButton(QtGui.QPixmap(data.listening_unpressed_image),QtGui.QPixmap(data.listening_hover_image),QtGui.QPixmap(data.listening_pressed_image),self)
        self.listening_button.setFixedSize(200,200)
        self.listening_button.move(200,150)

        self.listening_label = QtGui.QLabel(self)
        self.listening_label.setFont(newFont)
        self.listening_label.setText('Tap To Start')
        self.listening_label.setStyleSheet('color: white')
        self.listening_label.move(220,380)
        self.listening_label.resize(250,40) 
        

        
        self.open_button = PicButton(QtGui.QPixmap(data.open_unpressed_image),QtGui.QPixmap(data.open_hover_image),QtGui.QPixmap(data.open_pressed_image),self)
        self.open_button.setFixedSize(72,72)
        self.open_button.setShortcut('Ctrl+o')
        self.open_button.move(200,515)

        self.open_label = QtGui.QLabel(self)
        self.open_label.setFont(newFont)
        self.open_label.setText('Edit File')
        self.open_label.setStyleSheet('color: white')
        self.open_label.move(300,530)

        self.save_button = PicButton(QtGui.QPixmap(data.save_button_unpressed_image),QtGui.QPixmap(data.save_button_hover_image),QtGui.QPixmap(data.save_button_pressed_image),self)
        self.save_button.setFixedSize(72,72)
        self.save_button.setShortcut('Ctrl+s')
        self.save_button.move(200,615)
        self.save_button.setEnabled(False)

        self.save_label = QtGui.QLabel(self)
        self.save_label.setFont(newFont)
        self.save_label.setText('Save File')
        self.save_label.setStyleSheet('color: white')
        self.save_label.move(300,630)


        self.font_editor = QtGui.QFont()
        self.font_editor.setFamily("Courier")
        self.font_editor.setFixedPitch(True)
        self.font_editor.setPointSize(12)

        
        self.text_editor = QtGui.QTextEdit(self)
        self.text_editor.setVisible(False)
        self.text_editor.move(600,100)
        self.text_editor.resize(600,600)
        self.text_editor.setFont(self.font_editor)
        
        self.listening_button.clicked.connect(self.listening_button_method)
        self.open_button.clicked.connect(self.open_file_method)
        self.save_button.clicked.connect(self.save_file_method)
        self.back_button.clicked.connect(self.parent().back_button)
    #---------------------------------------------------------------------------   




    #---------------------------------------------------------------------------
    def listening_button_method(self):
        thread1 = Thread(target = self.enable_the_things)
        thread1.start()
        thread2 = Thread(target=self.execute_listening)
        thread2.start()
        self.listening_label.setText('Listening . . .')        
    #---------------------------------------------------------------------------




    #---------------------------------------------------------------------------
    def enable_the_things(self):
        self.save_button.setEnabled(True)
        self.text_editor.setVisible(True)
    #---------------------------------------------------------------------------




    #---------------------------------------------------------------------------    
    def tap_to_start(self):
        self.listening_label.setText('Tap To Start')
    #---------------------------------------------------------------------------





    #---------------------------------------------------------------------------
    def please_repeat(self):
        self.listening_label.setText('Please Repeat')
    #---------------------------------------------------------------------------




    #---------------------------------------------------------------------------    
    def execute_listening(self):
        r = sr.Recognizer()
##        m = sr.Microphone()
##        with m as source: r.adjust_for_ambient_noise(source)
        with sr.Microphone() as source:
            audio = r.listen(source)
        try:
            self.text_editor.textCursor().insertText(' ' + r.recognize_google(audio))
            self.tap_to_start()
        except sr.UnknownValueError:
            self.please_repeat()
        except: pass
    #----------------------------------------------------------------------------------



    
    
    #----------------------------------------------------------------------------------
    def open_file_method(self):
        self.save_button.setEnabled(True)
        file_name = QtGui.QFileDialog.getOpenFileName(self,'Open File to Edit')
        try:
            file_data = open(file_name,'r')
            self.text_editor.setVisible(True)
            with file_data:
                self.text = file_data.read()
                self.text_editor.setText(self.text)
        except: pass
    #------------------------------------------------------------------------------



    
    #------------------------------------------------------------------------------
    def save_file_method(self):
        try:
            name = QtGui.QFileDialog.getSaveFileName(self,'Save File')
            file = open(name,'w')
            self.text = self.text_editor.toPlainText()
            file.write(self.text)
            file.close()
            self.text = ''
        except: pass
    #------------------------------------------------------------------------------- 
  



####################################################################################################        
#########           Class Speech_to_text_class ends here                              ##############        
#################################################################################################### 






####################################################################################################        
#####         Class containing  the functionality of text_to_Speech option            ##############        
####################################################################################################
class Text_to_speech_class(QtGui.QWidget):

    mixer.init()
    text = ''  #Contains the text which will be shared by different methods i.e the text in the editor
    directory = 'D:\\qw1r_esw32c_e_w321232' #A temporary directory
    counter = 0
    tts = ''   #For storing  the audio file data
    def __init__(self, parent=None):
        super(Text_to_speech_class, self).__init__(parent)
        mainWindow = QtGui.QWidget()
        self.initUI_Text_to_speech_class()


    def initUI_Text_to_speech_class(self):
        
        #Font for the text in the Pdf_scanner window
        newFont = QtGui.QFont(data.font_helvetica,data.font_size,QtGui.QFont.Normal)

        #--------------------------------------------------------------------------------------
        self.back_button = PicButton(QtGui.QPixmap(data.back_button_unpressed_image),QtGui.QPixmap(data.back_button_hover_image),QtGui.QPixmap(data.back_button_pressed_image),self)
        self.back_button.setFixedSize(72,72)
        self.back_button.setShortcut('Esc')
        #--------------------------------------------------------------------------------------




        #--------------------------------------------------------------------------------------
        self.open_button = PicButton(QtGui.QPixmap(data.open_unpressed_image),QtGui.QPixmap(data.open_hover_image),QtGui.QPixmap(data.open_pressed_image),self)
        self.open_button.setFixedSize(72,72)
        self.open_button.setShortcut('Ctrl+o')
        #--------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------
        self.open_label = QtGui.QLabel(self)
        self.open_label.setFont(newFont)
        self.open_label.setText('Open File')
        self.open_label.setStyleSheet('color: white')
        #--------------------------------------------------------------------------------------




        #--------------------------------------------------------------------------------------
        self.convert_speech_button = PicButton(QtGui.QPixmap(data.speak_unpressed_image),QtGui.QPixmap(data.speak_hover_image),QtGui.QPixmap(data.speak_pressed_image),self)
        self.convert_speech_button.setFixedSize(72,72)
        self.convert_speech_button.setShortcut('Ctrl+t')
        self.convert_speech_button.setEnabled(True)
        #--------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------
        self.convert_speech_label = QtGui.QLabel(self)
        self.convert_speech_label.setFont(newFont)
        self.convert_speech_label.setText('Audify It')
        self.convert_speech_label.setStyleSheet('color: white')
        #--------------------------------------------------------------------------------------




        #-------------------------------------------------------------------------------------- 
        self.radio_group_box =QtGui.QGroupBox('Choose Language Accent')
        self.radio_group_box.setStyleSheet('color: white')
        self.radio_group_box.setFont(QtGui.QFont(data.font_helvetica,12,QtGui.QFont.Normal))
        self.radio_button_group = QtGui.QButtonGroup()
        self.radio_button_list = []
        radio_button_name_list = ["English-US","English-UK","English-Indian"]
        for each in radio_button_name_list:
            self.radio_button_list.append(QRadioButton(each))
        for i in range(3):
            self.radio_button_list[i].setStyleSheet('color: white')
            self.radio_button_list[i].setFont(QtGui.QFont(data.font_helvetica,10,QtGui.QFont.Normal))
        self.radio_button_list[0].setChecked(True)
        #--------------------------------------------------------------------------------------


        
        #--------------------------------------------------------------------------------------
        self.save_button = PicButton(QtGui.QPixmap(data.save_button_unpressed_image),QtGui.QPixmap(data.save_button_hover_image),QtGui.QPixmap(data.save_button_pressed_image),self)
        self.save_button.setFixedSize(72,72)
        self.save_button.setShortcut('Ctrl+s')
        self.save_button.setEnabled(False)
        #--------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------
        self.save_label = QtGui.QLabel(self)
        self.save_label.setFont(newFont)
        self.save_label.setText('Save File')
        self.save_label.setStyleSheet('color: white')
        #--------------------------------------------------------------------------------------




        #--------------------------------------------------------------------------------------
        self.font_set = QtGui.QFont()
        self.font_set.setFamily("Courier")
        self.font_set.setFixedPitch(True)
        self.font_set.setPointSize(12)        
        self.text_show = QtGui.QTextEdit(self)
        self.text_show.setFont(self.font_set)
        #--------------------------------------------------------------------------------------




        #--------------------------------------------------------------------------------------    
        self.play_button = PicButton(QtGui.QPixmap(data.play_unpressed_image),QtGui.QPixmap(data.play_hover_image),QtGui.QPixmap(data.play_pressed_image),self)
        self.play_button.setFixedSize(72,72)
        self.play_button.setShortcut(' ')
        self.play_button.setEnabled(False)
        #--------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------
        self.play_label = QtGui.QLabel(self)
        self.play_label.setFont(newFont)
        self.play_label.setText('Play')
        self.play_label.setStyleSheet('color: white')
        #--------------------------------------------------------------------------------------




        #--------------------------------------------------------------------------------------
        self.pause_button = PicButton(QtGui.QPixmap(data.pause_unpressed_image),QtGui.QPixmap(data.pause_hover_image),QtGui.QPixmap(data.pause_pressed_image),self)
        self.pause_button.setFixedSize(72,72)
        self.pause_button.setShortcut(' ')
        self.pause_button.setEnabled(False)
        #--------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------
        self.pause_label = QtGui.QLabel(self)
        self.pause_label.setFont(newFont)
        self.pause_label.setText('Stop')
        self.pause_label.setStyleSheet('color: white')
        #--------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------
        self.progress_bar = QtGui.QProgressBar(self)
        self.progress_bar.hide()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)
        #--------------------------------------------------------------------------------------
        


        #--------------------------------------------------------------------------------------
        horz_top_Layout = QtGui.QHBoxLayout()
        horz_top_Layout.addWidget(self.back_button)
        horz_top_Layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))  #To add spacing in the layout
        horz_top_Layout.setContentsMargins(0, 0, 0, 0)
        #--------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------
        ver_11_Layout = QtGui.QVBoxLayout()
        ver_11_Layout.addWidget(self.open_button)
        ver_12_Layout = QtGui.QVBoxLayout()
        ver_12_Layout.addWidget(self.open_label)
        hor_1_Layout = QtGui.QHBoxLayout()
        hor_1_Layout.addLayout(ver_11_Layout,1)
        hor_1_Layout.addWidget(QtGui.QLabel('     '))
        hor_1_Layout.addLayout(ver_12_Layout,1)
        #--------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------
        ver_21_Layout = QtGui.QVBoxLayout()
        ver_21_Layout.addWidget(self.convert_speech_button)
        ver_22_Layout = QtGui.QVBoxLayout()
        ver_22_Layout.addWidget(self.convert_speech_label)
        hor_2_Layout = QtGui.QHBoxLayout()
        hor_2_Layout.addLayout(ver_21_Layout,1)
        hor_2_Layout.addWidget(QtGui.QLabel('      '))
        hor_2_Layout.addLayout(ver_22_Layout,1)
        #--------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------
        self.radio_button_ver_Layout = QtGui.QVBoxLayout()
        counter = 1
        for each in self.radio_button_list:
            self.radio_button_ver_Layout.addWidget(each)
            self.radio_button_group.addButton(each)
            self.radio_button_group.setId(each,counter)
            counter +=1
        self.radio_group_box.setLayout(self.radio_button_ver_Layout)
        #--------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------
        ver_3_Layout = QtGui.QVBoxLayout()
        ver_3_Layout.addWidget(self.radio_group_box)
        ver_3_Layout.setContentsMargins(60, 10, 0, 50)
        #--------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------
        ver_41_Layout = QtGui.QVBoxLayout()
        ver_41_Layout.addWidget(self.save_button)
        ver_42_Layout = QtGui.QVBoxLayout()
        ver_42_Layout.addWidget(self.save_label)
        hor_4_Layout = QtGui.QHBoxLayout()
        hor_4_Layout.addLayout(ver_41_Layout,1)
        hor_4_Layout.addWidget(QtGui.QLabel('      '))
        hor_4_Layout.addLayout(ver_42_Layout,1)
        #--------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------
        ver_text_show_Layout  = QtGui.QVBoxLayout()
        ver_text_show_Layout.addWidget(self.text_show)
        ver_text_show_Layout.setContentsMargins(50,0,50,50)
        #--------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------
        ver_61_Layout = QtGui.QVBoxLayout()
        ver_61_Layout.addWidget(self.play_button)
        ver_62_Layout = QtGui.QVBoxLayout()
        ver_62_Layout.addWidget(self.play_label)
        hor_6_Layout = QtGui.QHBoxLayout()
        hor_6_Layout.addLayout(ver_61_Layout)
        hor_6_Layout.addLayout(ver_62_Layout)
        #--------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------
        ver_71_Layout = QtGui.QVBoxLayout()
        ver_71_Layout.addWidget(self.pause_button)
        ver_72_Layout = QtGui.QVBoxLayout()
        ver_72_Layout.addWidget(self.pause_label)
        hor_7_Layout = QtGui.QHBoxLayout()
        hor_7_Layout.addLayout(ver_71_Layout)
        hor_7_Layout.addLayout(ver_72_Layout)
        #--------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------
        horz_bar_Layout = QtGui.QHBoxLayout()
        horz_bar_Layout.addWidget(self.progress_bar)
        horz_bar_Layout.setContentsMargins(100,0,50,20)
        #--------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------
        hor_play_pause_Layout = QtGui.QHBoxLayout()
        hor_play_pause_Layout.addLayout(hor_6_Layout)
        hor_play_pause_Layout.addLayout(hor_7_Layout)
        hor_play_pause_Layout.setContentsMargins(200,0,100,0)
        #--------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------
        vert_right_Layout = QtGui.QVBoxLayout()
        vert_right_Layout.addLayout(ver_text_show_Layout,4)
        vert_right_Layout.addLayout(hor_play_pause_Layout)
        vert_right_Layout.setContentsMargins(0,50,0,50)
        #--------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------
        vert_left_Layout = QtGui.QVBoxLayout()
        vert_left_Layout.addLayout(hor_1_Layout)
        vert_left_Layout.addLayout(hor_2_Layout)
        vert_left_Layout.addLayout(ver_3_Layout)
        vert_left_Layout.addLayout(hor_4_Layout)
        vert_left_Layout.setContentsMargins(100,50,100,150)
        #--------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------
        horz_bottom_Layout = QtGui.QHBoxLayout()
        horz_bottom_Layout.addLayout(vert_left_Layout,1.5)
        horz_bottom_Layout.addLayout(vert_right_Layout,2)
        #--------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------
        vert_main_Layout  = QtGui.QVBoxLayout()
        vert_main_Layout.addLayout(horz_top_Layout)
        vert_main_Layout.addLayout(horz_bottom_Layout)
        vert_main_Layout.addLayout(horz_bar_Layout)
        self.setLayout(vert_main_Layout)
        #--------------------------------------------------------------------------------------

        self.convert_speech_button.clicked.connect(self.main_function_convert)
        self.open_button.clicked.connect(self.open_file_method)
        self.save_button.clicked.connect(self.save_file_method)
        
        self.back_button.clicked.connect(self.remove_directory)
        self.back_button.clicked.connect(self.parent().back_button)
        
        self.play_button.clicked.connect(lambda: self.play_pause_audio(1))
        self.pause_button.clicked.connect(lambda: self.play_pause_audio(2))    

    #--------------------------------------------------------------------------------------
    def remove_directory(self):
        try:
            shutil.rmtree(self.directory)
        except: pass
    #--------------------------------------------------------------------------------------
    



    #--------------------------------------------------------------------------------------
    def selected_button(self):
        return self.radio_button_group.checkedId()
    #--------------------------------------------------------------------------------------



    
    #--------------------------------------------------------------------------------------
    def open_file_method(self):
        self.convert_speech_button.setEnabled(True)
        file_name = QtGui.QFileDialog.getOpenFileName(self,'Text file to audify','','*.txt')
        try:
            file_data = open(file_name,'r')
            with file_data:
                text = file_data.read()
                self.text_show.setText(text)
        except: pass
    #--------------------------------------------------------------------------------------    




    #--------------------------------------------------------------------------------------
    def save_file_method(self):
        try:
            name = QtGui.QFileDialog.getSaveFileName(self,'Save Audio File','','*.mp3')
            self.tts.save(name)
            self.text = ''
        except: pass
    #--------------------------------------------------------------------------------------



    #--------------------------------------------------------------------------------------
    def show_progress_bar(self):
        self.progress_bar.show()
    #--------------------------------------------------------------------------------------


        
    #--------------------------------------------------------------------------------------
    def main_function_convert(self):
        thread1 = Thread(target = self.show_progress_bar)
        thread1.start()
        thread2 = Thread(target = self.convert_speech_method)
        thread2.start()
    #--------------------------------------------------------------------------------------



        
    #--------------------------------------------------------------------------------------    
    def convert_speech_method(self):
       
        self.text = self.text_show.toPlainText()
        if len(self.text) >0:
            lang = ''       #For storing the output language
            self.progress_bar.show()
            selected_button_id = self.selected_button()
            if selected_button_id == 1:
                lang = 'en-us'
            elif selected_button_id ==2:
                lang = 'en-uk'
            else:
                lang = 'hi'
                
            mixer.music.load('C:\\Github\\Multitasker\\Data\\hello.mp3')
            self.tts = gTTS(self.text,lang)
            try:
                shutil.rmtree(self.directory)
            except: pass
            
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
            
            self.tts.save(self.directory+'\\'+'vbe179jakvhbra.mp3')
            try:
                self.progress_bar.hide()
            except: pass
            self.save_button.setEnabled(True)
            self.play_button.setEnabled(True)
            self.pause_button.setEnabled(True)

    #--------------------------------------------------------------------------------------




    #--------------------------------------------------------------------------------------
    def play_pause_audio(self,flag):
       temp_f = 0
       if flag == 1:
           try:
               mixer.music.load(self.directory+'\\'+'vbe179jakvhbra.mp3')
               mixer.music.play()
           except:pass
       else:
           try:
               mixer.music.load(self.directory+'\\'+'vbe179jakvhbra.mp3')
               mixer.music.pause()
               temp_f = 1
           except: pass
           self.pause_button.setEnabled(False)
    #-------------------------------------------------------------------------------------
    
####################################################################################################        
#########                      Class Text_to_speech_class ends here                   ##############        
#################################################################################################### 






####################################################################################################        
#####               Class containing  the functionality of about us option            ##############        
####################################################################################################
class About_us_class(QtGui.QWidget):

    flag = 0
    def __init__(self, parent=None):
        super(About_us_class, self).__init__(parent)
        mainWindow = QtGui.QWidget()
        self.initUI_About_us_class()


    def initUI_About_us_class(self):

        #Font for the text in the Pdf_scanner window
        newFont = QtGui.QFont(data.font_helvetica,data.font_size,QtGui.QFont.Normal)

        #--------------------------------------------------------------------------------------
        self.back_button = PicButton(QtGui.QPixmap(data.back_button_unpressed_image),QtGui.QPixmap(data.back_button_hover_image),QtGui.QPixmap(data.back_button_pressed_image),self)
        self.back_button.setFixedSize(72,72)
        self.back_button.setShortcut('Esc')
        #--------------------------------------------------------------------------------------


        #--------------------------------------------------------------------------------------
        self.instance = vlc.Instance()
        self.mediaplayer = self.instance.media_player_new()
        self.widget = QtGui.QWidget(self)
        self.videoframe = QtGui.QFrame()
        self.videoframe.setFixedSize(600,400)
        #--------------------------------------------------------------------------------------

        
        #--------------------------------------------------------------------------------------    
        self.play_button = PicButton(QtGui.QPixmap(data.play_unpressed_image),QtGui.QPixmap(data.play_hover_image),QtGui.QPixmap(data.play_pressed_image),self)
        self.play_button.setFixedSize(72,72)
        #--------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------
        self.play_label = QtGui.QLabel(self)
        self.play_label.setFont(newFont)
        self.play_label.setText('Play')
        self.play_label.setStyleSheet('color: white')
        #--------------------------------------------------------------------------------------




        #--------------------------------------------------------------------------------------
        self.pause_button = PicButton(QtGui.QPixmap(data.pause_unpressed_image),QtGui.QPixmap(data.pause_hover_image),QtGui.QPixmap(data.pause_pressed_image),self)
        self.pause_button.setFixedSize(72,72)
        #--------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------
        self.pause_label = QtGui.QLabel(self)
        self.pause_label.setFont(newFont)
        self.pause_label.setText('Stop')
        self.pause_label.setStyleSheet('color: white')
        #--------------------------------------------------------------------------------------



        #--------------------------------------------------------------------------------------
        self.email_label = QtGui.QLabel(self)
        self.email_label.setFont(newFont)
        self.email_label.setText('Email us')
        self.email_label.setStyleSheet('color: white')
        #--------------------------------------------------------------------------------------

        #--------------------------------------------------------------------------------------
        self.email = QtGui.QLabel(self)
        self.email.setFont(QFont("Courier",12))
        self.email.setText('Email :')
        self.email.setStyleSheet('color: white')
        #--------------------------------------------------------------------------------------
    
        #--------------------------------------------------------------------------------------
        self.password = QtGui.QLabel(self)
        self.password.setFont(QFont("Courier",12))
        self.password.setText('Password :')
        self.password.setStyleSheet('color: white')
        #--------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------
        self.message = QtGui.QLabel(self)
        self.message.setFont(QFont("Courier",12))
        self.message.setText('Message :')
        self.message.setStyleSheet('color: white')
        #--------------------------------------------------------------------------------------


        #--------------------------------------------------------------------------------------
        self.email_edit = QtGui.QLineEdit()
        self.email_edit.setFont(QFont("Courier",14))
        self.email_edit.setFixedSize(500,30)
        #--------------------------------------------------------------------------------------


        #--------------------------------------------------------------------------------------
        self.password_edit = QtGui.QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.password_edit.setFont(QFont("Courier",14))
        self.password_edit.setFixedSize(500,30)
        #--------------------------------------------------------------------------------------



        #--------------------------------------------------------------------------------------    
        self.send = PicButton(QtGui.QPixmap(data.send_unpressed),QtGui.QPixmap(data.send_hover),QtGui.QPixmap(data.send_pressed),self)
        self.send.setFixedSize(72,72)
        send_vert = QtGui.QVBoxLayout()
        send_vert.addWidget(self.send)
        send_vert.setContentsMargins(440,0,0,0)
        #--------------------------------------------------------------------------------------
        


        #--------------------------------------------------------------------------------------
        self.message_edit = QtGui.QTextEdit()
        self.message_edit.setFont(QFont("Courier",14))
        self.message_edit.setFixedSize(500,300)
        #--------------------------------------------------------------------------------------

        #--------------------------------------------------------------------------------------
        form_layout = QtGui.QFormLayout()
        form_layout.addRow('',self.email)
        form_layout.addRow('',self.email_edit)
        form_layout.addRow('',self.password)
        form_layout.addRow('',self.password_edit)
        form_layout.addRow('',self.message)
        form_layout.addRow('',self.message_edit)
        #--------------------------------------------------------------------------------------


        #--------------------------------------------------------------------------------------
        self.error_message = QtGui.QLabel(self)
        self.error_message.setFont(QFont("Times",18))
        self.error_message.setText('Some Error Occured !!')
        self.error_message.setStyleSheet('color: red')
        self.error_message.move(170,700)
        self.error_message.hide()
        #--------------------------------------------------------------------------------------



        
        #--------------------------------------------------------------------------------------
        form_layout_vert = QtGui.QVBoxLayout()
        form_layout_vert.addWidget(self.email_label) 
        form_layout_vert.addLayout(form_layout)
        form_layout_vert.addLayout(send_vert)
        form_layout_vert.setContentsMargins(0,20,30,0)
        #--------------------------------------------------------------------------------------


        #--------------------------------------------------------------------------------------
        horizontal_1 = QtGui.QHBoxLayout()
        horizontal_1.addWidget(self.play_button)
        horizontal_1.addWidget(self.play_label)
        horizontal_1.setContentsMargins(20,0,50,0)
        #--------------------------------------------------------------------------------------


        #--------------------------------------------------------------------------------------
        horizontal_2 = QtGui.QHBoxLayout()
        horizontal_2.addWidget(self.pause_button)
        horizontal_2.addWidget(self.pause_label)
        horizontal_2.setContentsMargins(50,0,20,0)
        #--------------------------------------------------------------------------------------


        #-------------------------------------------------------------------------------------
        horizontal_3 = QtGui.QHBoxLayout()
        horizontal_3.addLayout(horizontal_1)
        horizontal_3.addLayout(horizontal_2)
        horizontal_3.setContentsMargins(100,0,100,0)
        #-------------------------------------------------------------------------------------


        
        #-------------------------------------------------------------------------------------
        self.hbackbutton = QtGui.QHBoxLayout()
        self.hbackbutton.addWidget(self.back_button)
        s = '     '  # For space
        s = s*28
        self.hbackbutton.addWidget(QtGui.QLabel(s))
        self.hbackbutton.addLayout(horizontal_3)
        self.hbackbutton.setContentsMargins(10,10,10,0)
        #-------------------------------------------------------------------------------------

        #-------------------------------------------------------------------------------------
        self.videoframebox = QtGui.QVBoxLayout()
        self.videoframebox.addWidget(self.videoframe)
        self.videoframebox.setContentsMargins(10,50,10,0)
        #-------------------------------------------------------------------------------------

        #-------------------------------------------------------------------------------------
        self.horz_mediumlayout = QtGui.QHBoxLayout()
        self.horz_mediumlayout.addLayout(form_layout_vert)
        self.horz_mediumlayout.addLayout(self.videoframebox)
        self.horz_mediumlayout.setContentsMargins(50,0,50,0)
        #-------------------------------------------------------------------------------------
        

        #-------------------------------------------------------------------------------------
        self.vboxlayout = QtGui.QVBoxLayout()
        self.vboxlayout.addLayout(self.hbackbutton)
        self.vboxlayout.addLayout(self.horz_mediumlayout)
        self.widget.setLayout(self.vboxlayout)
        #-------------------------------------------------------------------------------------

        #-------------------------------------------------------------------------------------
        self.send.clicked.connect(self.send_method)
        self.pause_button.clicked.connect(self.stop)
        self.play_button.clicked.connect(self.play_pause)
        self.back_button.clicked.connect(self.back_pressed)
        self.back_button.clicked.connect(self.parent().back_button)
        self.email_edit.textChanged.connect(self.doSomething)
        #-------------------------------------------------------------------------------------



    #-------------------------------------------------------------------------------------
    def doSomething(self):
        if self.flag == 0:
            self.error_message.hide()
            self.flag = 1
    #-------------------------------------------------------------------------------------

            
    #-------------------------------------------------------------------------------------
    def send_method(self):
        self.error_message.hide()
        sender_email = self.email_edit.text()
        sender_password = self.password_edit.text()
        sender_message = self.message_edit.toPlainText()
        reciever_email = 'kvikesh800@gmail.com'
        #Sending the email to the developer whose id is 'kvikesh800@gmail.com'
        try:
            server = smtplib.SMTP("smtp.gmail.com", 587, timeout=120)
            server.starttls()
            server.login(sender_email,sender_password)
            server.sendmail(sender_email,reciever_email,sender_message)
            server.quit()
        except:
            self.error_message.show()
            self.flag = 0
    #-------------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------------
    def back_pressed(self):
        self.error_message.hide()
        self.mediaplayer.stop()
    #-------------------------------------------------------------------------------------





    #-------------------------------------------------------------------------------------
    def play_pause(self):
        self.error_message.hide()
        if self.mediaplayer.is_playing() == False:
            if self.mediaplayer.play() == -1:
                self.media = self.instance.media_new(data.filename)
                self.mediaplayer.set_media(self.media)
                self.mediaplayer.set_hwnd(self.videoframe.winId())
            self.mediaplayer.play()
    #-------------------------------------------------------------------------------------




    #-------------------------------------------------------------------------------------
    def stop(self):
        self.error_message.hide()
        self.mediaplayer.stop()
    #-------------------------------------------------------------------------------------

####################################################################################################        
#########                 Class About_us_class ends here                              ##############        
#################################################################################################### 






####################################################################################################        
#####     Class containing  main window from where we switch to different options     ##############        
####################################################################################################
class Main_window(QtGui.QWidget):   # Main_window Inherit QtGui.QWidget

    #------------------------------------------------------------------
    #Constructor
    def __init__(self,parent=None):

        super(Main_window,self).__init__(parent)  #Super class constructor
        mainWindow = QtGui.QWidget()
        self.initUI_Main_window()
        

    #------------------------------------------------------------------
    def initUI_Main_window(self):       ##Main Window
        
        #Font for the text in the main window
        newFont = QtGui.QFont(data.font_helvetica,data.font_size,QtGui.QFont.Normal)



        
        self.button_photo_ocr = PicButton(QtGui.QPixmap(data.photo_ocr_unpressed_image),QtGui.QPixmap(data.photo_ocr_hover_image),QtGui.QPixmap(data.photo_ocr_pressed_image),self)
        self.button_photo_ocr.setFixedSize(200,200)
        self.button_photo_ocr.setShortcut('Ctrl+1')
        # Qlabel to show the text below  the button
        photo_ocr_qlabel = QtGui.QLabel(self)
        photo_ocr_qlabel.setFont(newFont)
        photo_ocr_qlabel.setText('                      Photo ocr')
        photo_ocr_qlabel.setStyleSheet('color: white')

        
        
        self.button_pdf_scanner = PicButton(QtGui.QPixmap(data.pdf_scanner_unpressed_image),QtGui.QPixmap(data.pdf_scanner_hover_image),QtGui.QPixmap(data.pdf_scanner_pressed_image),self)
        self.button_pdf_scanner.setFixedSize(200,200)
        self.button_pdf_scanner.setShortcut('Ctrl+2')
        # Qlabel to show the text below  the button
        pdf_scanner_qlabel = QtGui.QLabel(self)
        pdf_scanner_qlabel.setFont(newFont)
        pdf_scanner_qlabel.setStyleSheet('color: white')
        pdf_scanner_qlabel.setText('              Pdf Scanner')

        
        
        self.button_speech_to_text = PicButton(QtGui.QPixmap(data.speech_to_text_unpressed_image),QtGui.QPixmap(data.speech_to_text_hover_image),QtGui.QPixmap(data.speech_to_text_pressed_image),self)
        self.button_speech_to_text.setFixedSize(200,200)
        self.button_speech_to_text.setShortcut('Ctrl+3')
        # Qlabel to show the text below  the button
        speech_to_text_qlabel = QtGui.QLabel(self)
        speech_to_text_qlabel.setFont(newFont)
        speech_to_text_qlabel.setStyleSheet('color: white')
        speech_to_text_qlabel.setText('      Speech To Text')



        self.button_text_to_speech = PicButton(QtGui.QPixmap(data.text_to_speech_unpressed_image),QtGui.QPixmap(data.text_to_speech_hover_image),QtGui.QPixmap(data.text_to_speech_pressed_image),self)
        self.button_text_to_speech.setFixedSize(200,200)                   
        self.button_text_to_speech.setShortcut('Ctrl+4')
        # Qlabel to show the text below  the button
        text_to_speech_qlabel = QtGui.QLabel(self)
        text_to_speech_qlabel.setFont(newFont)
        text_to_speech_qlabel.setStyleSheet('color: white')
        text_to_speech_qlabel.setText('                 Text To Audio')



        self.button_about_us = PicButton(QtGui.QPixmap(data.about_us_unpressed_image),QtGui.QPixmap(data.about_us_hover_image),QtGui.QPixmap(data.about_us_pressed_image),self)
        self.button_about_us.setFixedSize(200,200)
        self.button_about_us.setShortcut('Ctrl+5')
        # Qlabel to show the text below  the button
        about_us_qlabel = QtGui.QLabel(self)
        about_us_qlabel.setFont(newFont)
        about_us_qlabel.setStyleSheet('color: white')
        about_us_qlabel.setText('   About Us')



        #Horizontal Layout for just space
        horz_Layout_0 = QtGui.QHBoxLayout()
        horz_Layout_0.addWidget(QtGui.QLabel(self))

        #Horizontal Layout for top 3 button
        horz_Layout_1 = QtGui.QHBoxLayout()
        horz_Layout_1.addWidget(self.button_photo_ocr)
        horz_Layout_1.addWidget(self.button_pdf_scanner)
        horz_Layout_1.addWidget(self.button_speech_to_text)

        #Horizontal Layout for top 3 text labels
        horz_Layout_2 = QtGui.QHBoxLayout()
        horz_Layout_2.addWidget(photo_ocr_qlabel)
        horz_Layout_2.addWidget(pdf_scanner_qlabel)
        horz_Layout_2.addWidget(speech_to_text_qlabel)
        horz_Layout_2.setContentsMargins(0,10,0,50)

        #Horizontal Layout for bottom 2 button
        horz_Layout_3 = QtGui.QHBoxLayout()
        horz_Layout_3.addWidget(QtGui.QLabel(self))    #Just to add space
        horz_Layout_3.addWidget(QtGui.QLabel(self))    #Just to add space
        horz_Layout_3.addWidget(self.button_text_to_speech)
        horz_Layout_3.addWidget(QtGui.QLabel(self))    #Just to add space
        horz_Layout_3.addWidget(self.button_about_us)
        horz_Layout_3.addWidget(QtGui.QLabel(self))    #Just to add space
        horz_Layout_3.addWidget(QtGui.QLabel(self))    #Just to add space

        #Horizontal Layout for bottom 2 labels
        horz_Layout_4 = QtGui.QHBoxLayout()
        horz_Layout_4.addWidget(text_to_speech_qlabel)
        horz_Layout_4.addWidget(about_us_qlabel)
        horz_Layout_4.setContentsMargins(200,10,0,50)

        
        #Final vertical layout that contains all the horizontal layouts
        ver_Layout_main = QtGui.QVBoxLayout()
        ver_Layout_main.addLayout(horz_Layout_0)
        ver_Layout_main.addLayout(horz_Layout_1)
        ver_Layout_main.addLayout(horz_Layout_2)
        ver_Layout_main.addLayout(horz_Layout_3)
        ver_Layout_main.addLayout(horz_Layout_4)
        ver_Layout_main.setContentsMargins(0, 0, 0, 0)
        self .setLayout(ver_Layout_main)


        #Calling the appropriate method to switch the layout
        self.button_photo_ocr.clicked.connect(self.parent().photo_ocr)
        self.button_pdf_scanner.clicked.connect(self.parent().pdf_scanner)
        self.button_speech_to_text.clicked.connect(self.parent().speech_to_text)
        self.button_text_to_speech.clicked.connect(self.parent().text_to_speech)
        self.button_about_us.clicked.connect(self.parent().about_us)
        

####################################################################################################        
#########                      Class Main_window ends here                            ##############        
####################################################################################################         




    

####################################################################################################        
#########       Class for creating a custom image button that has different states    ##############        
####################################################################################################
class PicButton(QtGui.QAbstractButton):
    def __init__(self, pixmap, pixmap_hover, pixmap_pressed, parent=None):
        super(PicButton, self).__init__(parent)
        self.pixmap = pixmap
        self.pixmap_hover = pixmap_hover
        self.pixmap_pressed = pixmap_pressed

        self.pressed.connect(self.update)
        self.released.connect(self.update)

    def paintEvent(self, event):
        pix = self.pixmap_hover if self.underMouse() else self.pixmap
        if self.isDown():
            pix = self.pixmap_pressed

        painter = QtGui.QPainter(self)
        painter.drawPixmap(event.rect(), pix)

    def enterEvent(self, event):
        self.update()

    def leaveEvent(self, event):
        self.update()

    def sizeHint(self):
        return QtCore.QSize(200, 200)
    
####################################################################################################        
#########                      Class PicButton ends here                              ##############        
####################################################################################################        







####################################################################################################        
#########                  Main entering point of the application                     ##############        
#################################################################################################### 
if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    Gui = App_Window()
    Gui.show()
    sys.exit(app.exec_())
