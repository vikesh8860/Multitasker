# Multitasker
This Repository contains a python based application that implements Photo Ocr, Photos to Pdf converter, Text to speech converter and Speech to text converter.

The Multitasker Project is a Machine Learning project made in python language and  UI made in PyQt framework. 
The parts of the app are:
- ### Photo Ocr
  The Photo Ocr is implemented in machine learning with the help of **SVM** library of skLearn and the **OpenCv** for image manipution. 
  The Photo Ocr application gives nearly 80% accuray on the digital printed cleaned text. The accuracy of prediction decrease as the   level of noise increases.

- ### Pdf Scanner
  The pdf scanner takes the images from the user in .png , .jpg , .jpeg, and .gif and then using the  Py2Pdf Python Library convert it to the pdf files. The features of this application are that you can add ,remove, change the order of images dynamically. After you convert it to pdf you can easily save in any directory.
 
- ### Speech To Text
  Sometimes there are situations when you want to write something rapidly but you have to write everthing by you hand so, what Speech to Text basically do is it directly transforms your spoken text in to an editable text where you can also edit the text in the editor.It uses the Python gtts module to convert the spoken words to editable text.
  
- ### Text To Speech
  This is an application by which you can cconvert any editable text document to an audio file and play instantly or save it for later use.Currently it supports three accents i.e English-US ,English-UK, English-Indian. It uses the python Speech recognition library to do the conversion.
  
The app is designed with Qt4 framework and is successfully tested on Windows 10 and Windows 8.1 


If you want to contact me, then feel free to ping me here : https://kvikesh800.wixsite.com/learner/contact

### If you want to contribute to the project then also you are welcomed

# ScreenShots

| ![home](https://user-images.githubusercontent.com/11665612/27263683-3a7c69de-548c-11e7-931a-d25fb68dec94.png) | ![photo_ocr](https://user-images.githubusercontent.com/11665612/27263688-5bf59432-548c-11e7-8a64-d26afa92eef0.png) |
|:---:|:---:|
| **Main Window** | **Photo Ocr Window** |

| ![pdf_scanner](https://user-images.githubusercontent.com/11665612/27263706-ab55c254-548c-11e7-9f0d-365165f65dfb.png) | ![speech_to_text](https://user-images.githubusercontent.com/11665612/27263721-af28dd08-548c-11e7-97c2-4e48e4d63bd5.png) | 
|:---:|:---:|
| **Pdf Scanner Window** | **Speech To Text Window** | 

| ![text_to_speech](https://user-images.githubusercontent.com/11665612/27263722-b6875f48-548c-11e7-9710-37cbd36213c2.png) | ![about_us](https://user-images.githubusercontent.com/11665612/27263723-ba2faa06-548c-11e7-921a-06b2a0af0185.png) | 
|:---:|:---:|
| **Text To Speech Window** | **About Us Window** | 
