
from django.shortcuts import render, redirect
from .models import Handler, Text
from .forms import HandlerForm
from django.conf import settings


# Objects detections
from objectdet.detector import Detector
import cv2
import numpy as np
import os
import glob
import pytesseract
import urllib
import time


def home(request):
    return render(request, "detect/home.html")


def upload(request):
    if request.method == 'POST':
        imageForm = HandlerForm(request.POST, request.FILES)
        if imageForm.is_valid():

            image = Handler(image=request.FILES['image'])
            image.save()

            return redirect('upload')
        else:
            return render(request, 'detect/home.html', {'error': 'Please upload image'})
    else:
        imageForm = HandlerForm()

    # Grab the latest image
    imageId = Handler.objects.order_by('-id')[:1]
    image = Handler.objects.get(id=imageId)
    return render(request, 'detect/home.html', {'image': image})


def detection(request):
    if request.method == 'POST':
        # Grab the image
        # find the id of the last image entry in database
        imageId = Handler.objects.order_by('-id')[:1]
        # Get the image object using the id
        imageObject = Handler.objects.get(id=imageId)
        # get the path of the image
        oImage = imageObject.image.path

        # Detectioin process starts here
        detectOT = Detector()
        detectOT.set_image(oImage)
        start = time.time()
        image = detectOT.to_detect()
        
        # Save processed image
        path = os.path.join(settings.BASE_DIR, 'media/images/processed_image/result.jpg')
        cv2.imwrite(path, image)

        # Path of processed image
        pImage = settings.IMAGE_ULR

        # Delete old photos from the database
        Handler.objects.exclude(pk=imageId).delete()
        end = time.time()
        # Initiate a flag
        odet = True

        timetaken = end - start
        return render(request, 'detect/home.html', {'odet': odet, 'pImage': pImage, 'oImage': imageObject, 'time': timetaken})
    else:
        return render(request, 'detect/home.html')


def ocr(request):
    if request.method == 'POST':
        # Grab an image id
        start = time.time()
        imageId = Handler.objects.order_by('-id')[:1]

        # Get the image object
        imageObject = Handler.objects.get(id=imageId)
        # Get the path of an image
        image = imageObject.image.path

        # Extract texts
        config = ('-l eng --oem 1 --psm 3')
        texts = pytesseract.image_to_string(image, config=config)

        # Create a flag
        ocr = True
        #
        end = time.time()

        timetaken = end-start
        # Delete old photos from the database
        Handler.objects.exclude(pk=imageId).delete()
        print(image)
        return render(request, 'detect/home.html', {'ocr': ocr, 'oimage': imageObject, 'texts': texts, 'time':timetaken})
    else:
        return render(request, 'detect/home.html')


def save(request):
    if request.method == 'POST':
        # Gran inputs
        title = request.POST['title']
        texts = request.POST['text']

        if title and texts:
            data = Text(title=title, text=texts)
            data.save()
            success = 'scanned texts successful saved to the database'
            return render(request, 'detect/home.html', {'success': success})
    else:
        return render(request, 'detect/home.html')
