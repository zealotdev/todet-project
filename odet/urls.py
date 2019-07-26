from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from detect import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('upload/', views.upload, name='upload'),
    path('detection', views.detection, name='detection'),
    path('ocr', views.ocr, name='ocr'),
    path('save', views.save,  name='save')

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
