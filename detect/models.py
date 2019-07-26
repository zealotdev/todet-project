from django.db import models


class Handler(models.Model):
    image = models.ImageField(upload_to='images/uploads/')


class Text(models.Model):
    title = models.CharField(max_length=40, default="Title")
    text = models.TextField()

    def __str__(self):
        return self.title
