from django import forms
from .models import Handler

class HandlerForm(forms.ModelForm):
    class Meta:
        model = Handler
        fields = ('image',)
