# data_analysis/forms.py

from django import forms
from .models import DataSet

class UploadDatasetForm(forms.ModelForm):
    class Meta:
        model = DataSet
        fields = ['data_file']
