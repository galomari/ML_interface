from django.db import models

class DataSet(models.Model):
    data_file = models.FileField(upload_to='datasets/')
    date_uploaded = models.DateTimeField(auto_now_add=True)