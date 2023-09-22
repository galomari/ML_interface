from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser, FormParser, MultiPartParser
from rest_framework.decorators import api_view, parser_classes
import pandas as pd
from sklearn.linear_model import LinearRegression  # or other linear models
from sklearn.cluster import KMeans
import os
from django.conf import settings
from data_analysis.models import DataSet  # Import the model you defined earlier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@csrf_exempt
@api_view(['POST'])
@parser_classes([FormParser, MultiPartParser])
def upload_dataset(request):
    if 'dataset' in request.FILES:
        dataset_file = request.FILES['dataset']
        dataset_path = os.path.join(settings.MEDIA_ROOT, dataset_file.name)

        if not os.path.exists(settings.MEDIA_ROOT):
            os.makedirs(settings.MEDIA_ROOT)

        with open(dataset_path, 'wb+') as destination:
            for chunk in dataset_file.chunks():
                destination.write(chunk)

        # After storing the uploaded file, process it
        success, result = process_uploaded_file(dataset_path)

        if not success:
            # If processing was not successful, return an error message
            return JsonResponse({'error': result}, status=400)

        # Save the dataset file path to the database
        data_set = DataSet(data_file=dataset_file.name)
        data_set.save()

        return JsonResponse({'message': 'File uploaded and processed successfully'}, status=200)
    else:
        return JsonResponse({'error': 'No file uploaded'}, status=400)
@csrf_exempt
@api_view(['GET'])
def process_data_endpoint(request):
    # Retrieve the latest uploaded dataset path
    latest_dataset = DataSet.objects.latest('date_uploaded')
    dataset_path = os.path.join(settings.MEDIA_ROOT, latest_dataset.data_file.name)
    
    success, result = process_uploaded_file(dataset_path)
    
    if success:
        return JsonResponse({'message': 'Data processed successfully'}, status=200)
    else:
        return JsonResponse({'error': result}, status=400)

def process_uploaded_file(file_path):
    """
    Process the uploaded dataset:
    - Validate it for the required structure.
    - Transform by handling missing values and scaling.
    - Store the transformed dataset.
    """

    # Expected columns in the dataset
    EXPECTED_COLUMNS = ['Column1', 'Column2']

    try:
        # Reading the dataset
        df = pd.read_csv(file_path, encoding='ISO-8859-1')

        # Validation: Ensure the file has the required columns
        if not set(EXPECTED_COLUMNS).issubset(df.columns):
            return False, "Uploaded dataset doesn't have the expected columns."

        # Transformation:

        # 1. Handle missing values - fill with mean of the column
        imputer = SimpleImputer(strategy='mean')
        df[EXPECTED_COLUMNS] = imputer.fit_transform(df[EXPECTED_COLUMNS])

        # 2. Scale the columns to have a mean of 0 and standard deviation of 1
        scaler = StandardScaler()
        df[EXPECTED_COLUMNS] = scaler.fit_transform(df[EXPECTED_COLUMNS])

        # Storage: Save the processed data back to the same location
        df.to_csv(file_path, index=False, encoding='ISO-8859-1')

        return True, df  # success flag and processed data

    except Exception as e:
        return False, str(e)  # error flag and error message



@csrf_exempt
@api_view(['GET'])
def get_analysis(request):
    try:
        # Retrieve the most recently uploaded dataset path from the database
        latest_dataset = DataSet.objects.latest('date_uploaded')
        dataset_path = os.path.join(settings.MEDIA_ROOT, latest_dataset.data_file.name)
        
        df = pd.read_csv(dataset_path,encoding='ISO-8859-1')
        
        # Preprocess data if you have a preprocessing function
        df.columns = df.columns.str.strip()
        print(df)
        
        # For demonstration, predicting using the mean value of Column1
        value_to_predict = df['Column1'].mean()
        
        # Assuming a supervised learning scenario using Linear Regression on two columns
        # This will predict 'Column2' based on 'Column1'
        X_supervised = df[['Column1']]
        y_supervised = df['Column2']
        model = LinearRegression().fit(X_supervised, y_supervised)
        supervised_prediction = model.predict([[value_to_predict]])
        
        # Assuming an unsupervised learning scenario using KMeans on two columns
        # This will group data into clusters
        X_unsupervised = df[['Column1', 'Column2']]
        kmeans = KMeans(n_clusters=3)  # Assuming 3 clusters for simplicity
        clusters = kmeans.fit_predict(X_unsupervised)
        
        return JsonResponse({
            'supervised_result': supervised_prediction.tolist(),
            'unsupervised_clusters': clusters.tolist()
        })
    
    except DataSet.DoesNotExist:
        return JsonResponse({'error': 'No dataset uploaded yet'}, status=400)