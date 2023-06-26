from pathling import PathlingContext
from pyspark.sql import SparkSession, functions
from pathling import find_jar
import requests
import wandb
import nibabel as nib
import matplotlib.pyplot as plt
from minio import Minio

# Instantiate SparkSession with necessary configurations
def get_spark_session():
    
    spark = SparkSession.builder.config('spark.jars', find_jar()) \
        .config('spark.jars.packages', 'org.apache.hadoop:hadoop-aws:3.3.4') \
        .config('spark.hadoop.fs.s3a.impl', 'org.apache.hadoop.fs.s3a.S3AFileSystem') \
        .config('spark.hadoop.fs.s3a.endpoint', 'http://minio:9000') \
        .config('spark.hadoop.fs.s3a.aws.credentials.provider', 
                'org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider') \
        .config('spark.hadoop.fs.s3a.access.key', 'minio') \
        .config('spark.hadoop.fs.s3a.secret.key', 'minio123') \
        .config('spark.hadoop.fs.s3a.path.style.access', 'true') \
        .config('spark.executor.memory', '8g') \
        .getOrCreate()

    return spark

# Initialize PathlingContext with the created Spark session and terminology server URL
def get_pathling_context(spark):
    
    pc = PathlingContext.create(spark, 
        terminology_server_url='https://r4.ontoserver.csiro.au/fhir')
    
    return pc

# Function to load and encode resources from NDJSON files
def load_resource(pc, path, resource_type):
    json_resources = pc.spark.read.text(path)
    return pc.encode(json_resources, resource_type)

# Loading the resource data
def load_resources(pc, resources):
    resource_data = {
        resource_type: load_resource(pc, f's3a://coda/fhir/{resource_type}.ndjson', 
                                    resource_type)
        for resource_type in resources
    }
    return resource_data

# Using Spark functions
def extract_patient_id(obj):
    return functions.element_at(
        functions.split(obj.patient.reference, '/'), 
    2)

# Using Spark functions
def extract_subject_id(obj):
    return functions.element_at(
        functions.split(obj.subject.reference, '/'), 
    2)

# Save a dataset to Minio and register as a W&B artifact
def save_artifact(data_frame, project_name, artifact_name, type='dataset'):

    file_uri = 'coda/datasets/%s/%s.parquet' % (project_name, artifact_name)
    data_frame.write.parquet('s3a://' + file_uri, mode='overwrite')

    # Add a reference to the artifact in W&B
    run = wandb.init(project=project_name)
    artifact = wandb.Artifact(artifact_name, type=type)
    artifact.add_reference('s3://' + file_uri)
    run.log_artifact(artifact)

# Save a dataset to Minio and register as a W&B artifact
def save_artifact_from_file(file_path, project_name, artifact_name, type='Object3D'):

    client = Minio("minio:9000", access_key="minio", secret_key="minio123", secure=False)
    bucket_name = 'coda'
    path_in_bucket = 'datasets/%s/nii/%s' % (project_name, artifact_name)
    client.fput_object('coda', path_in_bucket, file_path)
    print(f's3://{bucket_name}/{path_in_bucket}', file_path)
    # Add a reference to the artifact in W&B
    run = wandb.init(project=project_name)
    artifact = wandb.Artifact(artifact_name, type=type)
    artifact.add_reference(f's3://{bucket_name}/{path_in_bucket}')
    run.log_artifact(artifact)

# Query the Orthanc API for studies
def query_orthanc_api(server_url, query):
  
    url = f"{server_url}/studies?{query}"

    try:
        response = requests.get(url, headers={
          'Authorization': 'Basic b3J0aGFuYzpvcnRoYW5jMTIz'
        })
        if response.status_code == 200:
            studies = response.json()
            
            return studies
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

# Query the Orthanc API for series
def query_series_by_name(server_url, study_instance_uid, series_name):

    url = f"{server_url}/studies/{study_instance_uid}/series"

    try:
        response = requests.get(url, headers={
          'Authorization': 'Basic b3J0aGFuYzpvcnRoYW5jMTIz'
        })
        if response.status_code == 200:
            series = response.json()
            for series_data in series:
                if series_data["MainDicomTags"]["SeriesDescription"].lower() == series_name:
                    return series_data
            print(f"Series '{series_name}' not found for StudyInstanceUID '{study_instance_uid}'")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

# Retrieve a series as a NIfTI file
def retrieve_series_nifti(server_url, series_instance_uid):

    url = f"{server_url}/series/{series_instance_uid}/nifti"

    try:
        response = requests.get(url, headers={
          'Authorization': 'Basic b3J0aGFuYzpvcnRoYW5jMTIz'
        })
        if response.status_code == 200:
            return response.content
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

def display_nifti_images(nifti_file):
    # Load the NIfTI file
    nifti = nib.load(nifti_file)

    # Get the image data
    data = nifti.get_fdata()

    # Get the number of slices in the z-axis
    num_slices = data.shape[2]

    # Determine the indices of the first, middle, and last slices
    first_slice_index = 0
    middle_slice_index = num_slices // 2
    last_slice_index = num_slices - 1

    # Retrieve the first, middle, and last slices
    first_slice = data[:, :, first_slice_index]
    middle_slice = data[:, :, middle_slice_index]
    last_slice = data[:, :, last_slice_index]

    # Display the first, middle, and last slices
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(first_slice, cmap='gray')
    axes[0].set_title('First Slice')
    axes[1].imshow(middle_slice, cmap='gray')
    axes[1].set_title('Middle Slice')
    axes[2].imshow(last_slice, cmap='gray')
    axes[2].set_title('Last Slice')

    # Remove the axis labels
    for ax in axes:
        ax.axis('off')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()