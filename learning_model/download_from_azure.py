import os
# Azure storage imports
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

connect_str = 'DefaultEndpointsProtocol=https;AccountName=croppedfacesdataset;AccountKey=DTvc7Q8EQb0XBCUBiaWV/sWOnci1GbjfbMdhUhyzEFqL2EWDxtrZASnrkGfeL/QUyjiyrFF7b4/e+AStb3a86w==;EndpointSuffix=core.windows.net'
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_name = 'imagecaptions'
captions_txt_blob_name = 'custom_captions.txt'
container_client = blob_service_client.get_container_client(container_name)
blob_client = blob_service_client.get_blob_client(container=container_name, blob=captions_txt_blob_name)

def get_azure_custom_captions_txt_file():
    captions_txt_content = blob_client.download_blob().content_as_text()
    with open("data/textFiles/custom_captions.txt", "w") as f:
        f.write(captions_txt_content)

def save_blob(file_name, file_content):
    # Get full path to the file
    
    download_file_path = os.path.join("data/Images", file_name)

    # for nested blobs, create local path as well!
    os.makedirs(os.path.dirname(download_file_path), exist_ok=True)

    with open(download_file_path, "wb") as file:
        file.write(file_content)

get_azure_custom_captions_txt_file()
my_blobs = container_client.list_blobs()
for blob in my_blobs:
    print(blob.name)
    bytes = container_client.get_blob_client(blob).download_blob().readall()




