# import os
# from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

# connect_str = 'DefaultEndpointsProtocol=https;AccountName=croppedfacesdataset;AccountKey=DTvc7Q8EQb0XBCUBiaWV/sWOnci1GbjfbMdhUhyzEFqL2EWDxtrZASnrkGfeL/QUyjiyrFF7b4/e+AStb3a86w==;EndpointSuffix=core.windows.net'
# blob_service_client = BlobServiceClient.from_connection_string(connect_str)

# container_name = 'imagecaptions'
# container_client = blob_service_client.get_container_client(container_name)

# local_file_path = './custom_captions.txt'
# blob_name = 'custom_captions.txt'

# with open(local_file_path, "rb") as data:
#     container_client.upload_blob(name=blob_name, data=data, overwrite=True)

# print(f"The file {blob_name} has been uploaded to the {container_name} container.")

import os
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

connect_str = 'DefaultEndpointsProtocol=https;AccountName=croppedfacesdataset;AccountKey=DTvc7Q8EQb0XBCUBiaWV/sWOnci1GbjfbMdhUhyzEFqL2EWDxtrZASnrkGfeL/QUyjiyrFF7b4/e+AStb3a86w==;EndpointSuffix=core.windows.net'
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_name = 'imagecaptions'
captions_txt_blob_name = 'custom_captions.txt'

container_client = blob_service_client.get_container_client(container_name)
blob_client = blob_service_client.get_blob_client(container=container_name, blob=captions_txt_blob_name)
file_contents = blob_client.download_blob().content_as_text()
num_lines = len(file_contents.split("\n"))
print(file_contents)
print(f'The custom_captions.txt file has {num_lines} lines.')
