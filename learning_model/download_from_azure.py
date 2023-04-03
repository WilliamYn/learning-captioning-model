import os
# Azure storage imports
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

connect_str = 'DefaultEndpointsProtocol=https;AccountName=croppedfacesdataset;AccountKey=DTvc7Q8EQb0XBCUBiaWV/sWOnci1GbjfbMdhUhyzEFqL2EWDxtrZASnrkGfeL/QUyjiyrFF7b4/e+AStb3a86w==;EndpointSuffix=core.windows.net'
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_name = 'imagecaptions'
captions_txt_blob_name = 'custom_captions.txt'
container_client = blob_service_client.get_container_client(container_name)
blob_client = blob_service_client.get_blob_client(container=container_name, blob=captions_txt_blob_name)

DATA_DIR = "data"
IMAGES_DIR = os.path.join(DATA_DIR, "Images")
class BlobDownloader:
  def __init__(self, container_client):
    self.my_container = container_client

  def get_azure_custom_captions_txt_file():
    captions_txt_content = blob_client.download_blob().content_as_text()
    with open("data/textFiles/custom_captions.txt", "w") as f:
        f.write(captions_txt_content)

  def save_blob(self, file_name, file_content):
    # Get full path to the file
    download_file_path = os.path.join(IMAGES_DIR, file_name)

    # for nested blobs, create local path as well!
    os.makedirs(os.path.dirname(download_file_path), exist_ok=True)

    with open(download_file_path, "wb") as file:
      file.write(file_content)

  def download_all_blobs_in_container(self):
    my_blobs = self.my_container.list_blobs()
    for blob in my_blobs:
      if blob.name == "custom_captions.txt":
        continue
      
      print(blob.name)
      bytes = self.my_container.get_blob_client(blob).download_blob().readall()
      
      # Save the blob in the data/Images directory
      self.save_blob(blob.name, bytes)

def main():
  downloader = BlobDownloader(container_client)
  downloader.get_azure_custom_captions_txt_file()
  downloader.download_all_blobs_in_container()

if __name__ == "__main__":
  main()
