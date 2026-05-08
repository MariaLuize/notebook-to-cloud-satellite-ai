import os, sys, rioxarray
import numpy as np,  logging, re, pygeoj, rasterio, json, time
import tensorflow as tf
from patchify import patchify, unpatchify

import matplotlib.pyplot as plt
from osgeo import ogr, gdal
import subprocess

import warnings
import urllib3
# from distributed.client import Future
from packaging.version import parse as parse_version

import os.path
import keras

from model import UNetModel

# Google Cloud
import psutil
from google.cloud import storage


BUCKET_NAME  = "workshop-satellite-data"
EPOCH        = 1
KERNEL_SIZE  = 256
MOSAIC_SCALE = 10
REGION_ID    = 1139
DISTANCE     = 48034
DROPOUT      = 0.3
LOSS         = 'BinaryCrossentropy'
METRICS      = ['RootMeanSquaredError']

sys.stdout.flush()


ROOT_PATH = str(sys.argv[8])

def create_directory(new_folder):
  if not os.path.exists(new_folder):
      print(f'lets make the directory: {new_folder}')
      os.makedirs(new_folder)
  else: return

def extract_image_id(path):
  match = re.search(r's2_daily_grid_\d+_(\d{8}T\d{6}_\d{8}T\d{6}_T\w{5})', path)
  if match:
      return match.group(1)

def env_settings():
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    logging.getLogger('googleapicliet.discovery_cache').setLevel(logging.ERROR)

def gpus_initialise():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"{len(gpus)} Physical GPU(s) found and configured for memory growth.")
        except RuntimeError as e:
            print(e)

def _load_model(optical_bands, optical_indices, dropout, loss, metrics, epoch):
    model_keras = f'{ROOT_PATH}/cp-000{epoch}.keras'  
    model_instance = UNetModel(input_shape=[None, None, len(optical_bands + optical_indices)], dropout_rate=dropout, loss=loss, metrics_list=metrics)
    model          = model_instance.get_model()
    model.load_weights(model_keras)
    return model

def upload_to_gcs_and_clean(local_file_path, gcs_bucket_name, gcs_destination_blob_name):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(gcs_bucket_name)
        blob = bucket.blob(gcs_destination_blob_name)
        
        print(f"Uploading {local_file_path} to gs://{gcs_bucket_name}/{gcs_destination_blob_name}")
        blob.upload_from_filename(local_file_path)
        
        # Apaga o arquivo local para economizar espaço no disco da VM
        os.remove(local_file_path)
        print("Upload complete. Local file removed.")
    except Exception as e:
        logging.error(f"GCS Upload Error: {e}")

def mosaic_predict(mosaic_lzw, region_id, output_path, version, kernel_dim, optical_bands, optical_indices, model, mosaic_scale):
    img_id = extract_image_id(mosaic_lzw)
    with rasterio.open(mosaic_lzw, 'r') as ds:
        arr = ds.read()
    
    print(f'Processing Image: {img_id} | Shape: {arr.shape}')

    arr = np.clip(arr, 0, None).astype(np.float32)
    img_arr = np.array(arr) 
    # print(img_arr.shape)

    ''' FULL BANDS (6) '''
    disired_dims    = kernel_dim*2 
    bands           = optical_bands + optical_indices
    patches         = patchify(img_arr, (len(bands), disired_dims, disired_dims), step=kernel_dim)
    dim             = patches.shape[1]
    patch2          = patches.reshape((1, dim**2, len(bands), disired_dims, disired_dims))
    patch2_reshaped = patch2.reshape((dim**2, -1, len(bands), disired_dims, disired_dims))
    patch3          = np.transpose(patch2_reshaped, [0,1,3,4,2])

    predictions_arr = np.array([])
    step = 32
    for i in range(step,patch3.shape[0],step):
        curr_patch      = tf.data.Dataset.from_tensor_slices(patch3[i-step:i])
        curr_prediction = model.predict(curr_patch, steps=None, verbose=0)
        if i ==step:
            predictions_arr = curr_prediction
        else:
            predictions_arr = np.concatenate((predictions_arr,curr_prediction), axis = 0)

    last_patch           = tf.data.Dataset.from_tensor_slices(patch3[i:])
    last_prediction_part = model.predict(last_patch, steps=None, verbose=0)
    predictions_arr      = np.concatenate((predictions_arr,last_prediction_part), axis = 0)

    # predictions_arr = model.predict(patch3, batch_size=32, verbose=1)

    patchesPerRow  = dim
    TotalPatches   = dim**2
    patchDimension = [disired_dims,disired_dims]

    counter       = 1
    rowCounter    = 1
    globalCounter = 0
    finalArray    = np.array([])
    rowArray = np.array([])
    for raw_record in predictions_arr:
        raw_record = np.squeeze(raw_record)
        rows,cols = raw_record.shape
        limite_esquerda = 128
        limite_direita  = 384
        limite_inferior = 384
        limite_superior = 128
        if rowCounter == 1:
            limite_superior = 0
        if (counter == 1) or (counter == patchesPerRow+1):
            limite_esquerda = 0
        if (counter == patchesPerRow+1) and rowCounter == 1:
            limite_superior = 128

        if counter == patchesPerRow:
            limite_direita = 512

        if rowCounter == (TotalPatches/patchesPerRow) or (rowCounter == (TotalPatches/patchesPerRow)-1 and counter == patchesPerRow+1):  # É O ULTIMA LINHA
            limite_inferior = 512
        raw_record = raw_record[limite_superior:limite_inferior,limite_esquerda:limite_direita]
        if rowCounter == 1:
            finalArray = rowArray
        if counter <= patchesPerRow:
            if counter == 1:
                rowArray = raw_record
            else:
                rowArray = np.concatenate((rowArray,raw_record), axis = 1)
            counter = counter+1
        else:
            counter = 2
            rowCounter = rowCounter+1
            if np.array_equal(finalArray,rowArray):
                finalArray = rowArray
            else:
                finalArray = np.concatenate((finalArray,rowArray),axis=0)
            rowArray = raw_record
        globalCounter = globalCounter+1
    finalArray = np.concatenate((finalArray,rowArray),axis=0)

    rows,cols = finalArray.shape

    scaled_array = (finalArray * 255).astype(np.uint8)
    scaled_array = np.array([scaled_array])

    create_directory(output_path)
    raster_uri     = output_path + '/UNET_v'+version+'_grid_'+str(region_id)+'.tif'
    raster_uri_lzw = output_path + '/outimage_v'+version+'_grid_'+str(region_id)+'_'+img_id+'_lzw.tif'

    with rasterio.open(raster_uri, 'w',
          driver="GTiff",
          height=rows,
          width=cols,
          count=1,
          dtype="uint8",          
          crs='EPSG:3857',
          transform=ds.transform) as dataset:
              dataset.write(scaled_array)
              
    dataset = gdal.Open(raster_uri, gdal.GA_Update)
    band    = dataset.GetRasterBand(1)
    band.SetScale(mosaic_scale)
    dataset = None
    
    subprocess.run(["gdal_translate", "-ot", "Byte", "-of", "GTiff", "-co", "COMPRESS=LZW", "-co", "PREDICTOR=2", "-co", "TILED=YES", raster_uri, raster_uri_lzw])
    os.remove(raster_uri)
    print("C'est finiz\n\n")

def main(args):
    env_settings()

    VERSION, OPTICAL_BANDS, OPTICAL_INDICES = args[0], args[1], args[2]
    OPTICAL_BANDS   = OPTICAL_BANDS.split(',')
    OPTICAL_INDICES = OPTICAL_INDICES.split(',')
    MOSAIC_PATH   = args[3]
    OUTPUT_PATH   = args[4]
    START_DATE    = args[5] # YYYY-MM-DD
    END_DATE      = args[6] # YYYY-MM-DD

    gpus_initialise()

    print('\n\n\nStarting...')
    try:
        loaded_model = _load_model(OPTICAL_BANDS, OPTICAL_INDICES, DROPOUT, LOSS, METRICS, EPOCH)
    except Exception as e:
        print("Failed to load model:", e)
        sys.exit(1)
    
    try:
        print('\n--- Starting Inference Pipeline ---')
        region_mosaic_path = f'{ROOT_PATH}/s2_daily_grid_1139_20260113T140711_20260113T141316_T21MWM_48034_lzw.tif'
        
        if not os.path.exists(region_mosaic_path):
            print(f"Error: Could not find input image at {region_mosaic_path}")
            sys.exit(1)

        mosaic_predict(
            region_mosaic_path, REGION_ID, OUTPUT_PATH, VERSION, KERNEL_SIZE,
            OPTICAL_BANDS, OPTICAL_INDICES, loaded_model, MOSAIC_SCALE)

        img_id = extract_image_id(region_mosaic_path)
        lzw_tif_name = f'outimage_v{VERSION}_grid_{str(REGION_ID)}_{img_id}_lzw.tif'
        local_tif_path = f'{OUTPUT_PATH}/{lzw_tif_name}'
        
        upload_to_gcs_and_clean(local_tif_path, BUCKET_NAME, f"output/{START_DATE}/{lzw_tif_name}")
    except Exception as e:
        logging.error("The error from mosaic_predict is: ",e)

if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        print("The error is __main__: ",e)
    finally:
        sys.exit()
