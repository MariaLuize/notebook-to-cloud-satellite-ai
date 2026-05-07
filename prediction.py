import os, sys, ee, xarray,rioxarray
import xee, numpy as np,  logging, re, pygeoj, rasterio, json, time
import tensorflow as tf
from datetime import timedelta, date
from patchify import patchify, unpatchify

import matplotlib.pyplot as plt
from osgeo import ogr, gdal
import subprocess

import warnings
import urllib3
from distributed.client import Future
import pprint
from packaging.version import parse as parse_version
import os.path
import keras


# Google Cloud
import psutil
from google.cloud import storage


BUCKET_NAME = "sag-output-us-central1"
UNET_VERSION = '3_DATASET_YA_KA_MU'
EPOCH = 26
KERNEL_SIZE = 256

sys.stdout.flush()



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
    def patched_del(self):
        if not getattr(sys, 'is_finalizing', False):
            try:
                original_del(self)
            except TypeError:
                pass

    original_del   = Future.__del__
    Future.__del__ = patched_del
    warnings.filterwarnings("ignore", category=UserWarning, module='distributed.utils_perf')
    dask.config.set({'logging.distributed': 'error'})
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    http = urllib3.PoolManager(maxsize=50, block=True)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    logging.getLogger('googleapicliet.discovery_cache').setLevel(logging.ERROR)

def gpus_intialise(GPU_AFFINTY, GPU_MEMORY_LIMIT_GB):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[GPU_AFFINTY], 'GPU')
            GPU_MEMORY_LIMIT_GB = GPU_MEMORY_LIMIT_GB * 1e3
            if GPU_MEMORY_LIMIT_GB == 0:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                tf.config.set_logical_device_configuration(gpus[GPU_AFFINTY],[tf.config.LogicalDeviceConfiguration(memory_limit=GPU_MEMORY_LIMIT_GB)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)


def _load_model(optical_bands, optical_indices, dropout, loss, metrics, unet_version, epoch):
    model_keras3 = f'{ROOT_PATH}/checkpoint/v{unet_version}/model_e{epoch}_keras3.keras' 
    model = keras.models.load_model(model_keras3)
    return model


def download_from_stack(id_grid, cur_grid, distance, start_date, end_date, mosaic_creator, optical_bands, optical_indices, mosaic_path, max_retries=500, wait_time=2):
    cur_grid        = ee.Geometry.Polygon(cur_grid)
    center          = ee.Feature(cur_grid).centroid()
    buf             = ee.Feature(center).buffer(distance,0.01)
    buffer_geometry = ee.Geometry(buf.bounds(0.01).geometry())

    ee_start_date = ee.Date(f'{start_date}')
    ee_end_date   = ee.Date(f'{end_date}')

    daily_mosaic_as_imgcol, mosaic_scale, mosaic_id = mosaic_creator.get_mosaic_s2(cur_grid=cur_grid, start_date=ee_start_date, end_date=ee_end_date, bands=optical_bands, indeces=optical_indices)
    if daily_mosaic_as_imgcol is None or mosaic_scale is None or mosaic_id is None:
        logging.warning(f'No data available for grid {id_grid} during the period {start_date} to {end_date} (exclusive), skipping prediction.')
        return None, None

    mosaic_path_lzw = f"{ROOT_PATH}/daily_mosaic/{mosaic_id[:8]}/s2_daily_grid_{id_grid}_{mosaic_id}_{distance}_lzw.tif"
    if os.path.exists(mosaic_path_lzw):
        print('Mosaic already downloaded \n')
        return mosaic_path_lzw, mosaic_scale

    attempts = 0
    while attempts < max_retries:
        try:
            ds_1 = xarray.open_dataset(daily_mosaic_as_imgcol,crs='EPSG:3857',scale=mosaic_scale,geometry=buffer_geometry,chunks={'time':1,'X': 512, 'Y': 512}, engine='ee').isel(time=0) #
            ds_1 = ds_1.rename({'X':'x','Y':'y'}).transpose('y', 'x')

            create_directory(mosaic_path)
            path_tif     = f"{mosaic_path}/s2_daily_grid_{id_grid}_{mosaic_id}_{distance}.tif"
            path_tif_lzw = f"{mosaic_path}/s2_daily_grid_{id_grid}_{mosaic_id}_{distance}_lzw.tif"

            ds_1.rio.to_raster(path_tif)
            # subprocess.run(["gdal_translate", "-of", "GTiff", "-co", "COMPRESS=LZW", "-co", "PREDICTOR=2", "-co", "TILED=YES", path_tif, path_tif_lzw])
            subprocess.run(["gdal_translate", "-ot", "Byte", "-of", "GTiff", "-co", "COMPRESS=LZW", "-co", "PREDICTOR=2", "-co", "TILED=YES", path_tif, path_tif_lzw])
            os.remove(path_tif)
            return path_tif_lzw, mosaic_scale
        except Exception as e:
            print(f"Attempt {attempts + 1} failed with error: {e}")
            attempts += 1
            time.sleep(wait_time)
    logging.error(f"All attempts ({max_retries}) failed to process the mosaic for grid {id_grid}.")
    return None, None

def upload_to_gcs_and_clean(local_file_path, gcs_bucket_name, gcs_destination_blob_name):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(gcs_bucket_name)
        blob = bucket.blob(gcs_destination_blob_name)
        
        print(f"Fazendo upload de {local_file_path} para gs://{gcs_bucket_name}/{gcs_destination_blob_name}")
        blob.upload_from_filename(local_file_path)
        
        # Apaga o arquivo local para economizar espaço no disco da VM
        os.remove(local_file_path)
        print("Upload concluído e arquivo local removido.")
    except Exception as e:
        logging.error(f"Erro ao fazer upload para o GCS: {e}")

def mosaic_predict(mosaic_lzw, region_id, output_path, version, kernel_dim, optical_bands, optical_indices, model, mosaic_scale):
    img_id       = extract_image_id(mosaic_lzw)
    path_segmentation = f'{output_path}/outimage_v{version}_grid_{str(region_id)}_{img_id}_lzw.tif'
    if os.path.exists(path_segmentation):
        print('File already predicted \n')
        return
    with rasterio.open(mosaic_lzw, 'r') as ds:
        arr = ds.read()
    
    print(f'{img_id}:{arr.shape}')

    arr = np.clip(arr, 0, None)
    arr = arr.astype(np.float32)
    img_arr = np.array(arr) 
    print(img_arr.shape)

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
    # driver    = gdal.GetDriverByName("GTiff")

    threshold      = 0.5
    finalArray     = np.array([finalArray])
    binary_array   = np.where(finalArray > threshold, 1, 0) 
    binary_array   = np.array(binary_array.astype(np.float32)) 
    create_directory(output_path)
    raster_uri     = output_path + '/UNET_v'+version+'_grid_'+str(region_id)+'.tif'
    raster_uri_lzw = output_path + '/outimage_v'+version+'_grid_'+str(region_id)+'_'+img_id+'_lzw.tif'

    with rasterio.open(raster_uri,'w',
          driver="GTiff",
          height=rows,
          width=cols,
          count=1,
          dtype="float32",
          crs='EPSG:3857',#mixer["projection"]["crs"],
          transform=ds.transform, #mixer["projection"]["affine"]["doubleMatrix"],
          nodata="nan") as dataset:
              dataset.write(binary_array)
    dataset = gdal.Open(raster_uri, gdal.GA_Update)
    band    = dataset.GetRasterBand(1)
    band.SetScale(mosaic_scale)
    dataset = None
    subprocess.run(["gdal_translate", "-of", "GTiff", "-co", "COMPRESS=LZW", "-co", "PREDICTOR=2", "-co", "TILED=YES", raster_uri, raster_uri_lzw])
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

    gpu_dict    = {'4090':{'GPU_AFFINTY' : 0, 'GPU_MEMORY_LIMIT_GB':8},
                   '2070':{'GPU_AFFINTY':1, 'GPU_MEMORY_LIMIT_GB':8}}
    sel_gpu     = '4090'
    GPU_AFFINTY = gpu_dict[sel_gpu]['GPU_AFFINTY'] #GeForce RTX 2070
    GPU_MEMORY_LIMIT_GB = gpu_dict[sel_gpu]['GPU_MEMORY_LIMIT_GB']
    gpus_intialise(GPU_AFFINTY, GPU_MEMORY_LIMIT_GB)

    DROPOUT       = 0.3
    LOSS          = 'BinaryCrossentropy'
    METRICS       = ['RootMeanSquaredError']
    UNET_VERSION  = '3_DATASET_YA_KA_MU'
    EPOCH         = 26

    KERNEL_SIZE = 256
    GRID        = pygeoj.load(f'{ROOT_PATH}/GRID-ALLCALSSES-COL9.geojson')
    DISTANCE    = 48034


    # GRIDS_LIST = [1139]
    # filtered_grid = [feature for feature in GRID if feature.properties['id'] in GRIDS_LIST]

    n_cores = os.cpu_count()
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    dask_memory = f"{int(ram_gb * 0.75)}G" # Usa 75% da RAM para o Dask, deixa 25% pro OS e GPU
    
    print(f"Iniciando Dask com {n_cores} cores e limite de {dask_memory} por worker.")

    cluster = dd.LocalCluster(
        processes=False,
        n_workers=n_cores,
        threads_per_worker=2,
        memory_limit=dask_memory,
        silence_logs=logging.ERROR
    )

    with dd.Client(cluster) as client:
        ee_initialise()

    init_future = cluster.get_client().submit(ee_initialise)
    print(cluster.dashboard_link)

    try:
        loaded_model   = _load_model(OPTICAL_BANDS, OPTICAL_INDICES, DROPOUT, LOSS, METRICS, UNET_VERSION, EPOCH)
    except Exception as e:
        print("The error load_model is: ",e)

    mosaic_creator = DailyS2MosaicCreator()
    try:
        start           = time.time()
        print('\n\n\nStarting...')
        for region in filtered_grid:
            region_id = int(region.properties['id'])
            print(f'\n================ Region: {region_id} ================')
            try:
                region_mosaic_path, mosaic_scale = download_from_stack(
                    region_id, region.geometry.coordinates[0][0], DISTANCE, START_DATE, END_DATE,
                    mosaic_creator, OPTICAL_BANDS, OPTICAL_INDICES, MOSAIC_PATH
                    )
            except Exception as e:
                logging.error("The error from download_from_stack is: ",e)
                continue
            if region_mosaic_path is None or mosaic_scale is None:
                # logging.warning(f'No data available for Region {region_id}, skipping prediction.')
                continue
            try:
                mosaic_predict(
                    region_mosaic_path, region_id, OUTPUT_PATH, VERSION, KERNEL_SIZE,
                    OPTICAL_BANDS, OPTICAL_INDICES, loaded_model, mosaic_scale)
                
                bucket_name = "sag-output-us-central1"
                img_id = extract_image_id(region_mosaic_path)
                lzw_tif_name = f'outimage_v{VERSION}_grid_{str(region_id)}_{img_id}_lzw.tif'
                local_tif_path = f'{OUTPUT_PATH}/{lzw_tif_name}'
                
                # Envia para a nuvem na pasta da data (ex: /20251108/outimage_v6...)
                upload_to_gcs_and_clean(local_tif_path, bucket_name, f"output/{START_DATE}/{lzw_tif_name}")
            
            except Exception as e:
                logging.error("The error from mosaic_predict is: ",e)
        end = time.time()
        print('Prediction Time per year = '+str(end - start)+'\n')
        # with open(OUTPUT_PATH+'/lock', 'w'):
            # pass
    except Exception as e:
        logging.error("The error from the loop  is: ",e)
    finally:
        client.close()
        cluster.close()

if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        print("The error is __main__: ",e)
    finally:
        sys.exit()
