# Workshop: From Notebook to Cloud
## Satellite Monitoring with Keras and Google Cloud (GDE Workshop)

Tired of seeing your computer crash when processing heavy satellite images? In this practical workshop, we will transition your AI models from a local environment to Google Cloud. We focus on organizing data in Cloud Storage, packaging code with Docker for portability, and choosing the ideal hardware (NVIDIA L4 GPUs) using cost-effective VMs.

---

## 1. Artifact Registry Setup
The first step is to create a home for our Docker images in the cloud.

**Structure of an Image URL:**
`[REGION]-docker.pkg.dev/[PROJECT_ID]/[REPOSITORY_NAME]/[IMAGE_NAME]:[TAG]`

**Command:**
```bash
gcloud artifacts repositories create workshop-repo \
    --repository-format=docker \
    --location=us-central1 \
    --description="AI Workshop Repository" \
    --project=[PROJECT_ID]
```

##  2. Docker Image Generation

We need to package our Keras model, GDAL, and satellite libraries into a portable container.
Authenticate Docker to GCP

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
```

### Build the Image

If you are on a Mac (M1/M2/M3), you MUST specify the platform architecture, otherwise, the Cloud VM will fail to run the container.

**Standard Build:** 
```bash
docker build -t us-central1-docker.pkg.dev/[PROJECT_ID]/workshop-repo/workshop-app-image:poc-v1 .
```

**Mac (Silicon Apple) Build:** 
```bash
docker build --platform linux/amd64 -t us-central1-docker.pkg.dev/[PROJECT_ID]/workshop-repo/workshop-app-image:poc-v1 .
```

**Push to Cloud** 
```bash
docker push us-central1-docker.pkg.dev/[PROJECT_ID]/workshop-repo/workshop-app-image:poc-v1
```
---

## 3. Handling GPU Quota Issues
If you see the error: `Limit: 0.0 globally (metric: GPUS_ALL_REGIONS)`, it means your project is locked by default for security.

1. In GCP, go to **IAM & Admin > Quotas**.
2. Search for `GPUS_ALL_REGIONS`.
3. Click **Edit Quotas**, set the new limit to 1, and provide a justification (e.g., "Educational Workshop for Satellite AI Monitoring").
4. Wait for the approval email (ussually takes a few minutes to an hour)

---

## 4. Deploying the Inference VM
We will use a **G2-Standard-8** machine with a modern **NVIDIA L4 GPU**.

### Check Machine Availability
If a zone is full, check others using:
```bash
gcloud compute accelerator-types list --filter="name=nvidia-l4" --format="table(zone.basename())"
gcloud compute machine-types list --filter="name=g2-standard-8"
```

### Create the Instance
This command creates the VM, installs the drivers, downloads the model/image, and shuts down automatically after finishing to save costs.

```bash
gcloud compute instances create workshop-inference-poc-v1 \
    --project=[PROJECT_ID] \
    --zone=us-east1-b \
    --machine-type=g2-standard-8 \
    --accelerator=count=1,type=nvidia-l4 \
    --image-family=common-cu129-ubuntu-2204-nvidia-580 \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=100GB \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --metadata=install-nvidia-driver=True,startup-script='#!/bin/bash
      # Execution Log (View this via Serial Port or Cloud Logging)
      exec > >(tee -a /tmp/workshop_log.txt /dev/ttyS0) 2>&1

      echo "=== [START] WORKSHOP POC: $(date) ==="
      
      # 1. Setup Docker
      apt-get update && apt-get install -y docker.io
      systemctl start docker
      
      # 2. Wait for NVIDIA Drivers
      echo "Waiting for GPU..."
      until nvidia-smi; do sleep 5; done
      
      # 3. Registry Auth
      gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
      
      # 4. Download Assets (Model and Static Image)
      mkdir -p /opt/workshop/data
      gsutil cp gs://[PROJECT_ID]/cp-0001-dummy.keras /opt/workshop/data/
      gsutil cp gs://[PROJECT_ID]/s2_daily_grid_1139_20260113T140711_20260113T141316_T21MWM_48034_lzw.tif /opt/workshop/data/
      
      # 5. Run Container with Volume Injection
      echo "Starting Docker Inference..."
      docker run --gpus all \
        -v /opt/workshop/data/cp-0001-dummy.keras:/app/cp-0001-dummy.keras \
        -v /opt/workshop/data/s2_daily_grid_1139_20260113T140711_20260113T141316_T21MWM_48034_lzw.tif:/app/s2_daily_grid_1139_20260113T140711_20260113T141316_T21MWM_48034_lzw.tif \
        us-central1-docker.pkg.dev/[PROJECT_ID]/workshop-repo/workshop-app-image:poc-v1
      
      echo "=== [FINISH] Uploading logs and shutting down... ==="
      gsutil cp /tmp/workshop_log.txt gs://[PROJECT_ID]/logs/log_$(date +%Y%m%d_%H%M%S).txt
      
      # Auto-destruct to avoid unnecessary costs
      shutdown -h now'
```
---

## 5. Instructions for Students (Downloading Assets)
Before running the cloud command or testing locally, make sure you have the repository and the required data assets ready.
1. Clone the repository:
```bash
git clone <YOUR-GITHUB-REPO-URL>
cd <YOUR-REPO-FOLDER>
```
2. Download the Dummy Model and Test Image:
I have provided a dummy model and a static Sentinel-2 image in a public Cloud Storage bucket for you to test the pipeline. Run the following commands in your terminal to download them into your current folder:

```bash
# Download the Keras Dummy Model
gsutil cp gs://workshop-satellite-data/cp-0001-dummy.keras .

# Download the Static Sentinel-2 Image
gsutil cp gs://workshop-satellite-data/s2_daily_grid_1139_20260113T140711_20260113T141316_T21MWM_48034_lzw.tif .
````

(Note: If you don't have gsutil installed locally, you can run these commands directly inside the Google Cloud Shell).
