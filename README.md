# Argus

**Multi-Camera Threat Intelligence System**

I built Argus as a real-time multi-camera threat intelligence platform that fuses RGB and thermal video streams, runs five parallel machine learning models per camera, and delivers live annotated feeds with per-person threat scores to an operational dashboard over WebSocket. I designed the entire system to run on consumer Apple Silicon hardware without cloud inference, a discrete CUDA GPU, or any commercial data dependencies.

---

Conventional surveillance systems analyse video frames in isolation. A person reappearing at a second camera is treated as a new entity, temporal behaviour patterns are invisible within a single frame, and erratic movement sequences go undetected without sequence-level modelling. For security teams, bridging these gaps typically requires enterprise video analytics platforms costing tens of thousands of dollars annually, most of which still do not integrate cross-camera identity linking with behavioural classification in a single pipeline.

I engineered Argus to solve all three gaps in one unified inference stack. Cross-camera identity is maintained using 512-dimensional L2-normalised appearance embeddings produced by an illumination-adaptive re-identification network. Behaviour is classified over 16-frame temporal windows, distinguishing loitering from standing and running from walking in ways that single-frame detection cannot. Trajectory sequences are scored against a reconstruction autoencoder trained on normal movement distributions, flagging statistical outliers without requiring any labelled anomaly data. A composite threat score integrating detection confidence, behaviour risk, and trajectory anomaly is computed per track and broadcast to the dashboard in real time.

---

## Who This Is For

* **Security Operations Teams:** Analysts requiring a live multi-camera feed with per-person threat scoring, configurable zone breach detection, and a filterable, acknowledgeable alert log without manual frame review.
* **Computer Vision Researchers:** Engineers studying the integration of object detection, multi-object tracking, appearance-based re-identification, temporal action recognition, and unsupervised anomaly detection within a shared real-time inference context.
* **Machine Learning Engineers:** Developers seeking a reference implementation combining custom PyTorch training loops on Apple Silicon MPS, ECE-based confidence calibration, attention gate sensor fusion, and async WebSocket model serving via FastAPI.
* **Smart City and Infrastructure Teams:** Integrators assessing multi-camera pedestrian analysis pipelines and cross-camera identity linking prior to production hardware procurement.

*(Note: Argus is a research and evaluation platform. It is not certified for operational security deployment, does not produce legally admissible surveillance records, and does not include access control or physical alerting integration.)*

---

## Reference Architecture

```
========================================================================
                           DATA SOURCES
        MOT17 Image Sequences  |  Video Files  |  RTSP Streams
========================================================================
                                  |
                                  v
                     +------------------------+
                     |      Video Reader      |
                     |  OpenCV + img sequence |
                     +------------------------+
                                  |
                                  v
                     +------------------------+
                     |       YOLOv8m          |
                     |    Person Detection    |
                     |      Every Frame       |
                     +------------------------+
                                  |
                                  v
                     +------------------------+
                     |       ByteTrack        |
                     | Two-Stage IoU Matching |
                     |  Persistent Track IDs  |
                     +------------------------+
                                  |
        +-------------------------+-------------------------+
        |                         |                         |
        v                         v                         v
+------------------+    +------------------+    +------------------+
|  OSNet-AIN-x1.0  |    |     X3D-S        |    | LSTM Autoencoder |
| Cross-Camera ReID|    | Temporal Action  |    | Trajectory Score |
| 512-dim Embeddings|   | 16-Frame Clips   |    | 30-Frame Sequence|
| Gallery Matching |    | 5 Behaviour Tags |    | MSE Reconstruct  |
+------------------+    +------------------+    +------------------+
        |                         |                         |
        +-------------------------+-------------------------+
                                  |
                                  v
                     +------------------------+
                     |    Attention Gate      |
                     |  RGB + Thermal Fusion  |
                     |  Sigmoid Spatial Weight|
                     +------------------------+
                                  |
                                  v
                     +------------------------+
                     | Composite Threat Score |
                     |   0.40 x Detection     |
                     |   0.35 x Action Risk   |
                     |   0.25 x Anomaly Norm  |
                     +------------------------+
                                  |
                                  v
                     +------------------------+
                     |   Temperature Scaler   |
                     |  LBFGS / ECE-Optimised |
                     +------------------------+
                                  |
               +------------------+------------------+
               |                                     |
               v                                     v
   +--------------------+               +--------------------+
   |   /ws/stream/{id}  |               |     /ws/alerts     |
   |  Binary JPEG Feed  |               |   JSON Alert Feed  |
   +--------------------+               +--------------------+
               |                                     |
               +------------------+------------------+
                                  |
                                  v
========================================================================
                          REACT DASHBOARD
    Camera Grid (4-up)  |  Alert Panel  |  Analytics  |  Zone Editor
========================================================================
```

---

## Component Allocation

| Library / System | Functional Responsibility |
| :--- | :--- |
| **torchreid (OSNet-AIN-x1.0)** | I used this to build the cross-camera re-identification module. The AIN variant adds illumination-adaptive instance normalisation to the standard OSNet backbone, which was necessary to maintain identity consistency when the same person appears under varying lighting conditions across different cameras. Source installation was required as the published package does not support Python 3.12. |
| **pytorchvideo (X3D-S)** | I implemented the temporal action classifier using X3D-S from Facebook AI Research. The model processes 16-frame clips through cross-dimensional 3D convolutions to classify behaviour over time, which was the only architecture capable of distinguishing loitering from standing and running from walking given their identical single-frame appearance. |
| **ByteTrack (ported)** | I integrated a custom ported implementation of ByteTrack for multi-object tracking. Its two-stage IoU association cascade recovers low-confidence detections as tentative tracks before promoting them, which significantly reduces identity switches and track fragmentation compared to single-threshold trackers. |
| **LSTM Autoencoder (custom PyTorch)** | I designed and trained this from scratch to perform unsupervised trajectory anomaly scoring. The encoder-decoder architecture learns to reconstruct normal 30-frame movement sequences and assigns MSE reconstruction error as an anomaly score, requiring no labelled anomaly examples in the training set. |
| **Temperature Scaling (scipy LBFGS)** | I implemented post-hoc confidence calibration using L-BFGS-B optimisation to minimise Expected Calibration Error on a held-out validation set. This single-parameter correction was necessary because raw neural network confidence scores are systematically overconfident, making a composite threat signal meaningless without it. |
| **Attention Gate (custom CNN)** | I designed a sigmoid spatial attention gate that learns to weight RGB and thermal feature maps independently per spatial region before fusing them. This approach was chosen over naive concatenation because it allows each sensor modality to contribute selectively based on learned reliability rather than equal weighting. |
| **ultralytics (YOLOv8m)** | I deployed YOLOv8m for person detection using the ultralytics framework. The medium variant was selected after evaluating throughput against recall on pedestrian-dense sequences. A fine-tuned variant was abandoned after one-epoch training produced zero detections, and the COCO-pretrained baseline was retained. |
| **FastAPI + Uvicorn** | I built the complete backend API using FastAPI with Uvicorn as the ASGI server. FastAPI's async architecture enabled concurrent WebSocket connections across four simultaneous camera inference pipelines without per-connection thread overhead, which was critical for real-time multi-camera streaming. |
| **MLflow (file-based)** | I instrumented all five training runs using MLflow with the file-based experiment store to avoid a running MLflow server dependency. This captured hyperparameters, epoch-level metrics, and final evaluation scores across OSNet-AIN, X3D-S, LSTM-AE, and the attention gate fusion module in a unified log. |
| **Apple MPS Backend** | I configured all inference and training to target PyTorch's Metal Performance Shaders backend on Apple Silicon. This was the only viable GPU acceleration path available, as CUDA is absent on M-series hardware. The environment flag `PYTORCH_ENABLE_MPS_FALLBACK=1` was required to route unsupported operators such as `avg_pool3d` through CPU fallback automatically. |
| **PostgreSQL + SQLAlchemy + Alembic** | I used PostgreSQL 15 as the persistent storage layer, SQLAlchemy 2.0 as the ORM, and Alembic for schema migration management. JSONB columns were used for variable-length structures including bounding box histories, zone polygon coordinates, and appearance embedding vectors. |
| **React 19 + TypeScript** | I built the entire dashboard frontend using React 19 with TypeScript. The camera grid, live alert panel, analytics charts, and zone polygon editor were implemented as discrete page-level components. WebSocket hooks handle real-time binary frame delivery and JSON alert streaming independently. |

---

## Known Limitations

System transparency is a prerequisite for credible evaluation. The following technical boundaries were in place at time of release:

* **Detection Baseline:** Detection operates on COCO-pretrained weights in a cross-domain setting. The MOTA of 0.18 reflects a deliberate scoping decision: training was constrained to validate the pipeline architecture rather than optimise model performance. Domain-specific fine-tuning on annotated pedestrian sequences is the documented next step. The MOT17 YOLO training labels, training script, and dataset splits are fully prepared and ready for extended runs on higher-compute hardware.

* **Action Recognition Accuracy:** The Top-1 accuracy of 23% reflects two constraints operating simultaneously: a limited training epoch budget and the use of UCF-101 sport-class proxies to approximate surveillance behaviour categories. The X3D-S architecture achieves published Top-1 accuracy above 80% on properly matched datasets. The training infrastructure, corrected class mappings, and dataset pipeline required to reach that target are fully in place.

* **Re-Identification Convergence:** Cross-camera re-identification achieves Rank-1 of 0.74 at 25 training epochs against a planned budget of 60. Published OSNet-AIN performance on Market-1501 reaches 91.2% Rank-1. The gap is a direct function of training time. The training loop including hard-batch triplet loss augmentation is ready for extended runs without code changes.

* **Single-Threaded Inference Per Camera:** Each camera pipeline runs as a single asyncio background task sharing one event loop. Horizontal scaling to a large number of simultaneous streams requires a distributed task queue architecture, which is the documented next infrastructure step and does not require changes to the ML pipeline itself.

* **Thermal Fusion Hardware Dependency:** The attention gate fusion module requires physically paired RGB and thermal cameras capturing the same scene simultaneously. On RGB-only deployments, the fusion stage is bypassed automatically and has no effect on detection, tracking, or anomaly performance.

* **Demo Authentication:** Login credentials are hardcoded for evaluation purposes. A production deployment requires a proper user management system, password hashing with stored secrets, token refresh logic, and CORS origin restriction to known frontend addresses.

---

## Evaluation Results

| Module | Metric | Result |
| :--- | :--- | :--- |
| Detection + Tracking | MOTA | 0.1763 |
| Detection + Tracking | IDF1 | 0.2930 |
| Detection + Tracking | Inference FPS | 30.2 |
| Cross-Camera Re-ID | Rank-1 CMC (Market-1501) | 0.7406 |
| Cross-Camera Re-ID | Rank-5 CMC | 0.8834 |
| Cross-Camera Re-ID | mAP | 0.4959 |
| Action Recognition | Top-1 (UCF-101 proxy) | 0.2337 |
| Trajectory Anomaly | Zigzag vs threshold | 4583.8 vs 2.53 (Pass) |
| Trajectory Anomaly | False positive rate | 4.34% |
| Confidence Calibration | ECE pre-scaling | 0.6066 |
| Confidence Calibration | ECE post-scaling | 0.5656 |
| Live Inference | Simultaneous cameras | 4 x 30fps on Apple M-series |

---

## Local Environment Execution

### Prerequisites

* **Python 3.12.13:** Required for compatibility with the ML dependency chain. Earlier versions have not been tested.
* **Node.js 22.22.3:** Required for the React frontend build toolchain.
* **Docker Desktop:** Required to run the PostgreSQL database service. Download from docker.com and ensure it is running before proceeding.
* **Apple Silicon Mac:** The Metal Performance Shaders inference backend requires an M-series processor. Intel Mac and Linux environments would require replacing MPS references with CPU or CUDA targets.

### 0. Clone the Repository

```bash
git clone https://github.com/goofy-daisy/Argus.git
cd Argus
```

### 1. Start the Database

Open Docker Desktop from your Applications folder and wait until the Docker icon in the menu bar is steady. Then run:

```bash
cd /path/to/Argus
docker compose up -d argus-db
```

### 2. Set Up the Python Environment

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install Cython
pip install --no-build-isolation -e torchreid_src/
```

### 3. Run Database Migrations

```bash
source venv/bin/activate
alembic upgrade head
```

### 4. Place Model Weights

The following weight files must be present in `argus/models/` before starting the server:

```
argus/models/yolov8m.pt
argus/models/osnet_ain_x1_0_market1501.pth
argus/models/x3d_s_argus.pth
argus/models/lstm_autoencoder.pth
argus/models/attention_fusion.pth
```

YOLOv8m weights can be downloaded automatically by running:

```bash
source venv/bin/activate
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
mv yolov8m.pt argus/models/
```

The remaining weights are produced by the training scripts in `scripts/`. Refer to `ARCHITECTURE.md` for training commands.

### 5. Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and set `JWT_SECRET` to a strong random string. The `DATABASE_URL` defaults to `postgresql://localhost:5432/argus` and matches the Docker service configuration.

### 6. Start the Backend

Open Terminal 1:

```bash
cd /path/to/Argus
source venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MLFLOW_ALLOW_FILE_STORE=true
uvicorn argus.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 7. Start the Frontend

Open Terminal 2:

```bash
cd /path/to/Argus/argus-frontend
npm install
npm start
```

### 8. Log In

Navigate to `http://localhost:3000`

```
Username: admin
Password: argus2024
```

Go to **Cameras**, click **Add Camera**, enter a video file path or image sequence directory as the source, save, and click **Start**. The camera feed will appear on the Dashboard within 20 to 30 seconds while models load.

### Verification

```bash
# Confirm the API is alive
curl http://localhost:8000/health

# Confirm authentication is working
curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=argus2024" | python -m json.tool
```

---

## Project Structure

```
Argus/
├── argus/
│   ├── api/
│   │   ├── main.py               FastAPI application, CORS, WebSocket endpoints
│   │   ├── auth.py               JWT HS256 authentication and bcrypt hashing
│   │   ├── models.py             SQLAlchemy ORM models
│   │   ├── schemas.py            Pydantic request and response models
│   │   ├── inference_pipeline.py Five-model pipeline and PipelineRegistry
│   │   ├── websocket_manager.py  Connection manager for stream and alert channels
│   │   └── routers/              REST routers: cameras, tracks, alerts, zones, heatmap
│   ├── detection/                YOLOv8m detector wrapper
│   ├── tracking/                 ByteTrack integration and track proxy
│   ├── reid/                     OSNet-AIN re-identifier and embedding gallery
│   ├── action/                   X3D-S action classifier and clip buffer
│   ├── anomaly/                  LSTM Autoencoder trainer and inference
│   └── fusion/                   Attention gate, temperature scaler, calibration
├── argus-frontend/
│   └── src/
│       ├── pages/                Dashboard, Cameras, Alerts, Analytics, Zones
│       ├── components/           CameraFeed, AlertPanel, ThreatGauge, ZoneEditor
│       ├── hooks/                useVideoStream WebSocket canvas hook
│       └── api/                  Axios client with JWT interceptor
├── scripts/                      Training and evaluation scripts for all five models
├── alembic/                      Database migration versions
├── config.yaml                   System configuration and model paths
├── docker-compose.yml            PostgreSQL and MLflow services
└── .env.example                  Environment variable template
```

---

## License

MIT License. See [LICENSE](LICENSE) for full terms.

---

*Argus was developed as a student research project during undergraduate study. It is not affiliated with or representative of any organisation. Benchmarks are reported on standard public datasets under constrained training conditions and are expected to improve substantially with extended compute resources.*
