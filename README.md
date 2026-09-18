# Face, Age & Gender Detection (OpenCV + CNN)

Detects your face live from the webcam and overlays an estimated **age range**
and **gender**, using OpenCV's DNN module with pretrained CNNs (no training
required, no GPU required).

- **Face detection**: OpenCV's built-in SSD face detector (ResNet-10 backbone)
- **Age prediction**: CNN trained by Levi & Hassner, outputs one of 8 age
  buckets: `(0-2) (4-6) (8-12) (15-20) (25-32) (38-43) (48-53) (60-100)`
- **Gender prediction**: CNN trained by Levi & Hassner, outputs `Male` / `Female`

> Age is inherently hard to guess precisely from a photo (lighting, makeup,
> expression, camera angle all affect it), which is why the model predicts a
> *range* rather than an exact number.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Download the model weights

Two files come pre-included in `models/` (the face detector and the network
architecture files). The two largest weight files (~90 MB each) are fetched
separately by running:

```bash
python download_models.py
```

After this, `models/` should contain 6 files:

```
models/
├── opencv_face_detector.pbtxt
├── opencv_face_detector_uint8.pb
├── age_deploy.prototxt
├── age_net.caffemodel        <- downloaded by the script above
├── gender_deploy.prototxt
└── gender_net.caffemodel     <- downloaded by the script above
```

The script verifies each file's SHA-256 checksum after downloading, so a
partial or corrupt download is caught immediately rather than showing up later
as a confusing OpenCV error.

**If the script fails**, just download the two files directly in your browser
and save them into the `models/` folder:

- https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel
- https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel

These come from the original authors' own repository. Each is about 45 MB.
Make sure the saved filenames are exactly `age_net.caffemodel` and
`gender_net.caffemodel` — Windows browsers sometimes append `.txt`.

## 3. Run it

```bash
python main.py
```

A window will open showing your webcam feed with a green box around each
detected face, labeled with predicted gender/age and confidence, e.g.:

```
Male (94%), (25-32) (61%)
```

**Controls**
- `q` — quit
- `s` — save a snapshot of the current frame to `snapshots/`

Use a different camera:

```bash
python main.py --camera 1
```

## How it works

1. Every frame from the webcam is passed through the face-detector SSD
   network, which returns bounding boxes for each face found.
2. Each detected face is cropped (with a little padding), resized to
   227×227, and mean-subtracted — the standard preprocessing these CNNs
   expect.
3. That preprocessed crop is passed through the gender network (2-class
   softmax) and the age network (8-class softmax); the highest-scoring class
   in each becomes the prediction, and its softmax probability becomes the
   confidence shown on screen.
4. The box and text are drawn onto the frame and displayed.

## Notes & limitations

- These are general-purpose demo models (not fine-tuned on you specifically),
  so predictions — especially age — can be off by a bucket or two. That's
  expected and normal for this kind of model.
- Works best with decent, even lighting and a face that's reasonably close to
  and facing the camera.
- Multiple faces in frame are all detected and labeled independently.
- Everything runs locally — no data is sent anywhere.

## Project structure

```
age_gender_project/
├── main.py              # webcam loop: detect faces, predict age/gender, draw overlay
├── download_models.py   # fetches the two large .caffemodel weight files
├── requirements.txt
├── README.md
├── models/               # network architecture + weight files
└── snapshots/            # created automatically when you press 's'
```
