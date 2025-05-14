# 🧼 Remove Background from Video using Segment Anything + YOLOv5

This repository provides a powerful pipeline to automatically **remove backgrounds from videos** using a combination of:

- 🎯 [YOLOv5](https://github.com/ultralytics/yolov5) for object detection  
- 🧠 [Segment Anything](https://github.com/facebookresearch/segment-anything) by Meta AI for pixel-accurate segmentation

https://github.com/jonibek95/Remove_background/assets/84657258/96906953-4dfb-42d9-bb53-3da9a8435250

---

## 🔍 Overview

This project processes an input video and returns a version with only the segmented foreground object (e.g., a person swinging a golf club) while removing the background. It works in two phases:

1. **Detection** — Uses YOLOv5 to find bounding boxes for each frame.
2. **Segmentation** — Uses Segment Anything (SAM) to generate fine-grained masks from those bounding boxes.

The output is a masked video that can be used for content creation, augmented reality, or visual analysis.

---

## 🧰 Features

- ⚡ Fast & automated background removal
- 🎥 Works on videos frame-by-frame
- 🖼️ Pixel-accurate segmentation
- 🧠 State-of-the-art SAM + YOLOv5
- 🎯 CUDA support for GPU acceleration

---

## 🗂️ Folder Structure
Remove_background/
├── input/                   # Input videos (e.g., golf6.mp4)
├── output/                  # Output videos
├── yolov5/                  # YOLOv5 model directory
├── segment_anything/        # SAM model directory
├── detect.py                # Main background removal script
├── requirements.txt
├── Background-Removal.ipynb
├── README.md
└── …

---

## ▶️ How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/jonibek95/Remove_background.git
cd Remove_background
2. Install Dependencies
pip install -r requirements.txt
Make sure also to clone and install:
	•	YOLOv5
	•	Segment Anything

3. Download Checkpoints
	•	YOLOv5 weights (e.g., yolov5m.pt) from YOLOv5 Releases
	•	SAM weights (e.g., vit_h.pth) from Meta AI SAM

Please place them in the corresponding folders.

4. Run the Script
python detect.py
This will:
	•	Load input video from ./input/golf6.mp4
	•	Detect person with YOLOv5
	•	Segment each frame with Segment Anything
	•	Save the masked video to ./output/masked_video.avi

📜 License

This project is licensed under the Apache-2.0 License.

💡 Future Enhancements
	•	Add multi-class support
	•	Export transparent PNG sequences
	•	Add webcam real-time segmentation
	•	Build Gradio or Streamlit demo UI
