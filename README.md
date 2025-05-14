# 🧼 Remove Background from Video using Segment Anything + YOLOv5

This repository provides a powerful pipeline to automatically **remove backgrounds from videos** using a combination of:

- 🎯 [YOLOv5](https://github.com/ultralytics/yolov5) for object detection  
- 🧠 [Segment Anything](https://github.com/facebookresearch/segment-anything) by Meta AI for pixel-accurate segmentation

!https://github.com/jonibek95/Remove_background/assets/84657258/96906953-4dfb-42d9-bb53-3da9a8435250

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

```
Remove_background/
├── input/                   # Input videos (e.g., golf6.mp4)
├── output/                  # Output videos
├── yolov5/                  # YOLOv5 model directory
├── segment_anything/        # SAM model directory
├── detect.py                # Main background removal script
├── requirements.txt
├── Background-Removal.ipynb
├── README.md
└── ...
```

---

## ▶️ How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/jonibek95/Remove_background.git
cd Remove_background
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure to also clone and install:

- [YOLOv5](https://github.com/ultralytics/yolov5)
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)

---

### 3. Download Model Checkpoints

- **YOLOv5 weights** (e.g., `yolov5m.pt`) from the [YOLOv5 Releases](https://github.com/ultralytics/yolov5/releases)
- **SAM weights** (e.g., `vit_h.pth`) from the [Segment Anything model page](https://github.com/facebookresearch/segment-anything#model-checkpoints)

Place them in their corresponding folders (`yolov5/` and `segment_anything/`).

---

### 4. Run the Script

```bash
python detect.py
```

This will:

- Load the input video from `./input/golf6.mp4`
- Detect the person using YOLOv5
- Segment each frame using Segment Anything
- Save the masked video to `./output/masked_video.avi`

---

## 📜 License

This project is licensed under the **Apache-2.0 License**.

---

## 💡 Future Enhancements

- [ ] Add multi-class support
- [ ] Export transparent PNG sequences
- [ ] Add real-time webcam segmentation
- [ ] Build a Gradio or Streamlit demo UI

---

## ⭐ Support

If you found this project helpful, please consider giving it a ⭐ and sharing it with others.
