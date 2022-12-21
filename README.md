# Emotion-ResNet50
using MTCNN drop face and convert to grayscale image for better identification
drop image data
```
python src/align_dataset_mtcnn.py  {input data} {output data} --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25
```
