from ultralytics import YOLO
 
model = YOLO("/home/aa/workspace/runs/detect/Fall/059/0318_3w_detr_fallv2/weights/best.pt") 
# model.export(format = "onnx")  # export the model to onnx format
model.export(format = "engine")  # export the model to engine format
