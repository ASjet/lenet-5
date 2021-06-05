# Set VideoCaptureDeviceID here
camera_id = 0

# Set model index path here
model_path = "model/"

# Set static image path here
img_path = "img/"

def readIndex():
    with open(model_path+"index",'r') as f:
        model_type, model_name = f.read().splitlines()
    return model_type, model_name