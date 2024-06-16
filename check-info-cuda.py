import cv2

print("OpenCV version:", cv2.__version__)
print("CUDA available:", cv2.cuda.getCudaEnabledDeviceCount() > 0)

if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print("CUDA devices:")
    for i in range(cv2.cuda.getCudaEnabledDeviceCount()):
        print(cv2.cuda.getDevice(i).getName())
