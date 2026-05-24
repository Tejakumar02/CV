import time
import usb.core
import usb.util
import cv2

# def button_check():

dev = usb.core.find(idVendor=0x1a86, idProduct=0x7523)
interface = 0
endpoint = dev[0][(0, 0)][0]
if dev.is_kernel_driver_active(interface) is True:
    # tell the kernel to detach
    dev.detach_kernel_driver(interface)
    # claim the device
    usb.util.claim_interface(dev, interface)


def capture_image_ds():
    while True:
        try:
            data = dev.read(endpoint.bEndpointAddress, endpoint.wMaxPacketSize)
            if data.tolist() == [65, 13, 10]:
                print("Button pressed")
                vid = cv2.VideoCapture(
                    "rtsp://sourab:tvsm123!@192.168.1.101/stream2")
                # print("h2")
                ret, frame = vid.read()
                time.sleep(1)
                ret, frame = vid.read()
                # print("h3",ret)
                if ret == False:
                    print("Unable to capture image")
                    raise Exception("Unable to capture image")
                else:
                    print("Image captured")
                    return frame
            else:
                print("Not pressed")
            # time.sleep(0.)
        except Exception as e:
            # print("error")
            # print(e.args)
            continue
