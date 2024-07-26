import time
import cv2
import sys
import argparse

import matplotlib.pyplot as plt

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision


def visualize_boundig_boxes(image, predictions):
    for prediction in predictions.detections:
        category = prediction.categories[0]
        category_name = category.category_name
        if category_name == 'person':
            bounding_box_prediction = prediction.bounding_box
            cv2.rectangle(
                image,
                (bounding_box_prediction.origin_x, bounding_box_prediction.origin_y),
                (bounding_box_prediction.origin_x + bounding_box_prediction.width, bounding_box_prediction.origin_y + bounding_box_prediction.height),
                (255,0,0),
                2
            )
    return image

def place_text(image, text , value, position):
    text = f'{text}' + " {:.2f}".format(value)
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,0,0),
        2
    )

def detect(model, TPU, camera=0, width=1200, height=900):
    
    start_test = False
    start_time_for_test = time.time()
    test_x = []
    test_y = []
    #real_x = [0,2,4,6,8,10,12,14,16,18]
    #real_y = [0,2,10,20,22,23,20,17,20,21]
    real_x = [0,3,6,9,12,15,18,21,24,27,30,33]
    real_y = [1,16,24,26,11,11,10,15,7,7,4,1]
    median = 0
    counter = 0
    write = False
    passed = 0
    
    video = cv2.VideoCapture(camera)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    base_options = core.BaseOptions(
        file_name=model, use_coral=TPU
    )
    detection_options = processor.DetectionOptions(
         category_name_allowlist = ['person'],
         score_threshold=0.23
    )
    vision_options = vision.ObjectDetectorOptions(
        base_options=base_options, detection_options=detection_options
    )
    vision_detector = vision.ObjectDetector.create_from_options(vision_options)

    fps_value = 0
    while video.isOpened():
        start_time = time.time()
        is_status_true, image = video.read()
        if is_status_true==False:
            sys.exit(
                'ERROR: webcam error'
            )
    
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_input_tensor = vision.TensorImage.create_from_array(rgb_image)
        result = vision_detector.detect(image_input_tensor)
        image_output = visualize_boundig_boxes(image, result)
        
        info_box_start_y = 30
        info_box_span = 30
        end_time = time.time()
        fps_value = 1 / (end_time - start_time)
        place_text(image, 'FPS', fps_value, (10,info_box_start_y))
        latency_value = (end_time - start_time) * 1000
        place_text(image, 'Latency [ms]', latency_value, (10,(info_box_start_y + info_box_span)))
        number_of_detection = []
        number_of_detection = len([number_of_detection.append(prediction) for prediction in result.detections if prediction.categories[0].category_name == 'person'])
        place_text(image, 'Detections', number_of_detection, (10,(info_box_start_y + (info_box_span * 2))))
        time_from_start = int(end_time - start_time_for_test)
        place_text(image, 'Time', time_from_start, (10,(info_box_start_y + (info_box_span * 3))))
        if cv2.waitKey(1) == 27:
            break
        if time_from_start >= 20 and (time_from_start % 3) == 0:
            median += number_of_detection
            counter += 1
        if time_from_start >= 20 and (time_from_start % 3) == 1:
                median_detection = median / counter
                write = True
        if time_from_start >= 20 and (time_from_start % 3) == 2 and write == True:
            test_x.append(3*passed + 2)
            test_y.append(median_detection)
            write = False
            median = 0
            counter = 0
            passed += 1

        cv2.imshow('People detection', image_output)

    video.release()
    cv2.destroyAllWindows()
    plt.plot(test_x,test_y, color='r', label='Camera')
    plt.plot(real_x,real_y, color='g', label='Real')
    plt.xlabel('time [s]')
    plt.ylabel('Detections')
    plt.title(model)
    plt.legend()
    plt.show()
    
        
def main():

    arg_Parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_Parser.add_argument(
        '--model',
        help='Model path',
        required=False,
        default='efficientdet_lite0.tflite'
    )
    arg_Parser.add_argument(
        '--TPU',
        help='Enable Edge TPU',
        required=False,
        default=False
    )
    args = arg_Parser.parse_args()

    detect(args.model, bool(args.TPU))


if __name__ == '__main__':
    main()

