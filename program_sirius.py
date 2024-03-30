import cv2
import numpy as np
import statistics

def apply_yolo_object_detection(image):
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (608, 608), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)
    class_indexes, class_scores, boxes = ([] for i in range(3))
    objects_count = 0


    for out in outs:
        for obj in out:
            scores = obj[5:]
            class_index = np.argmax(scores)
            class_score = scores[class_index]
            if class_score > 0:
                center_x, center_y, obj_width, obj_height = map(int, obj[:4] * [width, height, width, height])
                box = [center_x - obj_width // 2, center_y - obj_height // 2, obj_width, obj_height]
                boxes.append(box)
                
                class_indexes.append(class_index)
                class_scores.append(float(class_score))

    chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)
    for box_index in chosen_boxes:
        box = boxes[box_index]
        class_index = class_indexes[box_index]
        if classes[class_index] in classes_to_look_for:
            objects_count += 1
            image = draw_object_bounding_box(image, class_index, box)
        
    final_image, per_count = draw_queue_area(image, boxes)
    final_image = draw_object_count(final_image, objects_count,per_count)
    
    return final_image

def draw_object_bounding_box(image, index, box):
    x, y, w, h = box
    start = (x, y)
    end = (x + w, y + h)
    color = (0, 255, 0)
    width = 2
    final_image = cv2.rectangle(image, start, end, color, width)

    start = (x, y - 10)
    font_size = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 2
    text = classes[index]
    final_image = cv2.putText(final_image, text, start, font,font_size, color, width, cv2.LINE_AA)

    return final_image

def draw_queue_area(image, boxes):
    s_boxes = sorted(boxes, key=lambda x: x[0])
    aver_boxes = []
    for i in s_boxes:
        aver_boxes.append(abs(i[0]+i[1]))
    mean_value = statistics.mean(aver_boxes)
    std_deviation = statistics.stdev(aver_boxes)
    ny_boxes = [s_boxes[x] for x in range(len(aver_boxes)) if abs(aver_boxes[x] - mean_value) <= std_deviation*0.5 ]
    abc = len(ny_boxes)
    ny_boxes = np.array(ny_boxes,dtype=np.int32)
    ny_boxes = ny_boxes.reshape((-1,1,2))
    if(len(ny_boxes)>6):
        boom = cv2.convexHull(ny_boxes)
        extLeft = tuple(boom[boom[:, :, 0].argmin()][0])
        extRight = tuple(boom[boom[:, :, 0].argmax()][0])
        extTop = tuple(boom[boom[:, :, 1].argmin()][0])
        extBot = tuple(boom[boom[:, :, 1].argmax()][0])
        pts = [extTop,extLeft,extBot,extRight]
        pts = np.array(pts, np.int32)
        cv2.circle(image, extLeft, 8, (0, 0, 255), -1)
        cv2.circle(image, extRight, 8, (0, 255, 0), -1)
        cv2.circle(image, extTop, 8, (255, 0, 0), -1)
        cv2.circle(image, extBot, 8, (255, 255, 0), -1)
        final_image = cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        return final_image, abc
    else:
        final_image = image
        return final_image , 0




def draw_object_count(image, objects_count, obj_time):
    start = (10, 120)
    font_size = 1.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 3
    text = "Objects found: " + str(objects_count)
    time1 = (35 * obj_time) // 60 
    text1 = "Time: " + str(time1) + ' min' 
    white_color = (255, 255, 255)
    black_outline_color = (0, 0, 0)
    final_image = cv2.putText(image, text, start, font, font_size, black_outline_color, width * 3, cv2.LINE_AA)
    final_image = cv2.putText(final_image, text, start, font, font_size, white_color, width, cv2.LINE_AA)
    start_time_text_position = (10, 200)
    final_image = cv2.putText(final_image, text1, start_time_text_position, font, font_size, white_color, width, cv2.LINE_AA)
    
    return final_image

def start_video_object_detection(video: str):
    video_camera_capture = cv2.VideoCapture(video)
    while video_camera_capture.isOpened():
        ret, frame = video_camera_capture.read()
        if not ret:
            break
        frame = apply_yolo_object_detection(frame)
        frame = cv2.resize(frame, (1920 // 2, 1080 // 2))
        cv2.imshow("Video Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_camera_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    net = cv2.dnn.readNetFromDarknet("Resources/yolov4-tiny.cfg", "Resources/yolov4-tiny.weights")
    layer_names = net.getLayerNames()
    out_layers_indexes = net.getUnconnectedOutLayers()
    out_layers = [layer_names[index - 1] for index in out_layers_indexes]
    with open("Resources/coco.names.txt") as file:
        classes = file.read().split("\n")
    video = input("Path to video (or URL): ")
    classes_to_look_for = ['person']
    start_video_object_detection(video)
