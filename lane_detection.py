import cv2
import numpy as np
import matplotlib.pyplot as plt


road_image = './media/solidWhiteRight.jpg'
road_video = './media/solidWhiteRight.mp4'

def process_image(image):
  '''
  - Converts image to grayscale
  - Applies a gaussian blur to image to reduce noise
  - Cannies the image to detect edges 
  '''
  if image is not None:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
  '''
  - Defines a triangle that captures the lane we are focused on
  - Masks that region over the image to show only parts that are within the triangle 
  '''
  if image is not None:
    height = image.shape[0]
    lane = np.array([(200, height), (960, height), (450, 250)])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, np.int32([lane]), 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
  '''
  Loops through the lines given by the hough lines function 
  and adds them to an empty black image with the same shape as the given road image
  '''
  line_image = np.zeros_like(image)
  if lines is not None:
    for x1, y1, x2, y2 in lines:
      if abs(x1) > image.shape[1] or abs(y1) > image.shape[0] or abs(x2) > image.shape[1] or abs(y2) > image.shape[0]:
        continue
      else:
        cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 10)
  return line_image
  

def make_coordinates(image, line_parameters):
  '''
  Given an array of a slope and intercept of a line, the function converts it to an array of 2 coordinates:
  the starting point and ending point of the line
  '''
  slope, intercept = line_parameters
  y1 = image.shape[0]
  y2 = int(y1*(0.6))
  x1 = int((y1 - intercept)/slope)
  x2 = int((y2 - intercept)/slope)

  return np.array([x1, y1, x2, y2])

def avg_slope_intercept(image, lines):
  '''
  Loops through the lines given by the hough lines function 
  and groups them into a list by which side the line is on. 
  
  Then, calculates the average slopes and intercepts of the two lists to get
  an singular average line for both sides

  '''
  left_line = []
  right_line = []
  try:
    for line in lines:
      x1, y1, x2, y2 = line.reshape(4)
      parameters = np.polyfit((int(x1), int(x2)), (int(y1), int(y2)), 1)
      slope = parameters[0]
      intercept = parameters[1]
      if slope < 0:
        left_line.append((slope, intercept))
      else:
        right_line.append((slope, intercept))
    left_line_avg = np.average(left_line, axis=0)
    right_line_avg = np.average(right_line, axis=0)

    left_line = make_coordinates(image, left_line_avg)
    right_line = make_coordinates(image, right_line_avg)
    return left_line, right_line
  except:
    return None

def detect_lane_image(image):
  '''
  Detects the lane for an image
  '''
  image = cv2.imread(image)
  lane_image = np.copy(image)
  canny = process_image(lane_image)
  cropped_image = region_of_interest(canny)
  lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
  avg_lines = avg_slope_intercept(lane_image, lines)
  line_image = display_lines(lane_image, avg_lines)
  line_overlay = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
  cv2.imshow('road', line_overlay)
  cv2.waitKey(0)

def detect_lane_video(video):
  '''
  Detects the lane for a video
  '''
  cap = cv2.VideoCapture(video)
  while(cap.isOpened()):
    _, frame = cap.read()
    if frame is not None:
      canny = process_image(frame)
      cropped_image = region_of_interest(canny)
      lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
      avg_lines = avg_slope_intercept(frame, lines)
      line_image = display_lines(frame, avg_lines)
      line_overlay = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
      cv2.imshow('road', line_overlay)
    if cv2.waitKey(1) == ord('q'):
      break
  cap.release()
  cv2.destroyAllWindows

if __name__ == '__main__':
  
  # detect_lane_image(road_image)
  detect_lane_video(road_video)

