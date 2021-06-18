import cv2
import numpy as np
import matplotlib.pyplot as plt

def roi(image, vertices):
    mask = np.zeros_like(image)
    # channel_count = image.shape[2]
    # match_mask_color = (255,) * channel_count
    match_mask_color = 255  # we will be passing grayscale image so 
    cv2.fillPoly(mask , vertices, match_mask_color)
    # cv2.imshow('masked', mask) cv2.fillpoly mask other region with 0 and matched region with 1
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
    # region of intrest depends on the camera position 

def make_coordinates(image, line_parameters):
  # this will collect the line_parameters value (the average of all the slops from the average_slope_intercept function and unpack it)
    try:
        slope, intercept = line_parameters
    except:
        slope , intercept = 0.001,0
    y1 = image.shape[0]
    y2 = int(y1*(3/5))# we want to consider the line up to 3/5 of the y axis
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1,y1, x2,y2])

# avaerage_slope_interest function to avereage out the slops and y-intercept
def average_slope_intercept(image,lines):
   
    left_fit =[] # are the lists that collected the coordinates of the average value of the lines on the left
    right_fit=[] # are the lists that collected the coordinates of the average value of the lines on the right
    for line in lines: # loop all the line
        x1,y1,x2,y2 = line.reshape(4) # reshape them into the 4 dimensional array
        parameter = np.polyfit((x1,x2),(y1,y2), 1) # used to fit a first-degree polynomial (a linear function)
        slope=parameter[0] # is collected from the matrix of the parameters
        intercept= parameter[1]
        if slope <0: # value of the slope is -ve for the left side of the line 
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0) # we average out the intercept
    right_fit_average = np.average(right_fit, axis=0)
    left_line= make_coordinates(image, left_fit_average)
    right_line= make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def show_lines(image, lines):
    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            X1,Y1,X2,Y2 = line.reshape(4)
            cv2.line(lines_image, (X1,Y1), (X2,Y2), (255,0,0),10)
    return lines_image


def process(image):
    kernel_size = 1
    lane_line_image = image.copy()
    image = cv2.GaussianBlur(image,(kernel_size, kernel_size),0)
    #roi region of interest 
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]

    region_of_intrest = [
        (0, height), (width/2, height/2.2),(width, height)

    ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, threshold1=100, threshold2=100)
    cropped_image = roi(canny_image, np.array([region_of_intrest], np.int32))
    lines = cv2.HoughLinesP(cropped_image, rho = 2, theta = np.pi/180,threshold = 80,lines= np.array([]),
    minLineLength= 40,maxLineGap=100 )
    average_lines = average_slope_intercept(image, lines)


    line_image = show_lines(image, average_lines)
    combined_image = cv2.addWeighted(lane_line_image, 0.8, line_image, 1, 1)
    
    # rho means the distance from origin to line
    # maxlinegap is the max distance btween two lines to be considered as sinngle line

    # plt.imshow(canny_image)
    # cv2.imshow('new_window', image_with_lines)
    # cv2.imshow('window', cropped_image)
    # cv2.waitKey(0)
    # plt.show()
    return combined_image



cap = cv2.VideoCapture('8.2 test2.mp4')


while cap.isOpened():
    ret, frame = cap.read()

    frame = process(frame)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()