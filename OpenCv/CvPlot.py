import cv2

def CvShow():
    # read image
    img = cv2.imread('D:/my-image.png')

    # show image
    cv2.imshow('Example - Show image in window', img)

    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()  # destroys the window showing image

if __name__ == '__main__':
    CvShow()