import numpy as np
import cv2 
import argparse
import matplotlib.pyplot as plt
import math
from PIL import Image
from matplotlib.image import imread

#1-Colour Filter
def colorfilter():
    img = Image.open(r"../Images/1_2_rover.jpg").convert("RGB")
    width,height = img.size

    pixels = img.load()

    def red(r,g,b):
        newr = r
        newg = 0
        newb = 0
        return(newr,newg,newb)

    def blue(r,g,b):
        newr = 0
        newg = 0
        newb = b
        return(newr,newg,newb)

    def green(r,g,b):
        newr = 0
        newg = g
        newb = 0
        return(newr,newg,newb)

    def grey(r,g,b):
        newr = (r+g+b)//3
        newg = (r+g+b)//3
        newb = (r+g+b)//3
        return(newr,newg,newb)

    def sepia(r,g,b):
        newr = int((r*.393) + (g*.769) + (b*.189))
        newg = int((r*.349) + (g*.686) + (b*.168))
        newb = int((r*.272) + (g*.534) + (b*.131))
        return(newr,newg,newb)

    choice = '''
    enter your choice
    1 red
    2 blue
    3 green
    4 grey
    5 sepia
    '''


    print(choice)
    no = int(input())


    for py in range(height):
        for px in range(width):
            r,g,b = img.getpixel((px,py))
            if no==1:
                pixels[px,py] = red(r,g,b)
            if no==2:
                pixels[px,py] = blue(r,g,b)
            if no==3:
                pixels[px,py] = green(r,g,b)
            if no==4:
                pixels[px,py] = grey(r,g,b)
            if no==5:
                pixels[px,py] = sepia(r,g,b)

    img.show()
    img.save(r"../Images/1_new_filtering.jpg")

#2-Gray Scaling
def grayscale():
    input_image = imread(r"../Images/1_2_rover.jpg")
    r,g,b = input_image[:,:,0],input_image[:,:,1],input_image[:,:,2]

    #gamma = 1.04

    r_const,g_const,b_const = 0.2126, 0.7152, 0.0722

    grayscale_image = r_const * r  + g_const * g  + b_const + b

    fig = plt.figure(1)
    img1,img2 = fig.add_subplot(121), fig.add_subplot(122)

    img1.imshow(input_image)
    img2.imshow(grayscale_image, cmap=plt.cm.get_cmap('gray'))

    fig.show()
    plt.show()

#3-Black and White
def blackandwhite():
    img = cv2.imread(r"../Images/1_2_rover.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    fig = plt.figure(1)
    img1,img2 = fig.add_subplot(121), fig.add_subplot(122)

    img1.imshow(img)
    img2.imshow(thresh, cmap=plt.cm.get_cmap('gray'))

    fig.show()
    plt.show()

#4-Image Cropping
def crop():
    
    img = cv2.imread('../Images/3_4_turtle.jpg')

    print(img.shape) # Print image shape

    cv2.imshow("original", img)

    # Cropping an image

    cropped_image = img[300:800, 700:1100]

    # Display cropped image

    cv2.imshow("cropped", cropped_image)

    # Save the cropped image

    cv2.imwrite("../Images/4_CroppedImage.jpg", cropped_image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

#5-Image Resizing
def resize():
    try:
        # Read image from disk.
        img = cv2.imread('../Images/10_pig.jpg')
    
        # Get number of pixel horizontally and vertically.

        (height, width) = img.shape[:2]
    
        # Specify the size of image along with interploation methods.

        # cv2.INTER_AREA is used for shrinking, whereas cv2.INTER_CUBIC

        # is used for zooming.

        res = cv2.resize(img, (int(width / 2), int(height / 2)), interpolation = cv2.INTER_CUBIC)
    
        # Write image back to disk.
        cv2.imwrite('../Images/10_result.jpg', res)

        #showing the images
        cv2.imshow('original image',img)
        cv2.imshow('scaled image',res)   
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except IOError:
        print ('Error while reading files !!!') 

#6-Image Rotation
def rotation():
            image = cv2.imread('../Images/7_8_emoji.png')
            rotated_image = naive_image_rotate(image,90,'full')
            cv2.imshow("original image", image)
            cv2.imshow("rotated image",rotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def naive_image_rotate(image, degrees, option='same'):
    '''
    This function rotates the image around its center by amount of degrees
    provided. The rotated image can be of the same size as the original image
    or it can show the full image.
    
    inputs: image: input image (dtype: numpy-ndarray)
            degrees: amount of rotation in degrees (e.g., 45,90 etc.)
            option: string variable for type of rotation. It can take two values
            'same': the rotated image will have same size as the original image
                    It is default value for this variable.
            'full': the rotated image will show the full rotation of original
                    image thus the size may be different than original.
    '''
    # First we will convert the degrees into radians
    rads = math.radians(degrees)
    # Finding the center point of the original image
    cx, cy = (image.shape[1]//2, image.shape[0]//2)
    
    if(option!='same'):
        # Let us find the height and width of the rotated image
        height_rot_img = round(abs(image.shape[0]*math.sin(rads))) + \
                           round(abs(image.shape[1]*math.cos(rads)))
        width_rot_img = round(abs(image.shape[1]*math.cos(rads))) + \
                           round(abs(image.shape[0]*math.sin(rads)))
        rot_img = np.uint8(np.zeros((height_rot_img,width_rot_img,image.shape[2])))
        # Finding the center point of rotated image.
        midx,midy = (width_rot_img//2, height_rot_img//2)
    else:
        rot_img = np.uint8(np.zeros(image.shape))
     
    for i in range(rot_img.shape[0]):
        for j in range(rot_img.shape[1]):
            if(option!='same'):
                x= (i-midx)*math.cos(rads)+(j-midy)*math.sin(rads)
                y= -(i-midx)*math.sin(rads)+(j-midy)*math.cos(rads)
                x=round(x)+cy
                y=round(y)+cx
            else:
                x= (i-cx)*math.cos(rads)+(j-cy)*math.sin(rads)
                y= -(i-cx)*math.sin(rads)+(j-cy)*math.cos(rads)
                x=round(x)+cx
                y=round(y)+cy

            if (x>=0 and y>=0 and x<image.shape[0] and  y<image.shape[1]):
                rot_img[i,j,:] = image[x,y,:]
    return rot_img 

#7-Median Filter
def median_filter():
    path = '../Images/9_person.png'
    img = cv2.imread(path)

    #using medinaBlur() function to remove the noise from the given image

    median = cv2.medianBlur(img, 5)

    compare = np.concatenate((img, median), axis=1) 

    #displaying the noiseless image as the output on the screen

    #side by side comparison

    cv2.imshow('Median Filter', compare)

    cv2.waitKey(0)

    cv2.destroyAllWindows

#8-Box Blur Filter
def box_blur_filter():
    img = cv2.imread('../Images/5_desktop.jpg')
    plt.imshow(img)
    plt.show()
    
    box_blur_ker = np.array([[0.1111111, 0.1111111, 0.1111111],
                        [0.1111111, 0.1111111, 0.1111111],
                        [0.1111111, 0.1111111, 0.1111111]])
    
    # Applying Box Blur effect
    # Using the cv2.filter2D() function
    # src is the source of image(here, img)
    # ddepth is destination depth. -1 will mean output image will have same depth as input image
    # kernel is used for specifying the kernel operation (here, box_blur_ker)
    Box_blur = cv2.filter2D(src=img, ddepth=-1, kernel=box_blur_ker)
    
    # Showing the box blur image using matplotlib library function plt.imshow()
    plt.imshow(Box_blur)
    plt.show()


if __name__=='__main__':
    while True:
        print("\nMenu Driven Program")
        print("1.Colour Filter") 
        print("2.Gray Scaling") 
        print("3.Black and White")
        print("4.Image Cropping") 
        print("5.Image Resizing") 
        print("6.Image Rotation")   
        print("7.Median Filter") 
        print("8.Box Blur Filter")
        print("9.Exit")
        choice=int(input("Enter your choice:"))
        if choice==1:
            colorfilter()
        elif choice==2:
            grayscale()
        elif choice==3:
            blackandwhite()
        elif choice==4:
            crop()
        elif choice==5:
            resize()
        elif choice==6:
            rotation()
        elif choice==7:
            median_filter()
        elif choice==8:
            box_blur_filter()
        elif choice==9:
            break
        else:
            print("Invalid Choice")