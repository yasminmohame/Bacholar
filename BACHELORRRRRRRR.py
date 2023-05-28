import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


# Directory paths containing the images
segmented_directory = r"C:\Users\yasmin\Desktop\Segmented"
original_directory = r"C:\Users\yasmin\Desktop\RGB"
# Set the figure size for maximizing the plots



# Get the list of files in each directory
segmented_files = os.listdir(segmented_directory)
original_files = os.listdir(original_directory)

# Loop over the files in both directories
for segmented_file, original_file in zip(segmented_files, original_files):
    if segmented_file.endswith(".jpg") or segmented_file.endswith(".png"):
        # Construct the full file paths
        segmented_img_path = os.path.join(segmented_directory, segmented_file)
        original_img_path = os.path.join(original_directory, original_file)

        # Load the segmented image and original image
        segmented_img = cv2.imread(segmented_img_path)
        original_img = cv2.imread(original_img_path)

        # Get pixel color at specific coordinate
        x = 512  # x-coordinate
        y = 500  # y-coordinate
        blue = segmented_img[y, x, 0]
        green = segmented_img[y, x, 1]
        red = segmented_img[y, x, 2]

        # Define the color range to track (in BGR format)
        lower_color = np.array([blue, green, red], dtype=np.uint8)
        upper_color = np.array([blue, green, red], dtype=np.uint8)

        # Create a mask based on the color range
        mask = cv2.inRange(segmented_img, lower_color, upper_color)

        # Apply the mask to the input image
        masked_img = cv2.bitwise_and(segmented_img, segmented_img, mask=mask)
        inverted_img = cv2.bitwise_not(masked_img)

        # Perform edge detection on the masked image
        edges = cv2.Canny(masked_img, 50, 150)

        # Perform morphological closing to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Apply probabilistic Hough Line Transform to detect lines
        lines = cv2.HoughLinesP(closed_edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

        # Create a blank image for drawing the connected edges
        connected_edges = np.zeros_like(closed_edges)

        # Connect the detected lines together
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(connected_edges, (x1, y1), (x2, y2), (255), 2)   
        
        
        

    
    
    # Convert the masked image to grayscale
    masked_gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    
    # Create a mask by thresholding the grayscale masked image
    _, mask = cv2.threshold(masked_gray, 1, 255, cv2.THRESH_BINARY)
    
    # Invert the mask
    inverted_mask = cv2.bitwise_not(mask)
    
    # Apply the inverted mask to the original image
    result = cv2.bitwise_and(original_img, original_img, mask=inverted_mask)
    result1 = cv2.bitwise_and(original_img, original_img, mask=mask)
    
    
    
    
    # Plot the input, masked, edges, closed edges, and connected edges images
    fig, axes = plt.subplots(1, 2 )
    # axes[0].imshow(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))
    # axes[0].set_title("Segmented Image")
    # axes[0].axis("off")
    axes[0].imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Masked Image")
    axes[0].axis("off")
    # axes[2].imshow(edges, cmap="gray")
    # axes[2].set_title("Edges")
    # axes[2].axis("off")
    # axes[3].imshow(closed_edges, cmap="gray")
    # axes[3].set_title("Closed Edges")
    # axes[3].axis("off")
    # axes[4].imshow(connected_edges, cmap="gray")
    # axes[4].set_title("Connected Edges")
    # axes[4].axis("off")
    # axes[5].imshow(cv2.cvtColor(inverted_img, cv2.COLOR_BGR2RGB))
    # axes[5].set_title("Inverted image")
    # axes[5].axis("off")
    # axes[2].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    # axes[2].set_title("Original_img")
    # axes[2].axis("off")
    # axes[3].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    # axes[3].set_title("result")
    # axes[3].axis("off")
    # axes[4].imshow(cv2.cvtColor(result1, cv2.COLOR_BGR2RGB))
    # axes[4].set_title("result1")
    # axes[4].axis("off")
    
    
    
    # Display the result
    # cv2.imshow("Result1", result1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # Convert the masked image to grayscale
    # masked_gray = cv2.cvtColor(inverted_img, cv2.COLOR_BGR2GRAY)
    
    # # Create a mask by thresholding the grayscale masked image
    # _, mask = cv2.threshold(masked_gray, 1, 255, cv2.THRESH_BINARY)
    
    # # Invert the mask
    # inverted_mask = cv2.bitwise_not(mask)
    
    # # Apply the inverted mask to the original image
    # result = cv2.bitwise_and(Original_img, Original_img, mask=inverted_mask)
    
    
    # # Display the result
    # cv2.imshow("Result2", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()







# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import os

# # Directory path containing the images
# directory = r"C:\Users\yasmin\Desktop\input"

# # Loop over each file in the directory
# for filename in os.listdir(directory):
#     if filename.endswith(".jpg") or filename.endswith(".png"):
#         # Construct the full file path
#         img_path = os.path.join(directory, filename)

#         # Load the segmented image
#         segmented_img = cv2.imread(img_path)

#         # Get pixel color at specific coordinate
#         x = 512  # x-coordinate
#         y = 500  # y-coordinate
#         blue = segmented_img[y, x, 0]
#         green = segmented_img[y, x, 1]
#         red = segmented_img[y, x, 2]

#         # Define the color range to track (in BGR format)
#         lower_color = np.array([blue, green, red], dtype=np.uint8)
#         upper_color = np.array([blue, green, red], dtype=np.uint8)

#         # Create a mask based on the color range
#         mask = cv2.inRange(segmented_img, lower_color, upper_color)

#         # Apply the mask to the input image
#         masked_img = cv2.bitwise_and(segmented_img, segmented_img, mask=mask)

#         # Perform edge detection on the masked image
#         edges = cv2.Canny(masked_img, 100, 200)

#         # Plot the input, masked, and edges images
#         fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#         axes[0].imshow(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))
#         axes[0].set_title("Segmented Image")
#         axes[0].axis("off")
#         axes[1].imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
#         axes[1].set_title("Masked Image")
#         axes[1].axis("off")
#         axes[2].imshow(edges, cmap="gray")
#         axes[2].set_title("Edges")
#         axes[2].axis("off")

#         # Display the figures
#         plt.show()

#     # Move on to the next image
#     else:
#         continue












# import cv2
# import matplotlib.pyplot as plt


# # Load segmented image
# #segmented_img = cv2.imread(r"C:\Users\yasmin\Desktop\ppp\semantic_img_2023-05-14 09_24_21.928658.jpg")
# segmented_img = cv2.imread(r"C:\Users\yasmin\Desktop\input\semantic_img_2023-05-14 09_24_34.989592.jpg")


# # Get pixel color at specific coordinate
# x = 512# x-coordinate
# y = 500 # y-coordinate

# # Get pixel values (BGR format)
# blue = segmented_img[y, x, 0]
# green = segmented_img[y, x, 1]
# red = segmented_img[y, x, 2]


# # Print pixel color
# BGR =( blue, green, red)
# print("Pixel color (BGR): ", BGR)
# print (blue)
# # Alternatively, you can convert BGR to RGB format
# rgb = (red, green, blue)
# print("Pixel color (RGB): ", rgb)

# import numpy as np
# # Define the color range to track (in RGB format)
# lower_color = np.array([blue, green, red], dtype=np.uint8)
# upper_color = np.array([blue, green, red], dtype=np.uint8)
# #upper_color = np.array([194, 119 , 227 ])
# # Create a mask based on the color range
# mask = cv2.inRange(segmented_img, lower_color, upper_color)

# # Apply the mask to the input image
# masked_img = cv2.bitwise_and(segmented_img, segmented_img, mask=mask)
# # Plot the input and output images
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# axes[0].imshow(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))
# axes[0].set_title("Input Image")
# axes[0].axis("off")
# axes[1].imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
# axes[1].set_title("Output Image")
# axes[1].axis("off")

# plt.show()
# # Display the masked image
# cv2.imshow('Mask',mask)

# cv2.imshow("Color Mask", masked_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

