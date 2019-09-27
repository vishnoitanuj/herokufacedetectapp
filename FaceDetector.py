import cv2
import copy
import numpy

class Detector:

    def face_detect(self,imName):
        # Load the training image
        image1 = cv2.imread('./images/face.jpeg')

        image2 = cv2.cvtColor(numpy.array(imName), cv2.COLOR_BGR2RGB)
        # Convert the training image to RGB
        training_image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

        # Convert the query image to RGB
        query_image = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        # Convert the training image to gray scale
        training_gray = cv2.cvtColor(training_image, cv2.COLOR_BGR2GRAY)

        # Convert the query image to gray scale
        query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)

        # Set the parameters of the ORB algorithm by specifying the maximum number of keypoints to locate and
        # the pyramid decimation ratio
        orb = cv2.ORB_create(5000, 2.0)

        # Find the keypoints in the gray scale training and query images and compute their ORB descriptor.
        # The None parameter is needed to indicate that we are not using a mask in either case.  
        keypoints_train, descriptors_train = orb.detectAndCompute(training_gray, None)
        keypoints_query, descriptors_query = orb.detectAndCompute(query_gray, None)

        # Create copies of the query images to draw our keypoints on
        query_img_keyp = copy.copy(query_image)

        # Create a Brute Force Matcher object. We set crossCheck to True so that the BFMatcher will only return consistent
        # pairs. Such technique usually produces best results with minimal number of outliers when there are enough matches.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

        # Perform the matching between the ORB descriptors of the training image and the query image
        matches = bf.match(descriptors_train, descriptors_query)

        # The matches with shorter distance are the ones we want. So, we sort the matches according to distance
        matches = sorted(matches, key = lambda x : x.distance)

        # Connect the keypoints in the training image with their best matching keypoints in the query image.
        # The best matches correspond to the first elements in the sorted matches list, since they are the ones
        # with the shorter distance. We draw the first 85 mathces and use flags = 2 to plot the matching keypoints
        # without size or orientation.
        result = cv2.drawMatches(training_gray, keypoints_train, query_gray, keypoints_query, matches[:85], query_gray, flags = 2)

        # we display the image
        # plt.title('Best Matching Points', fontsize = 30)
        # cv2.imwrite('result.png',result)
        img = cv2.imencode('.jpg', result)[1].tobytes()
        return img
        # plt.show()

        # Print the number of keypoints detected in the training image
        # print("Number of Keypoints Detected In The Training Image: ", len(keypoints_train))

        # # Print the number of keypoints detected in the query image
        # print("Number of Keypoints Detected In The Query Image: ", len(keypoints_query))

        # # Print total number of matching Keypoints between the training and query images
        # print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))