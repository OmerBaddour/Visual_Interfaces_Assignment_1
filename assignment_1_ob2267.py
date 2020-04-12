import cv2
import numpy as np

# TO USE: input sequences of images (list of elements of below classes) on line 325

# images captured using front camera of iPhone 8
# 3088 px x 2320 px
# gcd(3088, 2320) = 16, can resize using img = cv2.resize(img, (193, 145))

# class: [images_in_class]
# key: TL = top left, BR = bottom right, S = splayed, F = fist
# one hand:
class_tl_s = [cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TL_S\\TL_S_1.jpg", 1),
              cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TL_S\\TL_S_2.jpg", 1),
              cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TL_S\\TL_S_3.jpg", 1),
              cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TL_S\\TL_S_4.jpg", 1)]
class_tl_f = [cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TL_F\\TL_F_1.jpg", 1),
              cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TL_F\\TL_F_2.jpg", 1),
              cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TL_F\\TL_F_3.jpg", 1),
              cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TL_F\\TL_F_4.jpg", 1)]
class_tr_s = [cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TR_S\\TR_S_1.jpg", 1),
              cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TR_S\\TR_S_2.jpg", 1),
              cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TR_S\\TR_S_1.jpg", 1),
              cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TR_S\\TR_S_4.jpg", 1)]
class_tr_f = [cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TR_F\\TR_F_1.jpg", 1),
              cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TR_F\\TR_F_2.jpg", 1),
              cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TR_F\\TR_F_3.jpg", 1),
              cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TR_F\\TR_F_4.jpg", 1)]
class_bl_s = [cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BL_S\\BL_S_1.jpg", 1),
              cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BL_S\\BL_S_2.jpg", 1),
              cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BL_S\\BL_S_3.jpg", 1),
              cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BL_S\\BL_S_4.jpg", 1)]
class_bl_f = [cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BL_F\\BL_F_1.jpg", 1),
              cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BL_F\\BL_F_2.jpg", 1),
              cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BL_F\\BL_F_3.jpg", 1),
              cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BL_F\\BL_F_4.jpg", 1)]
class_br_s = [cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BR_S\\BR_S_1.jpg", 1),
              cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BR_S\\BR_S_2.jpg", 1),
              cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BR_S\\BR_S_3.jpg", 1),
              cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BR_S\\BR_S_4.jpg", 1)]
class_br_f = [cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BR_F\\BR_F_1.jpg", 1),
              cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BR_F\\BR_F_2.jpg", 1),
              cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BR_F\\BR_F_3.jpg", 1),
              cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BR_F\\BR_F_4.jpg", 1)]
class_fp_1 = [cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\FP_1\\FP_1_1.jpg", 1),
              cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\FP_1\\FP_1_2.jpg", 1)]
class_fn_1 = [cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\FN_1\\FN_1_1.jpg", 1),
              cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\FN_1\\FN_1_2.jpg", 1)]

# two hands:
class_tl_tr_s = [cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TL_TR_S\\TL_TR_1.jpg", 1),
                 cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TL_TR_S\\TL_TR_2.jpg", 1)]
class_tl_tr_f = [cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TL_TR_F\\TL_TR_F_1.jpg", 1),
                 cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TL_TR_F\\TL_TR_F_2.jpg", 1)]
class_bl_br_s = [cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BL_BR_S\\BL_BR_S_1.jpg", 1),
                 cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BL_BR_S\\BL_BR_S_2.jpg", 1)]
class_bl_br_f = [cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BL_BR_F\\BL_BR_F_1.jpg", 1),
                 cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BL_BR_F\\BL_BR_F_2.jpg", 1)]
class_tl_br_s = [cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TL_BR_S\\TL_BR_S_1.jpg", 1),
                 cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TL_BR_S\\TL_BR_S_2.jpg", 1)]
class_tl_br_f = [cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TL_BR_F\\TL_BR_F_1.jpg", 1),
                 cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TL_BR_F\\TL_BR_F_2.jpg", 1)]
class_bl_tr_s = [cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BL_TR_S\\BL_TR_S_1.jpg", 1),
                 cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BL_TR_S\\BL_TR_S_2.jpg", 1)]
class_bl_tr_f = [cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BL_TR_F\\BL_TR_F_1.jpg", 1),
                 cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BL_TR_F\\BL_TR_F_2.jpg", 1)]
class_tl_tr_f_s = [cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TL_TR_F_S\\TL_TR_F_S_1.jpg"),
                   cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TL_TR_F_S\\TL_TR_F_S_2.jpg")]
class_tl_tr_s_f = [cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TL_TR_S_F\\TL_TR_S_F_1.jpg"),
                   cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TL_TR_S_F\\TL_TR_S_F_2.jpg")]
class_bl_br_f_s = [cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BL_BR_F_S\\BL_BR_F_S_1.jpg"),
                   cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BL_BR_F_S\\BL_BR_F_S_2.jpg")]
class_bl_br_s_f = [cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BL_BR_S_F\\BL_BR_S_F_1.jpg"),
                   cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BL_BR_S_F\\BL_BR_S_F_2.jpg")]
class_bl_tr_f_s = [cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BL_TR_F_S\\BL_TR_F_S_1.jpg"),
                   cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BL_TR_F_S\\BL_TR_F_S_2.jpg")]
class_bl_tr_s_f = [cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BL_TR_S_F\\BL_TR_S_F_1.jpg"),
                   cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\BL_TR_S_F\\BL_TR_S_F_2.jpg")]
class_tl_br_f_s = [cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TL_BR_F_S\\TL_BR_F_S_1.jpg"),
                   cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TL_BR_F_S\\TL_BR_F_S_2.jpg")]
class_tl_br_s_f = [cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TL_BR_S_F\\TL_BR_S_F_1.jpg"),
                   cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\TL_BR_S_F\\TL_BR_S_F_2.jpg")]
class_fp_2 = [cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\FP_2\\FP_2_1.jpg", 1),
              cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\FP_2\\FP_2_2.jpg", 1)]
class_fn_2 = [cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\FN_2\\FN_2_1.jpg", 1),
              cv2.imread("C:\\Users\\Omer Baddour\\PycharmProjects\\Visual_Interfaces\\src\\Assignment_1\\hand_gestures\\FN_2\\FN_2_2.jpg", 1)]

# use trackbars to find optimal HSV parameter values for hand recognition
def nothing(x):
    pass

def is_skin(list_class_names):

    cv2.namedWindow("Tracking")
    cv2.moveWindow("Tracking", 100, 200)
    cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)       # found optimal = 0
    cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)       # found optimal = 25
    cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)       # found optimal = 178
    cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)     # found optimal = 30
    cv2.createTrackbar("US", "Tracking", 255, 255, nothing)     # found optimal = 90
    cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)     # found optimal = 255

    # resize all images in each class_name in list_class_names and convert them to HSV representation
    imgs = []
    hsv_imgs = []
    for class_name in list_class_names:
        for img in class_name:
            imgs.append(cv2.resize(img, (193, 145)))
            hsv_imgs.append(cv2.cvtColor(imgs[-1], cv2.COLOR_BGR2HSV))

    while True:

        # update parameter values from trackbar values
        # lower bounds
        l_h = cv2.getTrackbarPos("LH", "Tracking")
        l_s = cv2.getTrackbarPos("LS", "Tracking")
        l_v = cv2.getTrackbarPos("LV", "Tracking")
        # upper bounds
        u_h = cv2.getTrackbarPos("UH", "Tracking")
        u_s = cv2.getTrackbarPos("US", "Tracking")
        u_v = cv2.getTrackbarPos("UV", "Tracking")
        # arrays of lower and upper bound values
        l_b = np.array([l_h, l_s, l_v])
        u_b = np.array([u_h, u_s, u_v])

        # create masks (binaries) of each hsv_image using parameter values
        masks = []
        for hsv_img in hsv_imgs:
            masks.append(cv2.inRange(hsv_img, l_b, u_b))

        # show all outputs
        x = 450
        y = 0
        for i in range(len(imgs)):
            cv2.imshow(str(i) + ".1", imgs[i])
            cv2.moveWindow(str(i) + ".1", x, y)
            x += 200
            cv2.imshow(str(i) + ".2", masks[i])
            cv2.moveWindow(str(i) + ".2", x, y)
            if (y + 180) > 500:
                y = 0
                x += 200
            else:
                y += 180
                x -= 200

        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()

# get binary images of skin regions using optimal HSV parameter values found with is_skin function
def get_skin_binaries(list_img_seq):

    # resize all images in each class_name in list_class_names and convert them to HSV representation
    imgs = []
    hsv_imgs = []
    for img in list_img_seq:
        imgs.append(cv2.resize(img, (193, 145)))
        hsv_imgs.append(cv2.cvtColor(imgs[-1], cv2.COLOR_BGR2HSV))

    # arrays of lower and upper bound values
    l_b = np.array([0, 25, 178])
    u_b = np.array([30, 90, 255])

    # create masks (binaries) of each hsv_image using parameter values
    masks = []
    for hsv_img in hsv_imgs:
        masks.append(cv2.inRange(hsv_img, l_b, u_b))

    return imgs, masks


# classify each image in list output of get_skin_binaries with (what, where) identifier
def classify_binaries(list_binary_img_seq, list_original_img_seq):

    img_seq_labels = []

    # construct new binary image to filter for hands (get rid of central face)
    dim_list = list(list_binary_img_seq[0].shape)
    dim_tup = tuple(dim_list)
    filter_hands = np.zeros(dim_tup, np.uint8)  # all black

    y = [int(0.05 * dim_list[0]), int(0.475 * dim_list[0]), int(0.525 * dim_list[0]), int(0.9 * dim_list[0])]
    x = [int(0.1 * dim_list[1]), int(0.35 * dim_list[1]), int(0.65 * dim_list[1]), int(0.9 * dim_list[1])]

    filter_hands = cv2.rectangle(filter_hands, (x[0], y[0]), (x[1], y[1]), (255, 255, 255), -1)
    filter_hands = cv2.rectangle(filter_hands, (x[2], y[0]), (x[3], y[1]), (255, 255, 255), -1)
    filter_hands = cv2.rectangle(filter_hands, (x[0], y[2]), (x[1], y[3]), (255, 255, 255), -1)
    filter_hands = cv2.rectangle(filter_hands, (x[2], y[2]), (x[3], y[3]), (255, 255, 255), -1)

    for i in range(len(list_binary_img_seq)):
        # blur image to reduce number of blobs found with connected components algorithm
        # without blur too sensitive to noise - extraneous blobs and unconnected blobs which should be connected
        list_binary_img_seq[i] = cv2.blur(list_binary_img_seq[i], (5, 5))

        # compute bitwise and with each binary image and the hand filter to get binaries of only hands
        hands = cv2.bitwise_and(list_binary_img_seq[i], filter_hands)

        ret, labelled_image = cv2.connectedComponents(hands, connectivity=8)

        # store labels of sufficient size to be either fist or splayed hand
        hand_labels_likely = []

        area = labelled_image.shape[0] * labelled_image.shape[1]

        for label in range(np.max(labelled_image) + 1):
            # find area of blob by counting all occurrences of label
            count = 0
            for val in np.nditer(labelled_image):
                if val == label:
                    count += 1
            if count < (area * 0.125) and count > (area * 0.02):
                hand_labels_likely.append(label)

        if len(hand_labels_likely) == 0:
            list_original_img_seq[i] = cv2.putText(list_original_img_seq[i], "no label", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1)
            cv2.imshow(str(i), list_original_img_seq[i])

        # center of mass calculations
        # store {label : [com_x, weighted_x_sum_so_far, sum_x_so_far, cur_col, max_col, com_y, weighted_y_sum_so_far, sum_y_so_far, cur_row, max_row]} in dictionary
        label_com_dict = {}
        for label in hand_labels_likely:
            label_com_dict[label] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for row_index in range(len(labelled_image)):
            for col_index in range(len(labelled_image[row_index])):
                for label in hand_labels_likely:
                    if labelled_image[row_index][col_index] == label:
                        label_com_dict[label][8] += 1
            for label in hand_labels_likely:
                label_com_dict[label][6] += row_index * label_com_dict[label][8]
                label_com_dict[label][7] += label_com_dict[label][8]
                label_com_dict[label][9] = max(label_com_dict[label][8], label_com_dict[label][9])
                label_com_dict[label][8] = 0
        for label in hand_labels_likely:
            label_com_dict[label][5] = label_com_dict[label][6] / label_com_dict[label][7]

        labelled_image = np.transpose(labelled_image)

        for col_index in range(len(labelled_image)):
            for row_index in range(len(labelled_image[col_index])):
                for label in hand_labels_likely:
                    if labelled_image[col_index][row_index] == label:
                        label_com_dict[label][3] += 1
            for label in hand_labels_likely:
                label_com_dict[label][1] += col_index * label_com_dict[label][3]
                label_com_dict[label][2] += label_com_dict[label][3]
                label_com_dict[label][4] = max(label_com_dict[label][3], label_com_dict[label][4])
                label_com_dict[label][3] = 0
        for label in hand_labels_likely:
            label_com_dict[label][0] = label_com_dict[label][1] / label_com_dict[label][2]
            # check COM is within one of the white square quadrants of hands
            if x[0] < label_com_dict[label][0] < x[1] and y[0] < label_com_dict[label][5] < y[1]:
                # top left
                if label_com_dict[label][4] < label_com_dict[label][9]:
                    # fist
                    img_seq_labels.append([i, "fist top left"])
                    list_original_img_seq[i] = cv2.putText(list_original_img_seq[i], "fist", (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1)
                    list_original_img_seq[i] = cv2.putText(list_original_img_seq[i], "top", (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1)
                    list_original_img_seq[i] = cv2.putText(list_original_img_seq[i], "left", (10, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1)
                    cv2.imshow(str(i), list_original_img_seq[i])
                else:
                    # splayed
                    img_seq_labels.append([i, "splayed top left"])
                    list_original_img_seq[i] = cv2.putText(list_original_img_seq[i], "splayed", (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1)
                    list_original_img_seq[i] = cv2.putText(list_original_img_seq[i], "top", (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1)
                    list_original_img_seq[i] = cv2.putText(list_original_img_seq[i], "left", (10, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1)
                    cv2.imshow(str(i), list_original_img_seq[i])
            elif x[0] < label_com_dict[label][0] < x[1] and y[2] < label_com_dict[label][5] < y[3]:
                # bottom left
                if label_com_dict[label][4] < label_com_dict[label][9]:
                    # fist
                    img_seq_labels.append([i, "fist bottom left"])
                    list_original_img_seq[i] = cv2.putText(list_original_img_seq[i], "fist", (10, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1)
                    list_original_img_seq[i] = cv2.putText(list_original_img_seq[i], "bottom", (10, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1)
                    list_original_img_seq[i] = cv2.putText(list_original_img_seq[i], "left", (10, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1)
                    cv2.imshow(str(i), list_original_img_seq[i])
                else:
                    # splayed
                    img_seq_labels.append([i, "splayed bottom left"])
                    list_original_img_seq[i] = cv2.putText(list_original_img_seq[i], "splayed", (10, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1)
                    list_original_img_seq[i] = cv2.putText(list_original_img_seq[i], "bottom", (10, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1)
                    list_original_img_seq[i] = cv2.putText(list_original_img_seq[i], "left", (10, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1)
                    cv2.imshow(str(i), list_original_img_seq[i])
            elif x[2] < label_com_dict[label][0] < x[3] and y[0] < label_com_dict[label][5] < y[1]:
                # top right
                if label_com_dict[label][4] < label_com_dict[label][9]:
                    # fist
                    img_seq_labels.append([i, "fist top right"])
                    list_original_img_seq[i] = cv2.putText(list_original_img_seq[i], "fist", (95, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1)
                    list_original_img_seq[i] = cv2.putText(list_original_img_seq[i], "top", (95, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1)
                    list_original_img_seq[i] = cv2.putText(list_original_img_seq[i], "right", (95, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1)
                    cv2.imshow(str(i), list_original_img_seq[i])
                else:
                    # splayed
                    img_seq_labels.append([i, "splayed top right"])
                    list_original_img_seq[i] = cv2.putText(list_original_img_seq[i], "splayed", (95, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1)
                    list_original_img_seq[i] = cv2.putText(list_original_img_seq[i], "top", (95, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1)
                    list_original_img_seq[i] = cv2.putText(list_original_img_seq[i], "right", (95, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1)
                    cv2.imshow(str(i), list_original_img_seq[i])
            elif x[2] < label_com_dict[label][0] < x[3] and y[2] < label_com_dict[label][5] < y[3]:
                # bottom right
                if label_com_dict[label][4] < label_com_dict[label][9]:
                    # fist
                    img_seq_labels.append([i, "fist bottom right"])
                    list_original_img_seq[i] = cv2.putText(list_original_img_seq[i], "fist", (95, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1)
                    list_original_img_seq[i] = cv2.putText(list_original_img_seq[i], "bottom", (95, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1)
                    list_original_img_seq[i] = cv2.putText(list_original_img_seq[i], "right", (95, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1)
                    cv2.imshow(str(i), list_original_img_seq[i])
                else:
                    # splayed
                    img_seq_labels.append([i, "splayed bottom right"])
                    list_original_img_seq[i] = cv2.putText(list_original_img_seq[i], "splayed", (95, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1)
                    list_original_img_seq[i] = cv2.putText(list_original_img_seq[i], "bottom", (95, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1)
                    list_original_img_seq[i] = cv2.putText(list_original_img_seq[i], "right", (95, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1)
                    cv2.imshow(str(i), list_original_img_seq[i])
            else:
                print("COM not in one of four quadrants for image " + str(i))

    return img_seq_labels


if __name__ == "__main__":

    # feed in sequences here
    imgs_resized, binaries = get_skin_binaries([class_tl_s[0]]) # list of images read in beginning lines of program

    predictions = classify_binaries(binaries, imgs_resized)

    # print string form of final gesture classifications of images
    image_id = predictions[0][0]
    print(str(image_id) + ": ", end="")
    for elt in predictions:
        if elt[0] == image_id:
            print(str(elt[1]))
        else:
            image_id = elt[0]
            print("\n" + str(elt[0]) + ": " + str(elt[1]))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
