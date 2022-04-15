import cv2
import numpy as np
import matplotlib.pyplot as plt


# import sys
# sys.setrecursionlimit(100000)


def curve_search(curve_map, point):
    """
    Curve Search
    :param curve_map: single pixel wide line, can have some branch(will return multiple line)
    :param point: curve start point(get from "find_end_point" function)
    :return: [[line1], [line2]]  line: [pt1, pt2, ..., ptn]
    """
    position_guide = [[-1, -1], [-1, 0], [-1, 1], [0, -1],
                      [0, 1], [1, -1], [1, 0], [1, 1]]

    curve_path = list()
    point_list = [point]
    h, w = curve_map.shape[0: 2]
    print(curve_map.shape)
    curve_map[point[0], point[1]] = 0
    forward = True
    while True:
        count = 0
        for idx, pos in enumerate(position_guide):
            near_point = [point[0] + pos[0], point[1] + pos[1]]
            if 0 <= near_point[0] < h and 0 <= near_point[1] < w and \
                    curve_map[near_point[0], near_point[1]] > 0:
                forward = True
                point_list.append(point)
                curve_map[near_point[0], near_point[1]] = 0
                cv2.imshow('test', curve_map)
                cv2.waitKey(30)
                point = near_point
                break
            else:
                count += 1
        if count == len(position_guide) and forward:
            curve_path.append(point_list)
            del point_list[-1]
            point = point_list[-1]
            forward = False

        if curve_map.sum() == 0:
            curve_path.append(point_list)
            break

    return curve_path


def curve_dfs(curve_point, img_w, img_h, point, point_list, line_list, parent_position=-1):
    """
    This function can't work because recursion depth limitation of python,
     may there have another way to make it work?
    :param curve_point:
    :param img_w:
    :param img_h:
    :param point:
    :param point_list:
    :param line_list:
    :param parent_position:
    :return:
    """
    point_list.append(point)

    position_guide = [[-1, -1], [-1, 0], [-1, 1], [0, -1],
                      [0, 1], [1, -1], [1, 0], [1, 1]]

    position_idx = [8, 7, 6, 5, 3, 2, 1, 0]
    line_end = False
    count = 0
    print("middle")
    for idx, pos in enumerate(position_guide):
        print(point, pos)

        p = [point[0] + pos[0], point[1] + pos[1]]
        p_in_curve = p in curve_point
        print(len(point_list))
        if p_in_curve and idx != parent_position:
            line_end, point_list_out, line_list = curve_dfs(curve_point, img_w, img_h, p, point_list, line_list,
                                                            parent_position=position_idx[idx])
            if line_end:
                line_list.append(point_list_out)
            line_end = False
        else:
            count += 1
    if count == len(position_guide):
        line_end = True
    return line_end, point_list, line_list


def find_end_point(input_image):
    """this function get for stackoverflow """
    kernel_0 = np.array((
        [-1, -1, -1],
        [-1, 1, -1],
        [-1, 1, -1]), dtype="int")

    kernel_1 = np.array((
        [-1, -1, -1],
        [-1, 1, -1],
        [1, -1, -1]), dtype="int")

    kernel_2 = np.array((
        [-1, -1, -1],
        [1, 1, -1],
        [-1, -1, -1]), dtype="int")

    kernel_3 = np.array((
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, -1]), dtype="int")

    kernel_4 = np.array((
        [-1, 1, -1],
        [-1, 1, -1],
        [-1, -1, -1]), dtype="int")

    kernel_5 = np.array((
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, -1, -1]), dtype="int")

    kernel_6 = np.array((
        [-1, -1, -1],
        [-1, 1, 1],
        [-1, -1, -1]), dtype="int")

    kernel_7 = np.array((
        [-1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]), dtype="int")

    kernel = np.array((kernel_0, kernel_1, kernel_2, kernel_3, kernel_4, kernel_5, kernel_6, kernel_7))
    output_image = np.zeros(input_image.shape)
    for i in np.arange(8):
        out = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel[i, :, :])
        output_image = output_image + out

    return output_image


if __name__ == "__main__":
    img = cv2.imread("test.png", 0)
    img_end = find_end_point(img)
    # cv2.imshow("test1", img)
    # cv2.imshow("test2", img_end)
    # cv2.waitKey(0)
    coord = np.argwhere(img_end == 255)
    coord_x = coord[:, 1]
    coord_y = coord[:, 0]

    idx = np.argsort(coord_x)
    first_point = [coord_y[idx[0]], coord_x[idx[0]]]
    # _, _, lines = curve_dfs(curve_list, w, h, first_point, list(), list())
    lines = curve_search(img, first_point)
    print(lines)
    lines = np.asarray(lines[0])  # use first line, also can use max length line for your application
    coord_x = lines[:, 1]
    coord_y = lines[:, 0]

    """using for polynomial fit"""
    z1 = np.polyfit(coord_x, coord_y, 8)
    p1 = np.poly1d(z1)
    y_vals = p1(coord_x)
    plot1 = plt.plot(coord_x, coord_y, '*', label='original values')
    plot2 = plt.plot(coord_x, y_vals, 'r', label='polyfit values')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.legend(loc=4)
    plt.title('polyfitting')
    plt.show()
    # cv2.imshow('test', img_half)
    # cv2.waitKey(0)
