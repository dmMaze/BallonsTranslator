import cv2, os, time
import numpy as np


def calculate_derivatives(gx, gy):
    mag = np.sqrt(gx*gx + gy*gy)
    if mag==0:
        return False, -1, -1
    else:
        return True, gx / mag, gy / mag

def sw_calculator(mask, canny_img, gradient_x, gradient_y, show_process=False):
    height, width = canny_img.shape[0], canny_img.shape[1]

    if show_process:
        drawborder = np.zeros((canny_img.shape[0], canny_img.shape[1], 3), dtype=np.uint8)

    pnts = np.where(np.logical_and(canny_img != 0, mask!=0))
    total_pnt_num = pnts[0].shape[0]
    sample_pnt_num = 150
    sample_step = total_pnt_num / sample_pnt_num if total_pnt_num > sample_pnt_num else 1

    cur_pnt_ind = 0
    ray_list = []
    
    while cur_pnt_ind < total_pnt_num:
        start_x, start_y = pnts[1][cur_pnt_ind], pnts[0][cur_pnt_ind]
        ray_arr = [start_x, start_y, -1, -1, -1]
        valid, dx, dy = calculate_derivatives(gradient_x[start_y][start_x], gradient_y[start_y][start_x])

        if valid:
            inc = 0.2
            cur_x, cur_y = start_x + inc * dx, start_y + inc * dy
            while (True):
                tmp_curx, tmp_cury = int(cur_x), int(cur_y)
                if tmp_curx < 0 or tmp_curx >= width or tmp_cury <= 0 or tmp_cury >= height:
                    break
                if canny_img[tmp_cury][tmp_curx] == 0:
                    valid, dx_t, dy_t = calculate_derivatives(gradient_x[tmp_cury][tmp_curx], gradient_y[tmp_cury][tmp_curx])
                    if not valid:
                        break
                    if np.arccos(-dx * dx_t + -dy * dy_t) < np.pi / 2.0:
                        ray_arr[2] = tmp_curx
                        ray_arr[3] = tmp_cury
                        ray_arr[4] = np.sqrt((start_x - tmp_curx)**2 + (start_y - tmp_cury)**2)
                    break
                cur_x += dx
                cur_y += dy
            if ray_arr[2] != -1:
                ray_list.append(ray_arr)
                if show_process:
                    drawborder = cv2.arrowedLine(drawborder, (ray_arr[0], ray_arr[1]), (ray_arr[2], ray_arr[3]), 
                                                    (0, 255, 0), 1)

        cur_pnt_ind += sample_step
        cur_pnt_ind = int(round(cur_pnt_ind))
    if show_process and len(ray_list) != 0:
        ray_list.sort(key=lambda x: x[4])
        cv2.imshow("border", drawborder)
        cv2.imshow("cannyimg", canny_img)
        cv2.waitKey(0)
    return ray_list

def strokewidth_check(text_mask, labels, num_labels, stats, debug_type=0):
    rays_width = []
    height, width = text_mask.shape[0], text_mask.shape[1]
    
    blur_img = cv2.dilate(text_mask ,(3,3),cv2.BORDER_DEFAULT)
    
    # canny_img = cv2.Canny(cv2.dilate(text_mask, (3,3), 1), 170, 320, L2gradient=True, apertureSize=3)
    
    _, canny_img = cv2.threshold(text_mask, 1, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    blur2 = blur_img.astype(float) / 255
    gradient_x = cv2.Scharr(blur2, ddepth=-1, dx=1, dy=0)
    gradient_x = cv2.GaussianBlur(gradient_x ,(3, 3),cv2.BORDER_DEFAULT)
    gradient_y = cv2.Scharr(blur2, ddepth=-1, dx=0, dy=1)
    gradient_y = cv2.GaussianBlur(gradient_y ,(3, 3),cv2.BORDER_DEFAULT)

    img_area = text_mask.shape[0] * text_mask.shape[1]
    show_process = True if debug_type > 0 else False
    for lab in range(num_labels):
        stat = stats[lab]
        if lab != 0 and stat[4] > img_area * 0.002:
            x1, y1, x2, y2 = stat[0] - 2, stat[1] - 2, stat[0] + stat[2] + 2, stat[1] + stat[3] + 2
            x1, x2 = max(x1, 0), min(x2, width)
            y1, y2 = max(y1, 0), min(y2, height)
            labcord = np.where(labels==lab)
            labcord2 = (labcord[0] - y1, labcord[1] - x1)
            text_roi = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
            text_roi[labcord2] = 255
            text_roi = cv2.GaussianBlur(text_roi ,(3,3), cv2.BORDER_DEFAULT)
            ray_list = sw_calculator(text_roi,
                                    canny_img[y1: y2, x1: x2],
                                    gradient_x[y1: y2, x1: x2],
                                    gradient_y[y1: y2, x1: x2],
                                    show_process=show_process)
            if len(ray_list) != 0:
                ray_list.sort(key=lambda x: x[4])
                rays_width.append([int(lab), ray_list[int(len(ray_list)/2)][4]])
    
    if len(rays_width) != 0:
        rays_width = np.array(rays_width)
        mean_width = np.mean(rays_width[:, 1])
        ma = np.int0(rays_width[:, 0])
        mean_area = np.mean(stats[ma][:, 4])

        false_labels = np.where(rays_width[:, 1] > 2*mean_width)[0]
        false_labels = rays_width[false_labels, 0].astype(np.int32)
        for fl in false_labels:
            if stats[fl][4] > 2 * mean_area:
                text_mask[np.where(labels==fl)] = 0
    return text_mask

