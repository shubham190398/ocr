import easyocr


def bbox_extract(img):
    reader = easyocr.Reader(['en'])

    coords_list = []
    for t in reader.readtext(img):
        coords_list.insert(0, t[0])
    proper_coords_list = []
    for coords in coords_list:
        proper_coords = []
        for coord in coords:
            proper_coords.append([int(coord[0]), int(coord[1])])
        proper_coords_list.append(proper_coords)

    # height = img.shape[0]
    # width = img.shape[1]
    # bbox_img = img.copy()
    # blank_img = np.ones_like(img) * 255
    # for i, coords in enumerate(proper_coords_list):
    #
    #     cv2.rectangle(bbox_img, coords[0], coords[2], (0, 0, 255), 2)
    #     cv2.putText(blank_img, text_list[i], coords[1], cv2.FONT_HERSHEY_PLAIN,
    #                 2, (0, 0, 255), 2, cv2.LINE_AA)
    #
    # f.close()
    # cv2.imwrite('bbox/' + str(count) + '.png', bbox_img)
    # cv2.imwrite('images/' + str(count) + '.png', img)
    # cv2.imwrite('written_img/' + str(count) + '.png', blank_img)
    # cv2.waitKey(0)

    return proper_coords_list
