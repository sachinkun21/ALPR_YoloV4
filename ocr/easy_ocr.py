import easyocr
import re

reader = easyocr.Reader(['en'])


def ret_ocr(frame):
    results = reader.readtext(frame, allowlist= 'ABCDEFGHIJKLMNPQRSTUVWXYZ0123456789')
    print(results)

    plate_label = "plate"
    if len(results) > 0:
        if re.match("^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$", results[0][1]):
            plate_label = (results[0][1])


        elif len(plate_label)==8:
            plate_label = (results[0][1])

    return plate_label

#results = ret_ocr('../cropped.jpg')

# print(results)