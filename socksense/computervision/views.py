from django.shortcuts import render

import os
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from rembg import remove
from PIL import Image
import cv2
import numpy as np
import base64
import io
from rest_framework.decorators import api_view


@api_view(['POST'])
def similarity(request):
    if request.method == 'POST':
        try:
            # JSON 데이터 파싱
            json_data = json.loads(request.body)

            # 첫 번째 byte[] 데이터
            byte_data1 = base64.b64decode(json_data.get('bytes1'))
            nparr1 = np.frombuffer(byte_data1, np.uint8)
            image1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
            # 두 번째 byte[] 데이터
            byte_data2 = base64.b64decode(json_data.get('bytes2'))
            nparr2 = np.frombuffer(byte_data2, np.uint8)
            image2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)
            
            # remove background
            outputImage1 = remove(image1)
            outputImage2 = remove(image2)
            
            similarity_score = compare_images(outputImage1, outputImage2)
            
            # JSON 응답 생성
            response_data = {
                "similarity": similarity_score
            }

            # JSON을 HTTP 응답으로 반환
            return JsonResponse(response_data)
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except FileNotFoundError:
            return JsonResponse({'error': 'Image file not found'}, status=400)
         
def compare_images(image1, image2):
    # 히스토그램 계산
    hist1_blue = cv2.calcHist([image1], [0], None, [256], [1, 256])
    hist2_blue = cv2.calcHist([image2], [0], None, [256], [1, 256])

    hist1_green = cv2.calcHist([image1], [1], None, [256], [1, 256])
    hist2_green = cv2.calcHist([image2], [1], None, [256], [1, 256])

    hist1_red = cv2.calcHist([image1], [2], None, [256], [1, 256])
    hist2_red = cv2.calcHist([image2], [2], None, [256], [1, 256])

    # 히스토그램 비교
    similarity1 = cv2.compareHist(hist1_blue, hist2_blue, cv2.HISTCMP_CORREL)
    similarity2 = cv2.compareHist(hist1_green, hist2_green, cv2.HISTCMP_CORREL)
    similarity3 = cv2.compareHist(hist1_red, hist2_red, cv2.HISTCMP_CORREL)

    return (similarity1 + similarity2 + similarity3) / 3



@api_view(['POST'])
def sockColor(request):
    if request.method == 'POST':
        try:
            # JSON 데이터 파싱
            json_data = json.loads(request.body)

            # byte[] 데이터
            byte_data = base64.b64decode(json_data.get('bytes'))
            nparr = np.frombuffer(byte_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # remove background
            outputImage = remove(image)
            
            h,s,v = get_hsv_value(outputImage)
            
            sock_color = check_color(h,s,v)
            
            # JSON 응답 생성
            response_data = {
                "sockColor": sock_color
            }
            
            # JSON을 HTTP 응답으로 반환
            return JsonResponse(response_data)
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except FileNotFoundError:
            return JsonResponse({'error': 'Image file not found'}, status=400)
            
def get_hsv_value(image):
    # BGR 색상 공간에서 HSV 색상 공간으로 변환
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 이미지의 픽셀 값을 평균 내기
    non_zero_pixels = hsv_image[hsv_image.all(axis=2)]
    average_hue = np.mean(non_zero_pixels[:, 0])
    average_saturation = np.mean(non_zero_pixels[:, 1])
    average_value = np.mean(non_zero_pixels[:, 2])

    return average_hue, average_saturation, average_value

def check_color(h, s, v):
    if h > 125 and s > 180 and v > 130:
        return "빨강"
    elif h < 20 and s > 180 and v > 180:
        return "주황"
    elif h >= 20 and h < 30 and s > 160 and s < 200 and v >195:
        return "노랑"
    elif h > 65 and h < 95 and s > 130 and s < 205 and v > 110:
        return "초록"
    elif h > 105 and h < 130 and s > 160 and s < 220 and v > 50:
        return "파랑"
    elif h > 120 and h < 190 and s > 75 and s < 190 and v > 60 and v < 145:
        return "보라"
    elif s < 90 and v > 141:
        return "흰색"
    elif h > 80 and h < 120 and s < 40 and v > 100 and v <205:
        return "회색"
    elif s < 50 and v < 50:
        return "검정"
    else:
        return "회색"