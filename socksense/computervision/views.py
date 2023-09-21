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


@csrf_exempt
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